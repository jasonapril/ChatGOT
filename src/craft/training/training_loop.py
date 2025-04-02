#!/usr/bin/env python
"""
Training Loop Module
===================

This module contains the core training loop logic, separated from the main Trainer class
for better organization and testability.
"""

import torch
import torch.nn.functional as F
import logging
import time
import gc
import contextlib
from typing import Tuple, Dict, Any, Callable, List, Optional
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

# Use relative import for utils within the same package
from ..utils.logging import force_flush_logs, format_time
# from ..utils.memory import MemoryMonitor # Removed unused import
from .callbacks import CallbackList
from ..models.base import GenerativeModel # Import base model for type hinting
from .progress import ProgressTracker  # Import ProgressTracker

class TrainingLoop:
    """Handles the core training loop logic."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        log_interval: int = 10,
        callbacks: Optional[List[Any]] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        # Handle callbacks: Use provided CallbackList or create one
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(callbacks if callbacks is not None else [])
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize scaler for AMP using the recommended method
        # No need to import GradScaler specifically here if torch.amp is sufficient
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

    def train_epoch(
        self,
        current_epoch: int,
        global_step: int, # This is the step count *at the start* of the epoch
        progress: ProgressTracker,
        loaded_global_step: Optional[int] = None
    ) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        epoch_loss = 0.0
        num_valid_steps_in_epoch = 0
        epoch_start_time = time.time()
        total_tokens = 0
        steps_per_epoch = len(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True) # Reset gradients at epoch start

        # Retrieve max_steps from config at the start of the epoch
        max_steps = self.config.get('max_steps')

        # Determine batches to skip if resuming
        resume_batches = 0 # This variable seems unused

        # Batch skipping logic for resuming
        resume_batch_offset = -1
        is_resuming_this_epoch = False
        if loaded_global_step is not None and current_epoch == (loaded_global_step // steps_per_epoch):
            resume_batch_offset = loaded_global_step % steps_per_epoch
            self.logger.info(f"Resuming epoch {current_epoch+1} from batch offset {resume_batch_offset + 1}/{steps_per_epoch} (global step {loaded_global_step + 1})")
            is_resuming_this_epoch = True

        step_token_accumulator = 0
        step_time_accumulator = 0.0
        last_log_time = time.time()

        # Get total batches for progress tracking
        total_batches = len(self.train_dataloader)

        # Try to use tqdm, but fall back to simple enumeration if it fails
        try:
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=total_batches,
                desc=f"Epoch {current_epoch + 1}",
                position=0,
                leave=True,
                dynamic_ncols=True,
                mininterval=1.0
            )
            iterator = progress_bar
        except Exception as e:
            self.logger.warning(f"Failed to initialize tqdm progress bar: {e}. Using simple enumeration.")
            iterator = enumerate(self.train_dataloader)

        for i, batch in iterator:
            # Skip batches if resuming mid-epoch
            if is_resuming_this_epoch and i <= resume_batch_offset:
                continue

            step_logs = {}
            # Note: on_step_begin is called with the global_step *before* it's potentially incremented in this loop iteration
            self._callback_on_step_begin(global_step, logs=step_logs)

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                self.logger.error(f"Unexpected batch format: {type(batch)}. Expected list/tuple of tensors.")
                continue # Skip batch

            is_last_batch_step = (i + 1) == total_batches
            should_step = ((i + 1) % self.gradient_accumulation_steps == 0) or is_last_batch_step

            try:
                ddp_sync_context = contextlib.nullcontext() # Assuming no DDP for now

                with ddp_sync_context:
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        outputs = self.model(inputs)
                        loss_unscaled = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                    loss_val = loss_unscaled.item()
                    is_loss_invalid = torch.isnan(loss_unscaled).any() or torch.isinf(loss_unscaled).any()

                    if is_loss_invalid:
                        self.logger.warning(f"Step {global_step}, Batch {i+1}/{total_batches}: NaN/Inf loss detected: {loss_val}. Skipping backward/step.")
                        if should_step: self.optimizer.zero_grad(set_to_none=True) # Still zero grad if skipping step
                        continue

                    loss = loss_unscaled / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                # Accumulate metrics for valid steps
                epoch_loss += loss_val
                num_valid_steps_in_epoch += 1 # Increment steps actually processed in *this epoch*
                batch_tokens = inputs.numel()
                total_tokens += batch_tokens
                step_token_accumulator += batch_tokens

                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                    else:
                        self.optimizer.step()

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    # Increment global step *after* a successful optimizer step
                    global_step += 1

                    # Update Progress Tracker with the *new* global_step
                    progress.update(step=global_step, loss=loss_val)

                    # Check max_steps using the *updated* global_step and the *retrieved* max_steps
                    if max_steps is not None and global_step >= max_steps:
                        self.logger.info(f"Reached max_steps ({max_steps}). Ending epoch early.")
                        break

                    # Logging logic...
                    step_logs['loss'] = loss_val
                    if self.scheduler: step_logs['lr'] = self.optimizer.param_groups[0]['lr']
                    step_time_accumulator += (time.time() - last_log_time)
                    if (i + 1) % 1 == 0 or (global_step % self.log_interval == 0) or is_last_batch_step:
                       interval_tokens_sec = step_token_accumulator / step_time_accumulator if step_time_accumulator > 0 else 0
                       step_logs['T/s'] = f"{interval_tokens_sec:.0f}"
                       if hasattr(iterator, 'set_postfix'):
                           iterator.set_postfix(step_logs)
                       log_msg = f"Step: {global_step}, Batch: {i+1}/{total_batches}, Loss: {loss_val:.4f}"
                       if 'lr' in step_logs: log_msg += f", LR: {step_logs['lr']:.2e}"
                       log_msg += f", ~T/s: {step_logs['T/s']}"
                       self.logger.info(log_msg)
                       step_time_accumulator = 0.0
                       step_token_accumulator = 0
                    last_log_time = time.time()

                    # Step end callback is called with the *incremented* global_step
                    self._callback_on_step_end(global_step, logs=step_logs)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self.logger.error(f"CUDA OOM encountered at step {global_step}, batch {i+1}. Consider reducing batch size or enabling gradient accumulation/checkpointing.")
                    raise e
                else:
                    self.logger.error(f"RuntimeError at step {global_step}, batch {i+1}: {e}", exc_info=True)
                    raise e

        # End of epoch calculations
        avg_epoch_loss = epoch_loss / num_valid_steps_in_epoch if num_valid_steps_in_epoch > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0

        self.logger.info(f"Epoch {current_epoch + 1} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}, Tokens/sec: {tokens_per_sec:.2f}")

        epoch_metrics = {
            "loss": avg_epoch_loss,
            "tokens_per_sec": tokens_per_sec,
            "epoch_time_sec": epoch_time,
            "num_steps": num_valid_steps_in_epoch # num_steps tracks steps *within this epoch*
        }

        self.logger.info(f"Epoch {current_epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}, T/s: {tokens_per_sec:.0f}, Time: {epoch_time:.2f}s")

        if 'progress_bar' in locals() and progress_bar:
            progress_bar.close()

        return epoch_metrics

    def _callback_on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Callback hook for step begin."""
        logs = logs or {}
        # Iterate over the list inside CallbackList
        for callback in self.callbacks.callbacks:
            if hasattr(callback, 'on_step_begin'):
                callback.on_step_begin(step, logs=logs)

    def _callback_on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Callback hook for step end."""
        logs = logs or {}
        # Iterate over the list inside CallbackList
        for callback in self.callbacks.callbacks:
            if hasattr(callback, 'on_step_end'):
                callback.on_step_end(step, logs=logs)

# --- Standalone training function (potentially deprecated/redundant) --- #

# ... remove rest of the file ... 