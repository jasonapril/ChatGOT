#!/usr/bin/env python
"""
Training Loop Module
===================

This module contains the core training loop logic, separated from the main Trainer class
for better organization and testability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import gc
import contextlib
from typing import Tuple, Dict, Any, Callable, List, Optional, Union
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import os

# Attempt to import OmegaConf safely
try:
    from omegaconf import DictConfig, OmegaConf
    _OMEGACONF_AVAILABLE = True
except ImportError:
    _OMEGACONF_AVAILABLE = False
    DictConfig = dict # Define as dict if not available

# Use relative import for utils within the same package
from ..utils.logging import force_flush_logs, format_time
# from ..utils.memory import MemoryMonitor # Removed unused import
from .callbacks import CallbackList
from ..models.base import GenerativeModel # Import base model for type hinting
from .progress import ProgressTracker  # Import ProgressTracker
from .checkpointing import CheckpointManager, TrainingState # Import CheckpointManager and TrainingState

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
        callbacks: Optional[List[Any]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None, # Add checkpoint manager
        save_steps_interval: int = 0, # Add save interval
        max_steps: Optional[int] = None # Add max_steps
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
        self.checkpoint_manager = checkpoint_manager # Store manager
        self.save_steps_interval = save_steps_interval # Store interval
        self.max_steps = max_steps # Store max_steps

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

        # Get max_steps from config safely
        max_steps = getattr(self.config, 'max_steps', None)
        # Get log_interval from config safely
        self.log_interval = getattr(self.config, 'log_interval', 10)

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

        # Start the progress tracker provided by Trainer
        progress.start() 

        # Use simple enumeration, progress tracker handles the visual bar
        iterator = enumerate(self.train_dataloader)

        for i, batch in iterator:
            # Skip batches if resuming mid-epoch
            if is_resuming_this_epoch and i <= resume_batch_offset:
                continue

            # CHECK MAX STEPS *BEFORE* FORWARD PASS (using self.max_steps)
            if self.max_steps is not None and global_step >= self.max_steps:
                self.logger.info(f"Reached max_steps ({self.max_steps}) before processing batch {i+1}. Stopping epoch.")
                break

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

                    # Optimizer Step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                    else:
                        self.optimizer.step()
                    self.scaler.update()

                    # Scheduler Step
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Zero Grad
                    self.optimizer.zero_grad(set_to_none=True)

                    # Increment global step *after* a successful optimizer step
                    global_step += 1

                    # --- Periodic Checkpoint Save (BEFORE max_steps check) ---
                    if self.checkpoint_manager and self.save_steps_interval > 0 and global_step > 0 and global_step % self.save_steps_interval == 0:
                        filename = f"checkpoint_step_{global_step:06d}.pt" # Use padding
                        self.logger.info(f"[TrainingLoop] Step {global_step}: Triggering checkpoint save (filename: {filename})")
                        
                        # Construct state dictionary (Final Check)
                        serializable_config = self.config # Start with original config
                        try:
                            # Try OmegaConf resolution if available
                            if _OMEGACONF_AVAILABLE:
                                serializable_config = OmegaConf.to_container(self.config, resolve=True)
                        except Exception as e:
                            self.logger.warning(f"Could not serialize config for step checkpoint: {e}")
                            serializable_config = None # Fallback

                        tb_log_dir = None
                        if self.callbacks:
                           for cb in self.callbacks.callbacks:
                               if cb.__class__.__name__ == 'TensorBoardLogger' and hasattr(cb, 'resolved_log_dir'):
                                   tb_log_dir = cb.resolved_log_dir
                                   break

                        # --- Create TrainingState object (INSTEAD OF DICT) --- #
                        current_metrics = {'loss': loss_val, 'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else None}
                        state = TrainingState(
                            epoch=current_epoch,
                            global_step=global_step,
                            model_state_dict=self.model.state_dict(),
                            optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                            scaler_state_dict=self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else None,
                            best_val_metric=float('inf'), # Placeholder, step checkpoints aren't 'best' based on val
                            metrics=current_metrics,
                            config=serializable_config,
                            tensorboard_log_dir=tb_log_dir,
                            # callbacks_state=self.callbacks.state_dict() if hasattr(self.callbacks, 'state_dict') else None # Optional future addition
                            # tokenizer_path=self.checkpoint_manager.get_tokenizer_path(global_step) # Needs CheckpointManager method
                        )
                        # --- End create TrainingState object --- #

                        # Save checkpoint
                        self.checkpoint_manager.save_checkpoint(
                            state=state, 
                            filename=filename,
                            metrics=state.metrics,
                            is_best=False 
                        )
                    # --- End Periodic Checkpoint Save ---
                    
                    # Calculate metrics for update
                    current_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler else None
                    step_time = time.time() - last_log_time
                    tokens_per_sec = step_token_accumulator / step_time if step_time > 0 else 0

                    # Update Progress Tracker with the *new* global_step and metrics
                    step_logs['loss'] = loss_val
                    step_logs['lr'] = current_lr
                    step_logs['T/s'] = f"{tokens_per_sec:.0f}"
                    progress.update(
                        step=global_step,
                        loss=loss_val,
                        learning_rate=current_lr,
                        tokens_per_second=tokens_per_sec,
                        additional_metrics=None # Add other metrics if needed
                    )

                    # Reset step accumulators
                    step_time_accumulator = 0.0
                    step_token_accumulator = 0
                    last_log_time = time.time()

                    # Step end callback is called with the *incremented* global_step
                    self._callback_on_step_end(global_step, logs=step_logs)

                    # Check max_steps using the *updated* global_step (AFTER checkpointing and callbacks)
                    if max_steps is not None and global_step >= max_steps:
                        self.logger.info(f"Reached max_steps ({max_steps}). Ending epoch early.")
                        break

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
        
        # Restore epoch summary logging
        self.logger.info(f"Epoch {current_epoch + 1} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}, Tokens/sec: {tokens_per_sec:.2f}")

        epoch_metrics = {
            "loss": avg_epoch_loss,
            "tokens_per_sec": tokens_per_sec,
            "epoch_time_sec": epoch_time,
            "final_global_step": global_step # Return the final global step
        }

        # Restore progress bar closing
        progress.close()

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
