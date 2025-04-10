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
from typing import Tuple, Dict, Any, Callable, List, Optional, Union, cast
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

# Use relative import for utils within the same package
from ..utils.logging import force_flush_logs, format_time
# from ..utils.memory import MemoryMonitor # Removed unused import
from .callbacks import CallbackList, SampleGenerationCallback # Corrected import
from .callbacks.sample_generation import SampleGenerationCallback # Import specific callback
from .callbacks.tensorboard import TensorBoardLogger # Corrected import
from .callbacks.base import Callback # Added import
from ..models.base import GenerativeModel # Import base model for type hinting
from .progress import ProgressTracker  # Import ProgressTracker
from .checkpointing import CheckpointManager, TrainingState # Import CheckpointManager and TrainingState
from craft.config.schemas import TrainingConfig

# Helper to safely get learning rate
def get_current_lr(optimizer: Optional[torch.optim.Optimizer]) -> Optional[float]:
    """Safely retrieves the current learning rate from the optimizer."""
    if optimizer and optimizer.param_groups:
        lr = optimizer.param_groups[0].get('lr')
        return float(lr) if isinstance(lr, (int, float)) else None
    return None

# Helper to get CUDA memory stats (in GB)
def get_cuda_memory_stats(device: torch.device) -> Dict[str, float]:
    """Gets current and peak allocated CUDA memory in GB if available."""
    stats = {}
    if torch.cuda.is_available() and device.type == 'cuda':
        try:
            allocated = torch.cuda.memory_allocated(device) / (1024**3) # Convert bytes to GB
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            stats['vram_allocated_gb'] = allocated
            stats['vram_max_allocated_gb'] = max_allocated
        except Exception as e:
             # Log error if memory stats fail, but don't crash training
             logging.getLogger(__name__).warning(f"Could not get CUDA memory stats: {e}", exc_info=True)
    return stats

class TrainingLoop:
    """Handles the core training loop logic."""
    callbacks: CallbackList

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        config: TrainingConfig, # Type hint as TrainingConfig (Pydantic)
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None, # Add checkpoint manager
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.config = config # Store Pydantic TrainingConfig
        self.scheduler = scheduler

        # Access parameters from config
        self.use_amp = config.use_amp
        self.gradient_accumulation_steps = max(1, config.gradient_accumulation_steps)
        self.max_grad_norm = config.max_grad_norm
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval if config.save_interval is not None else 0
        self.max_steps = config.max_steps

        # Handle callbacks: Use provided CallbackList or create one
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(callbacks if callbacks is not None else [])
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize scaler for AMP using config value and device type
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.checkpoint_manager = checkpoint_manager # Store manager

        # Time tracking for interval-based actions
        self.last_time_based_save = time.time()

    def _prepare_training_state(self, epoch: int, global_step: int) -> TrainingState:
        """Gathers component states and creates a TrainingState object."""
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        scaler_state = self.scaler.state_dict() if self.scaler and self.use_amp else None

        callbacks_state = None
        if self.callbacks and hasattr(self.callbacks, 'state_dict') and callable(getattr(self.callbacks, 'state_dict')):
            callbacks_state = self.callbacks.state_dict()

        # Get TB log dir (handle potential absence of callback)
        tb_log_dir = None
        if self.callbacks:
            tb_logger = self.callbacks.get_callback(TensorBoardLogger)
            if tb_logger and hasattr(tb_logger, 'resolved_log_dir'):
                 tb_log_dir = tb_logger.resolved_log_dir

        serializable_config = None
        if self.config:
            serializable_config = self.config.model_dump()

        state = TrainingState(
            epoch=epoch,
            global_step=global_step,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            scheduler_state_dict=scheduler_state,
            scaler_state_dict=scaler_state,
            config=serializable_config,
            tensorboard_log_dir=tb_log_dir,
            callbacks_state=callbacks_state,
            # TODO: Consider how best_val_metric is managed and potentially saved
        )
        # Removed assert self.checkpoint_manager is not None, as it might be optional
        return state

    def _should_save_checkpoint(self, global_step: int, current_time: float) -> bool:
        """Checks if a checkpoint should be saved based on step or time intervals."""
        # Step-based trigger
        step_trigger = (
            self.checkpoint_manager
            and self.save_interval > 0
            and global_step > 0 # Avoid saving at step 0
            and global_step % self.save_interval == 0
        )
        # Time-based trigger
        time_trigger = False
        time_save_interval = getattr(self.config, 'time_save_interval_seconds', 0)
        if self.checkpoint_manager and time_save_interval > 0:
            elapsed_since_last_save = current_time - self.last_time_based_save
            if elapsed_since_last_save >= time_save_interval:
                time_trigger = True
                self.last_time_based_save = current_time

        return step_trigger or time_trigger

    # --- Main Epoch Training Method --- #
    def train_epoch(
        self,
        trainer: Any, # Pass the Trainer instance for callbacks needing broader context
        current_epoch: int,
        global_step: int, # Step count at the *start* of the epoch
        progress: ProgressTracker, # Progress tracker instance from Trainer
        loaded_global_step: Optional[int] = None # To handle resuming mid-epoch
    ) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        epoch_loss_total = 0.0
        num_optimizer_steps_in_epoch = 0 # Tracks optimizer steps within this epoch
        epoch_start_time = time.time()
        steps_per_epoch = len(self.train_dataloader)
        self.optimizer.zero_grad(set_to_none=True) # Reset gradients at epoch start

        # Get max_steps and log_interval from config safely
        max_steps = self.config.max_steps
        self.log_interval = self.config.log_interval # Can update dynamically if config changes

        # Batch skipping logic for resuming
        resume_batch_offset = -1
        is_resuming_this_epoch = False
        if loaded_global_step is not None and current_epoch == (loaded_global_step // steps_per_epoch):
            resume_batch_offset = loaded_global_step % steps_per_epoch
            self.logger.info(f"Resuming epoch {current_epoch+1} from batch offset {resume_batch_offset + 1}/{steps_per_epoch} (global step {loaded_global_step + 1})")
            is_resuming_this_epoch = True

        last_step_time = time.time() # Timer for step duration calculation
        accumulation_window_losses = [] # Track losses for current accumulation window

        # Start the progress tracker if it wasn't already started
        if progress.start_time is None:
            progress.start()
        iterator = enumerate(self.train_dataloader)

        batch_idx = -1 # Initialize batch_idx before the loop
        # --- Batch Loop ---
        for batch_idx, batch in iterator:
            try:
                # Check max_steps *before* processing the batch or try-except block
                current_effective_step = global_step + num_optimizer_steps_in_epoch
                if max_steps is not None and current_effective_step >= max_steps:
                     self.logger.info(f"Max steps ({max_steps}) reached at step {current_effective_step}. Stopping epoch before batch {batch_idx}.")
                     break # Exit batch loop

                # Handle resuming mid-epoch by skipping processed batches
                if is_resuming_this_epoch and batch_idx <= resume_batch_offset:
                    if batch_idx == resume_batch_offset:
                        self.logger.info(f"Fast-forwarded to batch {batch_idx + 1}. Resuming training.")
                    continue # Skip this batch

                # Adjust global step only AFTER skipping checks for resuming
                # current_global_step reflects the step *before* this batch's potential optimizer update
                current_global_step = global_step + num_optimizer_steps_in_epoch

                # --- Callback: On Step Begin (triggered before forward/backward) ---
                step_logs: Dict[str, Any] = {"batch_size": len(batch[0]) if isinstance(batch, (list, tuple)) and len(batch) > 0 else 1} # Example batch size
                self._callback_on_step_begin(batch_idx, current_global_step, step_logs)

                batch_start_time = time.time()

                # --- Data Transfer (moved inside try block) ---
                inputs = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)

                # --- Forward Pass ---
                # Use appropriate context manager for AMP
                amp_context = torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=self.use_amp)
                with amp_context:
                    # Assume model returns logits or (logits, loss) or dict with loss
                    output = self.model(inputs)
                    loss: Optional[torch.Tensor] = None
                    logits: Optional[torch.Tensor] = None
                    # Check output type and extract/calculate loss
                    if isinstance(output, torch.Tensor):
                        logits = output
                        flat_logits = logits.view(-1, logits.size(-1)).float()
                        flat_targets = targets.view(-1)
                        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
                    elif isinstance(output, tuple) and len(output) > 0:
                        logits = output[0] if isinstance(output[0], torch.Tensor) else None
                        # Check second element for loss tensor or dict with 'loss'
                        if len(output) > 1:
                            if isinstance(output[1], torch.Tensor):
                                loss = output[1]
                            # Safely check dict and key
                            elif isinstance(output[1], dict) and isinstance(output[1].get('loss'), torch.Tensor):
                                loss = output[1]['loss']
                        # Fallback: Calculate loss if not found in tuple and logits exist
                        if loss is None and logits is not None:
                            flat_logits = logits.view(-1, logits.size(-1)).float()
                            flat_targets = targets.view(-1)
                            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
                    elif isinstance(output, dict):
                        loss = output.get('loss') if isinstance(output.get('loss'), torch.Tensor) else None
                        logits = output.get('logits') if isinstance(output.get('logits'), torch.Tensor) else None
                        # Fallback: Calculate loss if not in dict and logits exist
                        if loss is None and logits is not None:
                            flat_logits = logits.view(-1, logits.size(-1)).float()
                            flat_targets = targets.view(-1)
                            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
                    else:
                        self.logger.warning(f"Unexpected model output type: {type(output)}. Cannot determine loss automatically.")

                    # Ensure loss is a scalar tensor
                    if loss is None:
                         self.logger.error("Loss calculation failed or was not performed based on model output. Using dummy loss.")
                         loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Dummy loss to avoid crash
                    elif loss.dim() != 0:
                         self.logger.warning(f"Loss tensor has dimensions {loss.shape}, expected scalar. Attempting to average.")
                         loss = loss.mean()

                    # Check for NaN/Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf loss detected at Step {current_global_step}, Batch {batch_idx}. Loss: {loss.item()}. Skipping batch.")
                        continue # Skip backward/optimizer step

                    # Normalize loss if using gradient accumulation
                    normalized_loss = loss / self.gradient_accumulation_steps

                # --- Backward Pass ---
                self.scaler.scale(normalized_loss).backward()

                # Accumulate loss for epoch average calculation (use un-normalized loss)
                # Use .item() to get Python number and avoid graph retention
                current_batch_loss_unscaled = loss.item() # type: ignore
                epoch_loss_total += current_batch_loss_unscaled
                accumulation_window_losses.append(current_batch_loss_unscaled)

                # --- Gradient Accumulation & Optimizer Step ---
                num_optimizer_steps_this_batch = 0 # Track if step happened *this batch*
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # --- Gradient Clipping ---
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        step_logs['grad_norm'] = total_norm.item()

                    # --- Optimizer Step ---
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    num_optimizer_steps_in_epoch += 1
                    num_optimizer_steps_this_batch = 1

                    # --- Scheduler Step (after optimizer step) ---
                    if self.scheduler:
                        self.scheduler.step()

                # --- Logging --- #
                # Log metrics only when an optimizer step occurs
                if num_optimizer_steps_this_batch > 0:
                    current_global_step = global_step + num_optimizer_steps_in_epoch # Step number *after* potential update
                    if current_global_step % self.log_interval == 0:
                        # Calculate average loss over the accumulation window
                        avg_loss_window = sum(accumulation_window_losses) / len(accumulation_window_losses)
                        accumulation_window_losses = [] # Reset window
                        current_lr = get_current_lr(self.optimizer)
                        step_time_taken = time.time() - last_step_time
                        last_step_time = time.time()
                        # Calculate samples/sec based on batches processed in interval
                        # Use defaults for optional ints just in case
                        log_interval = self.log_interval or 1
                        grad_accum_steps = self.gradient_accumulation_steps or 1
                        batch_size = step_logs.get("batch_size", 1)
                        samples_per_sec = log_interval * grad_accum_steps * batch_size / step_time_taken

                        # Add relevant info to step_logs
                        step_logs['lr'] = current_lr
                        # Cast loss to float for logging/progress
                        step_logs['loss'] = float(avg_loss_window)
                        step_logs['samples_per_sec'] = samples_per_sec
                        step_logs['step_time_s'] = step_time_taken
                        step_logs.update(get_cuda_memory_stats(self.device))

                        # Update progress tracker
                        progress.update(loss=float(avg_loss_window), step=current_global_step)

                # --- Callback: On Step End ---
                # Trigger after backward and potential optimizer step
                current_global_step = global_step + num_optimizer_steps_in_epoch # Use updated step
                self._callback_on_step_end(batch_idx, current_global_step, step_logs)

                # --- Checkpointing (Interval based) ---
                # Checkpoint based on global step count (optimizer steps)
                if self._should_save_checkpoint(current_global_step, time.time()):
                    self.logger.info(f"Interval reached. Saving checkpoint at global step {current_global_step}...")
                    # Prepare state and filename
                    state_to_save = self._prepare_training_state(epoch=current_epoch, global_step=current_global_step)
                    if self.checkpoint_manager:
                         # Generate filename using checkpoint manager's prefix
                         filename = f"{self.checkpoint_manager.checkpoint_prefix}_epoch_{state_to_save.epoch}_step_{state_to_save.global_step}.pt"
                         # Call save_checkpoint with state and filename
                         self.checkpoint_manager.save_checkpoint(
                              state=state_to_save,
                              filename=filename,
                              metrics=step_logs, # Pass step metrics
                              is_best=False # Interval save is not 'best'
                         )
                    else:
                         self.logger.warning("Checkpoint interval reached, but no CheckpointManager configured.")

                # Check max_steps again after potential optimizer step
                if max_steps is not None and current_global_step >= max_steps:
                    self.logger.info(f"Max steps ({max_steps}) reached after optimizer step {current_global_step}. Stopping epoch.")
                    break # Exit batch loop

            except Exception as batch_e:
                 self.logger.error(f"Error processing batch {batch_idx}: {batch_e}", exc_info=True)
                 self.optimizer.zero_grad(set_to_none=True)
                 continue

        # --- End of Epoch ---
        final_global_step = global_step + num_optimizer_steps_in_epoch
        epoch_duration = time.time() - epoch_start_time
        # Calculate average loss based on batches processed *before* max_steps might have stopped the loop
        # epoch_loss_total includes losses from all processed batches. Need number of batches that contributed.
        # How many batches were fully processed? This is tricky if loop broke mid-accumulation.
        # A simpler approach: Average loss over the optimizer steps taken.
        # We need the sum of losses for the steps taken. `epoch_loss_total` is sum over batches.
        # Let's stick to averaging over batches for now, acknowledging potential inaccuracy if loop breaks early.
        num_batches_processed = batch_idx + 1 # Assuming batch_idx is the index of the last processed batch (or last attempted)
        if max_steps is not None and (global_step + num_optimizer_steps_in_epoch) >= max_steps:
             # If max_steps caused early exit, adjust num_batches_processed?
             # num_batches_processed might be 1 more than actually contributed to steps if break happened before step
             # Let's use num_optimizer_steps * accumulation_steps as a proxy for processed batches leading to steps? No, too complex.
             # Stick with average over completed optimizer steps? Need to store losses per step.
             # Easiest: Average over batches run so far.
             avg_epoch_loss = epoch_loss_total / num_batches_processed if num_batches_processed > 0 else 0.0
        else:
            avg_epoch_loss = epoch_loss_total / steps_per_epoch if steps_per_epoch > 0 else 0.0


        epoch_metrics = {
            'train/loss_epoch': avg_epoch_loss,
            'epoch': current_epoch + 1,
            'duration_seconds': epoch_duration,
        }
        self.logger.info(f"Epoch {current_epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, Duration: {format_time(epoch_duration)}")

        # Trigger epoch end callbacks
        self._callback_on_epoch_end(current_epoch, final_global_step, epoch_metrics)

        # Return metrics including steps completed and average loss
        return {
            "steps_completed_in_epoch": num_optimizer_steps_in_epoch,
            "average_epoch_loss": avg_epoch_loss,
        }

    # --- Callback Triggers --- #
    def _callback_on_step_begin(self, step: int, global_step: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Triggers the on_step_begin method of registered callbacks."""
        if self.callbacks and hasattr(self.callbacks, 'on_step_begin'):
            try:
                # Pass global_step for consistent tracking
                self.callbacks.on_step_begin(step=step, global_step=global_step, logs=logs)
            except Exception as e:
                self.logger.error(f"Error in Callback on_step_begin: {e}", exc_info=True)

    def _callback_on_step_end(self, step: int, global_step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Triggers the on_step_end method of registered callbacks."""
        if self.callbacks and hasattr(self.callbacks, 'on_step_end'):
            try:
                # Pass the calculated metrics dict as 'metrics' argument
                self.callbacks.on_step_end(step=step, global_step=global_step, metrics=metrics)
            except Exception as e:
                self.logger.error(f"Error in Callback on_step_end: {e}", exc_info=True)

    def _callback_on_epoch_end(self, epoch: int, global_step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Triggers the on_epoch_end method of registered callbacks."""
        if self.callbacks and hasattr(self.callbacks, 'on_epoch_end'):
            try:
                self.callbacks.on_epoch_end(epoch=epoch, global_step=global_step, metrics=metrics)
            except Exception as e:
                self.logger.error(f"Error in Callback on_epoch_end: {e}", exc_info=True)
