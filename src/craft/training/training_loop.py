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
from .callbacks import CallbackList, SampleGenerationCallback
from .callbacks.sample_generation import SampleGenerationCallback # <-- Import specific callback
from ..models.base import GenerativeModel # Import base model for type hinting
from .progress import ProgressTracker  # Import ProgressTracker
from .checkpointing import CheckpointManager, TrainingState # Import CheckpointManager and TrainingState
from craft.config.schemas import TrainingConfig

class TrainingLoop:
    """Handles the core training loop logic."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        config: TrainingConfig, # Type hint as TrainingConfig (Pydantic)
        experiment_config: Optional[DictConfig] = None, # Add experiment_config (OmegaConf)
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        log_interval: int = 10,
        callbacks: Optional[List[Any]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None, # Add checkpoint manager
        save_steps_interval: int = 0, # Add save interval
        time_save_interval_seconds: int = 0, # <-- Add time interval
        max_steps: Optional[int] = None # Add max_steps
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.config = config # Store Pydantic TrainingConfig
        self.experiment_config = experiment_config # Store OmegaConf DictConfig
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
        self.time_save_interval_seconds = time_save_interval_seconds # <-- Store time interval
        self.max_steps = max_steps # Store max_steps

        # Time tracking for interval-based actions
        self.last_time_based_save = time.time()
        self.last_time_based_sample = time.time()

    def train_epoch(
        self,
        trainer: Any,
        current_epoch: int,
        global_step: int, # This is the step count *at the start* of the epoch
        progress: ProgressTracker,
        loaded_global_step: Optional[int] = None
    ) -> Dict[str, float]:
        """Runs a single training epoch."""
        print(f"DEBUG: TrainingLoop.train_epoch(current_epoch={current_epoch}, global_step={global_step}) - START", flush=True)
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

        # <<< ADD DEBUG PRINT HERE >>>
        print(f"[DEBUG TrainingLoop] Dataloader length: {total_batches}", flush=True)

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
                        self.logger.debug(f"Step {global_step}, Batch {i+1}: Before model forward pass. Input shape: {inputs.shape}")
                        outputs = self.model(inputs)
                        self.logger.debug(f"Step {global_step}, Batch {i+1}: After model forward pass. Output shape: {outputs.shape}")
                        self.logger.debug(f"Step {global_step}, Batch {i+1}: Before loss calculation. Target shape: {targets.shape}")
                        loss_unscaled = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                        self.logger.debug(f"Step {global_step}, Batch {i+1}: After loss calculation. Loss tensor: {loss_unscaled}")

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

                    # <<< REPLACE DEBUG PRINT WITH LOGGER.ERROR >>>
                    self.logger.error(f"[DEBUG TrainingLoop] Checkpoint Eval: Step={global_step}, Interval={self.save_steps_interval}, Manager Exists={self.checkpoint_manager is not None}")

                    # --- Periodic Checkpoint Save --- (MOVED: Now happens *before* max_steps check)
                    current_time = time.time()
                    self.logger.debug(f"[Checkpoint Check] Checking step save: global_step={global_step}, interval={self.save_steps_interval}")
                    step_save_triggered = False
                    if self.checkpoint_manager and self.save_steps_interval > 0 and global_step > 0 and global_step % self.save_steps_interval == 0:
                        # <<< ADD DEBUG PRINT HERE >>>
                        print(f"[DEBUG TrainingLoop] Step {global_step}: Entering step-based checkpoint save block.", flush=True)
                        self.logger.info(
                            f"[TrainingLoop] Step {global_step}: CONDITION MET for step-based checkpoint save "
                            f"(step % interval == {global_step % self.save_steps_interval}) (Filename: checkpoint_step_{global_step:06d}.pt)"
                        )
                        step_save_triggered = True
                        # Construct the state and call checkpoint manager
                        filename = f"checkpoint_step_{global_step:06d}.pt"
                        try:
                            serializable_config = None
                            if self.experiment_config and _OMEGACONF_AVAILABLE:
                                serializable_config = OmegaConf.to_container(self.experiment_config, resolve=True)
                            elif self.config: # Fallback to Pydantic
                                serializable_config = self.config.model_dump()
                            
                            tb_log_dir = None
                            if self.callbacks:
                                for cb in self.callbacks.callbacks:
                                    if cb.__class__.__name__ == 'TensorBoardLogger' and hasattr(cb, 'resolved_log_dir'):
                                        tb_log_dir = cb.resolved_log_dir
                                        break
                                
                            state = TrainingState(
                                epoch=current_epoch,
                                global_step=global_step,
                                model_state_dict=self.model.state_dict(),
                                optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                                scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                                scaler_state_dict=self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else None,
                                best_val_metric=float('inf'), # Step saves don't update best val metric here
                                metrics={'loss': loss_val, 'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else None}, # Include current step metrics
                                config=serializable_config,
                                tensorboard_log_dir=tb_log_dir,
                                callbacks_state=self.callbacks.state_dict() if hasattr(self.callbacks, 'state_dict') else None,
                            )
                            self.checkpoint_manager.save_checkpoint(
                                state=state,
                                filename=filename,
                                metrics={'loss': loss_val}, # Pass current metrics for potential naming/selection
                                is_best=False # Step saves are not based on validation metric
                            )
                        except Exception as e: # <<< CATCHES *ANY* EXCEPTION DURING STATE CREATION OR SAVE >>>
                            self.logger.error(f"Error during step-based checkpoint save for step {global_step}: {e}", exc_info=True)
                            raise e # Re-raise the exception to halt the process

                    else:
                        self.logger.debug(
                            f"[TrainingLoop] Step {global_step}: Condition NOT MET for step-based save "
                            f"(step % interval = {global_step % self.save_steps_interval if self.save_steps_interval > 0 else 'N/A'})"
                        )
                    # --- End Periodic Checkpoint Save ---
                    
                    # --- Time-Based Checkpoint Save --- (Also happens *before* max_steps check)
                    elapsed_save_time = current_time - self.last_time_based_save
                    self.logger.debug(f"[Time Check Save] Current: {current_time:.2f}, Last: {self.last_time_based_save:.2f}, Elapsed: {elapsed_save_time:.2f}, Interval: {self.time_save_interval_seconds}")
                    if self.checkpoint_manager and self.time_save_interval_seconds > 0 and elapsed_save_time >= self.time_save_interval_seconds:
                        self.logger.debug(f"[TIME SAVE BLOCK ENTERED] Step {global_step}. Condition met.")
                        filename = f"checkpoint_step_{global_step:06d}_time.pt" # Differentiate time-based saves
                        self.logger.info(f"[TrainingLoop] Step {global_step}: Triggering time-based checkpoint save (filename: {filename})")
                        serializable_config_dict_time = self.config.model_dump() if isinstance(self.config, TrainingConfig) else self.config
                        tb_log_dir = None
                        if self.callbacks:
                           for cb in self.callbacks.callbacks:
                               if cb.__class__.__name__ == 'TensorBoardLogger' and hasattr(cb, 'resolved_log_dir'):
                                   tb_log_dir = cb.resolved_log_dir
                                   break
                        
                        current_metrics = {'loss': loss_val, 'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else None}
                        state = TrainingState(
                            epoch=current_epoch,
                            global_step=global_step,
                            model_state_dict=self.model.state_dict(),
                            optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                            scaler_state_dict=self.scaler.state_dict() if hasattr(self, 'scaler') and self.scaler is not None else None,
                            best_val_metric=float('inf'),
                            metrics=current_metrics,
                            config=serializable_config_dict_time,
                            tensorboard_log_dir=tb_log_dir,
                        )
                        self.logger.debug(f"[TIME SAVE BLOCK] Calling checkpoint_manager.save_checkpoint...")
                        self.checkpoint_manager.save_checkpoint(
                            state=state,
                            filename=filename,
                            metrics=current_metrics,
                            is_best=False
                        )
                        self.last_time_based_save = current_time
                        self.logger.debug(f"[TIME SAVE BLOCK EXIT] Step {global_step}. Save completed, last_time_based_save updated.")
                    # --- End Time-Based Checkpoint Save ---

                    # --- Time-Based Sample Generation --- (Happens *before* max_steps check)
                    elapsed_sample_time = current_time - self.last_time_based_sample
                    self.logger.debug(f"[Time Check Sample] Current: {current_time:.2f}, Last: {self.last_time_based_sample:.2f}, Elapsed: {elapsed_sample_time:.2f}, Interval: {self.time_save_interval_seconds}")
                    if self.time_save_interval_seconds > 0 and elapsed_sample_time >= self.time_save_interval_seconds:
                        self.logger.debug(f"[TIME SAMPLE BLOCK ENTERED] Step {global_step}. Condition met.")
                        self.logger.info(f"[TrainingLoop] Step {global_step}: Triggering time-based sample generation.")
                        sample_cb_found = False
                        if self.callbacks:
                            self.logger.debug(f"[TIME SAMPLE BLOCK] Iterating through {len(self.callbacks.callbacks)} callbacks...")
                            for idx, cb in enumerate(self.callbacks.callbacks):
                                self.logger.debug(f"[TIME SAMPLE BLOCK] Checking callback #{idx}: {type(cb).__name__} (Instance of SampleGenerationCallback? {isinstance(cb, SampleGenerationCallback)})")
                                if isinstance(cb, SampleGenerationCallback):
                                    self.logger.debug(f"[TIME SAMPLE BLOCK] Found SampleGenerationCallback: {type(cb).__name__}. Calling generate_samples...")
                                    cb.generate_samples(trainer, f"Step {global_step} (Time Interval)")
                                    sample_cb_found = True
                        if sample_cb_found:
                            self.last_time_based_sample = current_time
                            self.logger.debug(f"[TIME SAMPLE BLOCK EXIT] Step {global_step}. Sample generated, last_time_based_sample updated.")
                        else:
                             self.logger.warning("Time interval reached for sample generation, but SampleGenerationCallback not found.")
                    # --- End Time-Based Sample Generation ---

                    # Check max_steps condition *AFTER* incrementing step and periodic saves/samples
                    if self.max_steps is not None and global_step >= self.max_steps:
                        self.logger.info(f"[TrainingLoop] Reached max_steps ({self.max_steps}) at step {global_step}. Breaking epoch loop after saves/samples.")
                        break # Exit the inner batch loop

                    # Calculate metrics for update
                    current_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler else None
                    step_time = time.time() - last_log_time
                    tokens_per_sec = step_token_accumulator / step_time if step_time > 0 else 0
                    step_logs['loss'] = loss_val
                    step_logs['lr'] = current_lr
                    step_logs['T/s'] = f"{tokens_per_sec:.0f}"
                    progress.update(
                        step=global_step,
                        loss=loss_val,
                        learning_rate=current_lr,
                        tokens_per_second=tokens_per_sec,
                        additional_metrics=None
                    )
                    step_time_accumulator = 0.0
                    step_token_accumulator = 0
                    last_log_time = time.time()
                    step_logs['global_step'] = global_step
                    self._callback_on_step_end(step=i, global_step=global_step, logs=step_logs)

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
        for callback in self.callbacks.callbacks:
            if hasattr(callback, 'on_step_begin'):
                callback.on_step_begin(step, logs=logs)

    def _callback_on_step_end(self, step: int, global_step: int, logs: Optional[Dict[str, Any]] = None):
        """Callback hook for step end."""
        logs = logs or {}
        # Iterate over the list inside CallbackList
        for callback in self.callbacks.callbacks:
            if hasattr(callback, 'on_step_end') and callable(getattr(callback, 'on_step_end')):
                try:
                     # Pass step, global_step, and metrics dictionary
                    callback.on_step_end(step=step, global_step=global_step, metrics=logs)
                except Exception as e:
                    self.logger.error(f"Error in callback {callback.__class__.__name__}.on_step_end: {e}", exc_info=True)
