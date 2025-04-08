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
from .callbacks import CallbackList, SampleGenerationCallback
from .callbacks.sample_generation import SampleGenerationCallback # <-- Import specific callback
from .callbacks.tensorboard import TensorBoardLogger # Corrected import
from .callbacks.base import Callback # Added import
from ..models.base import GenerativeModel # Import base model for type hinting
from .progress import ProgressTracker  # Import ProgressTracker
from .checkpointing import CheckpointManager, TrainingState # Import CheckpointManager and TrainingState
from craft.config.schemas import TrainingConfig

class TrainingLoop:
    """Handles the core training loop logic."""
    callbacks: CallbackList # <-- Add class-level type hint
    
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
        self.time_save_interval_seconds = config.time_save_interval_seconds if config.time_save_interval_seconds is not None else 0
        self.max_steps = config.max_steps
        
        # Handle callbacks: Use provided CallbackList or create one
        if isinstance(callbacks, CallbackList):
            self.callbacks = callbacks
        else:
            self.callbacks = CallbackList(callbacks if callbacks is not None else [])
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize scaler for AMP
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.checkpoint_manager = checkpoint_manager # Store manager

        # Time tracking for interval-based actions
        self.last_time_based_save = time.time()
        self.last_time_based_sample = time.time()
        self._last_sample_time = time.time() # Initialize _last_sample_time

    def _prepare_training_state(self, epoch: int, global_step: int) -> TrainingState:
        """Gathers component states and creates a TrainingState object."""
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        scaler_state = self.scaler.state_dict() if self.scaler and self.use_amp else None
        
        callbacks_state = None
        if self.callbacks and hasattr(self.callbacks, 'state_dict') and callable(getattr(self.callbacks, 'state_dict')):
            callbacks_state = self.callbacks.state_dict()

        # TODO: Gather metrics more reliably if needed for the state object itself
        # metrics = ...

        # Get TB log dir (handle potential absence of callback)
        tb_log_dir = None
        if self.callbacks:
            tb_logger = self.callbacks.get_callback(TensorBoardLogger)
            if tb_logger and hasattr(tb_logger, 'resolved_log_dir'):
                 tb_log_dir = tb_logger.resolved_log_dir
        
        serializable_config = None
        if self.config: 
            serializable_config = self.config.model_dump()
        else:
            pass # config is None, serializable_config remains None

        state = TrainingState(
            epoch=epoch,
            global_step=global_step,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            scheduler_state_dict=scheduler_state,
            scaler_state_dict=scaler_state,
            # best_val_metric= # This should come from Trainer/Evaluator state passed in
            # metrics=metrics,
            config=serializable_config,
            tensorboard_log_dir=tb_log_dir,
            callbacks_state=callbacks_state,
        )
        assert self.checkpoint_manager is not None
        return state

    def _should_save_checkpoint(self, global_step: int, current_time: float) -> bool:
        """Checks if a checkpoint should be saved based on step or time intervals."""
        # Step-based trigger using self.save_interval from config
        step_trigger = (
            self.checkpoint_manager
            and self.save_interval > 0
            and global_step > 0 # Avoid saving at step 0
            and global_step % self.save_interval == 0
        )
        # Time-based trigger using self.time_save_interval_seconds from config
        time_trigger = False
        if self.checkpoint_manager and self.time_save_interval_seconds > 0:
            elapsed_since_last_save = current_time - self.last_time_based_save
            if elapsed_since_last_save >= self.time_save_interval_seconds:
                time_trigger = True
                self.last_time_based_save = current_time # Reset timer *only* if triggered

        return step_trigger or time_trigger

    def _should_generate_sample(self, global_step: int, current_time: float) -> bool:
        """Check if sample generation should occur based on interval."""
        if not self.callbacks:
            return False

        # Find the SampleGenerationCallback instance by iterating
        sample_cb: Optional[SampleGenerationCallback] = None
        for cb in self.callbacks:
            if isinstance(cb, SampleGenerationCallback):
                sample_cb = cb
                break
        
        if sample_cb is None:
            # self.logger.debug("SampleGenerationCallback not found in CallbackList.")
            return False # No sampling callback found
        
        # Check if generation interval is met
        if sample_cb.generation_interval_seconds <= 0:
            return False # Interval disabled
            
        now = current_time # Use the provided current time
        if (now - self._last_sample_time) >= sample_cb.generation_interval_seconds:
            self._last_sample_time = now
            # self.logger.debug(f"Sample generation triggered at step {global_step}.")
            return True
            
        return False

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

            step_logs: Dict[str, Any] = {}
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
                        if should_step: 
                            self.optimizer.zero_grad(set_to_none=True) # Still zero grad if skipping step
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

                    # Call step end callback *after* step increment and actions
                    self._callback_on_step_end(i, global_step, step_logs)

                    # <<< REPLACE DEBUG PRINT WITH LOGGER.ERROR >>>
                    self.logger.error(f"[DEBUG TrainingLoop] Checkpoint Eval: Step={global_step}, Interval={self.save_interval}, Manager Exists={self.checkpoint_manager is not None}")

                    # --- Periodic Checkpoint Save --- #
                    current_time = time.time()
                    # Use the helper method to check trigger conditions
                    if self._should_save_checkpoint(global_step, current_time):
                        self.logger.info(f"[TrainingLoop] Triggering checkpoint save at step {global_step}.")
                        # Use helper to prepare state
                        training_state = self._prepare_training_state(current_epoch, global_step)
                        # Pass the state object to the manager
                        filename = f"checkpoint_step_{global_step:06d}.pt"
                        try:
                            # TODO: Determine metrics and is_best for saving
                            # This likely requires state passed from Trainer or Evaluator
                            metrics_for_ckpt = step_logs # Or epoch metrics? Needs clarity.
                            is_best_for_ckpt = False # Need logic based on validation
                            # Add check for None before calling
                            if self.checkpoint_manager:
                                self.checkpoint_manager.save_checkpoint(
                                    state=training_state,
                                    filename=filename,
                                    metrics=metrics_for_ckpt, # Pass current metrics
                                    is_best=is_best_for_ckpt # Pass best status
                                )
                            else:
                                self.logger.warning("Checkpoint save triggered, but CheckpointManager is None.")
                        except Exception as e:
                            self.logger.error(f"Checkpoint saving failed at step {global_step}: {e}", exc_info=True)
                    
                    # --- Periodic Sample Generation --- #
                    # Use the helper method to check trigger conditions
                    if self._should_generate_sample(global_step, current_time):
                        self.logger.info(f"[TrainingLoop] Triggering sample generation at step {global_step}.")
                        # Trigger via a generic callback hook if possible,
                        # or call the specific callback if necessary.
                        sample_cb = self.callbacks.get_callback(SampleGenerationCallback)
                        if sample_cb:
                            try:
                                # Pass context
                                sample_cb.generate_samples(trigger_event=f"step {global_step}") 
                            except Exception as e:
                                self.logger.error(f"Sample generation failed at step {global_step}: {e}", exc_info=True)
                        else:
                            self.logger.warning("Sample generation triggered, but SampleGenerationCallback not found.")

                    # --- Metric Logging & Progress Update --- #
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    step_time = time.time() - last_log_time
                    step_time_accumulator += step_time
                    tokens_per_sec = step_token_accumulator / step_time_accumulator if step_time_accumulator > 0 else 0.0

                    # Log metrics periodically or if it's the last step
                    is_last_step_overall = (self.max_steps is not None and global_step >= self.max_steps)
                    if global_step % self.log_interval == 0 or is_last_batch_step or is_last_step_overall:
                        avg_step_loss = loss_val # Current step loss (avg over accumulation)
                        # Prepare metrics dict for logging and callbacks
                        log_metrics = {
                            "loss": avg_step_loss,
                            "lr": current_lr,
                            "tokens_per_sec": tokens_per_sec
                        }
                        # Add GPU memory usage if available
                        if torch.cuda.is_available():
                             log_metrics["vram_allocated_gb"] = torch.cuda.memory_allocated(self.device) / (1024**3)
                             log_metrics["vram_max_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                             # Reset peak memory stats periodically if desired
                             # torch.cuda.reset_peak_memory_stats(self.device)
                        
                        # Update progress tracker (ensure it can handle these new metrics)
                        progress.update(
                             step=global_step,
                             loss=log_metrics["loss"],
                             learning_rate=log_metrics.get("lr"), # Pass LR if available
                             tokens_per_second=log_metrics.get("tokens_per_sec") # Pass T/s if available
                             # Pass other specific metrics if ProgressTracker supports them
                             # **log_metrics # Reverted: Pass only expected args
                        )
                        # Add metrics to step_logs for on_step_end callback
                        step_logs.update(log_metrics)
                        
                        # Reset step accumulators after logging
                        step_token_accumulator = 0
                        step_time_accumulator = 0.0

                    # Update progress bar description (Now just updates count)
                    if progress.progress_bar:
                        progress.progress_bar.update(1) # This call only increments the counter

            except Exception as e:
                self.logger.error(f"Error during training step {global_step}, batch {i+1}: {e}", exc_info=True)
                # Decide how to handle: skip batch, stop epoch, stop training?
                # For now, just log and continue (might lead to issues if persistent)
                if should_step: # Ensure grads are zeroed if an error occurred before step
                     self.optimizer.zero_grad(set_to_none=True)
                continue # Move to next batch
            finally:
                # Ensure logs are flushed periodically? Or rely on higher level flushing?
                # force_flush_logs() # Maybe too frequent here
                # Potential CUDA memory cleanup (use cautiously)
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                # gc.collect()
                pass

            # Callbacks: on_step_end - called AFTER potential step and increment
            # Pass metrics accumulated during the step (or micro-batches)
            # Ensure standard loss is always included
            if "loss" not in step_logs:
                 step_logs["loss"] = loss_val 
            # Add tokens/sec if calculated (might not be if not a logging step)
            # If step_logs already contains tokens_per_sec from the periodic log, this won't overwrite.
            # If it wasn't a logging step, we might want the *instantaneous* T/s?
            # For now, only log T/s periodically via log_metrics update above.
            # if "tokens_per_sec" not in step_logs and step_time_accumulator > 0:
            #      step_logs["tokens_per_sec"] = step_token_accumulator / step_time_accumulator
            
            self._callback_on_step_end(i, global_step, step_logs)

            # CHECK MAX STEPS *AFTER* THE STEP IS COMPLETED
            if self.max_steps is not None and global_step >= self.max_steps:
                self.logger.info(f"Reached max_steps ({self.max_steps}) after completing step {global_step}. Stopping epoch.")
                break # Exit the batch loop

        # --- Epoch End --- #
        progress.close() # Close the progress bar for the epoch
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_valid_steps_in_epoch if num_valid_steps_in_epoch > 0 else float('nan')
        avg_tokens_per_sec = total_tokens / epoch_duration if epoch_duration > 0 else 0.0

        self.logger.info(
            f"Epoch {current_epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, "
            f"Duration: {format_time(epoch_duration)}, Avg Tokens/Sec: {avg_tokens_per_sec:.2f}"
        )

        # Return metrics collected during the epoch
        # The global_step returned is the value *after* the last step completed in this epoch
        epoch_metrics: Dict[str, Any] = {
            "loss": avg_epoch_loss,
            "epoch_duration_sec": epoch_duration,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "final_global_step": global_step # Return the final step count
        }
        # self.callbacks.on_epoch_end(epoch=current_epoch, global_step=global_step, metrics=epoch_metrics)
        # ^^^ on_epoch_end should be called by Trainer after potential validation

        return epoch_metrics

    # Helper methods for callbacks to keep train_epoch cleaner
    def _callback_on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.callbacks:
            self.callbacks.on_step_begin(step=step, logs=logs)

    def _callback_on_step_end(self, step: int, global_step: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.callbacks:
            # Ensure metrics dict exists
            if logs is None: logs = {}
            self.callbacks.on_step_end(step=step, global_step=global_step, logs=logs)

    def run(self) -> Dict[str, Any]:
        """Runs the full training loop over all epochs."""
        # This run method might be simplified or moved entirely to Trainer?
        # Keeping a basic structure here for now.
        self.logger.info("Starting TrainingLoop run...")
        start_time = time.time()
        final_metrics: Dict[str, Any] = {}
        # Assume initial epoch/step are handled by the caller (Trainer)
        # Need to get initial state correctly (passed via __init__ or args?)
        # For now, assume Trainer manages the outer epoch loop and calls train_epoch

        # Placeholder logic: This loop structure belongs in the Trainer
        # for epoch in range(self.num_epochs): # num_epochs is not attr here
        #     epoch_metrics = self.train_epoch(epoch, self.global_step, ...)
        #     self.global_step = epoch_metrics['final_global_step']
        #     final_metrics.update(epoch_metrics)
        #     # Call validation etc.

        total_time = time.time() - start_time
        self.logger.info(f"TrainingLoop run finished in {format_time(total_time)}.")
        # Return metrics and final state needed by Trainer
        return {
            "metrics": final_metrics,
            "total_train_time": total_time,
            # "global_step": self.global_step, # Return updated state if managed here
            # "epoch": self.epoch
        }
