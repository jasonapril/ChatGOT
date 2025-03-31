#!/usr/bin/env python
"""
Trainer Module
==============

This module defines the core Trainer class responsible for orchestrating the
model training and evaluation process. It encapsulates the training loop,
evaluation logic, checkpointing, and callback integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import logging
import time
import os
from typing import Dict, Any, Optional, List, Tuple
import torch.nn.functional as F
from tqdm import tqdm
import contextlib # For potential future DDP no_sync context
from src.utils.io import ensure_directory # Revert to absolute import
from src.data.dataset import CharDataset # Import CharDataset
from src.training.generation import generate_text # Correct import for generate_text
import sys


class Trainer:
    """
    Core Trainer class for managing the training and evaluation loop.

    Handles device placement, mixed precision, gradient accumulation,
    checkpointing, and callbacks.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        epochs: int = 10,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 50,
        vocab_path: Optional[str] = None
    ):
        """
        Initializes the Trainer.

        Args:
            model: The neural network model to train.
            optimizer: The optimizer for updating model parameters.
            train_dataloader: DataLoader for the training dataset.
            val_dataloader: DataLoader for the validation dataset (optional).
            scheduler: Learning rate scheduler (optional).
            device: The device to run training on (CPU or CUDA).
            epochs: Total number of epochs to train for.
            config: Dictionary containing training configuration (optional, used for nested params).
            callbacks: List of callbacks to use during training (optional).
            use_amp: Whether to use Automatic Mixed Precision (AMP).
            gradient_accumulation_steps: Number of steps to accumulate gradients over.
            max_grad_norm: Maximum norm for gradient clipping (optional).
            checkpoint_dir: Directory to save checkpoints.
            log_interval: Frequency (in steps) for logging training progress.
            vocab_path: Path to the vocabulary file for sampling (optional).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.config = config if config is not None else {}
        self.callbacks = callbacks if callbacks is not None else []
        self.use_amp = use_amp
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.vocab_path = vocab_path

        # Time-based checkpointing (use self.config if available)
        # Access nested training config correctly
        training_cfg = self.config.get('training', {}) # Get the training sub-config
        self.time_save_interval_minutes = training_cfg.get('time_save_interval_minutes', 0)
        self.time_save_interval_seconds = self.time_save_interval_minutes * 60 if self.time_save_interval_minutes else 0
        # Read max_steps from the training sub-config
        self.max_steps = training_cfg.get('max_steps', float('inf'))
        self.logger = logging.getLogger(self.__class__.__name__) # Initialize logger
        self.last_time_save = time.time() # Initialize last save time
        if self.time_save_interval_seconds > 0:
            self.logger.info(f"Time-based checkpointing enabled every {self.time_save_interval_minutes} minutes.")
        if self.max_steps != float('inf'):
            self.logger.info(f"Training will stop after {self.max_steps} global steps.")

        self.scaler = GradScaler(enabled=self.use_amp)
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_global_step = -1
        self.best_val_metric = float('inf')
        self.metrics: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}

        ensure_directory(self.checkpoint_dir)
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Initializes callbacks by setting the trainer instance."""
        for callback in self.callbacks:
            # Assuming callbacks have a 'set_trainer' method
            if hasattr(callback, 'set_trainer') and callable(callback.set_trainer):
                callback.set_trainer(self)
            else:
                self.logger.warning(f"Callback {type(callback).__name__} does not have a set_trainer method.")

    def train(self):
        """Main training loop orchestrating epochs and steps."""
        self.logger.info("Starting training...")
        self._callback_on_train_begin()
        train_start_time = time.time()

        start_epoch = self.current_epoch # Resuming from loaded checkpoint

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            self._callback_on_epoch_begin(epoch)

            # Check if max_steps reached before starting epoch
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.logger.info(f"Reached max_steps ({self.max_steps}). Stopping training.")
                break

            # Train
            train_metrics = self._train_epoch()
            self.metrics['train_loss'].append(train_metrics.get('loss', float('nan')))

            # Evaluate ONLY if max_steps hasn't been reached
            val_metrics = {}
            if self.global_step < self.max_steps:
                if self.val_dataloader is not None:
                    val_metrics = self._evaluate()
                    current_val_loss = val_metrics.get('loss', float('inf'))
                    self.metrics['val_loss'].append(current_val_loss)

                    # Save best model checkpoint
                    if current_val_loss < self.best_val_metric:
                        self.best_val_metric = current_val_loss
                        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                        self.logger.info(f"New best validation metric: {self.best_val_metric:.4f}. Saving best model...")
                        self.save_checkpoint(best_model_path, is_best=True)
                else:
                    # Optionally log that validation is skipped if no dataloader exists
                    # self.logger.debug("No validation dataloader, skipping evaluation.")
                    pass
            else:
                self.logger.info(f"Skipping evaluation because max_steps ({self.max_steps}) was reached during training epoch.")

            # Periodic checkpoint saving
            save_interval = self.config.get('save_interval', 1)
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                 checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                 self.logger.info(f"Saving periodic checkpoint for epoch {epoch+1}...")
                 self.save_checkpoint(checkpoint_path)

            # Time-based checkpoint saving
            current_time = time.time()
            if self.time_save_interval_seconds > 0 and (current_time - self.last_time_save) > self.time_save_interval_seconds:
                timed_checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{self.global_step}.pt')
                self.logger.info(f"Saving timed checkpoint after {(current_time - self.last_time_save)/60:.1f} minutes (End of Epoch {epoch+1}, Global Step: {self.global_step})...")
                self.save_checkpoint(timed_checkpoint_path)
                self.last_time_save = current_time
                self._generate_sample_and_log()

            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch + 1}/{self.epochs} finished in {epoch_time:.2f}s | "
            log_msg += f"Train Loss: {train_metrics.get('loss', 'N/A'):.4f}"
            if val_metrics:
                log_msg += f" | Val Loss: {val_metrics.get('loss', 'N/A'):.4f}"
            self.logger.info(log_msg)

            # Epoch end callback
            epoch_logs = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._callback_on_epoch_end(epoch, logs=epoch_logs)

            self.logger.debug(f"End of epoch {epoch+1} loop iteration. Global step: {self.global_step}, Max steps: {self.max_steps}")

            # No need for the second max_steps check here, it's handled inside _train_epoch

        total_train_time = time.time() - train_start_time
        self.logger.info(f"Training finished in {total_train_time:.2f}s. Best Val Metric: {self.best_val_metric:.4f}")
        self._callback_on_train_end()

        return self.metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Trains the model for one epoch."""
        self.logger.info(f"Starting Epoch {self.current_epoch + 1}/{self.epochs}")
        self.model.train()
        epoch_loss = 0.0
        total_batches = len(self.train_dataloader)
        epoch_start_time = time.time()
        num_valid_steps_in_epoch = 0
        total_tokens = 0
        step_time_accumulator = 0.0
        step_token_accumulator = 0
        last_log_time = time.time()

        # --- Batch Skipping Logic for Resuming ---
        steps_per_epoch = len(self.train_dataloader)
        resume_batch_offset = -1
        is_resuming_this_epoch = False
        if self.loaded_global_step >= 0 and self.current_epoch == (self.loaded_global_step // steps_per_epoch):
            resume_batch_offset = self.loaded_global_step % steps_per_epoch
            self.logger.info(f"Resuming epoch {self.current_epoch+1} from batch offset {resume_batch_offset + 1}/{steps_per_epoch} (global step {self.loaded_global_step + 1})")
            is_resuming_this_epoch = True
            # Reset so we don't skip in subsequent epochs
            self.loaded_global_step = -1 
        # --- End Batch Skipping Logic ---

        self.optimizer.zero_grad(set_to_none=True)

        # Re-enable tqdm progress bar
        progress_bar = tqdm(
            enumerate(self.train_dataloader), 
            total=total_batches, 
            desc=f"Epoch {self.current_epoch + 1}",
            leave=True
        )
        for i, batch in progress_bar:
            # Skip batches if resuming mid-epoch
            if is_resuming_this_epoch and i <= resume_batch_offset:
                continue

            step_logs = {}
            self._callback_on_step_begin(self.global_step, logs=step_logs)

            # Correctly unpack the batch dictionary
            inputs = batch['input_ids'].to(self.device, non_blocking=True)
            targets = batch['labels'].to(self.device, non_blocking=True)

            # Determine if this is an accumulation step
            is_last_batch_step = (i + 1) == total_batches
            should_step = ((i + 1) % self.gradient_accumulation_steps == 0) or is_last_batch_step

            try:
                 # Use nullcontext for potential DDP later
                ddp_sync_context = contextlib.nullcontext() # Replace with model.no_sync() if using DDP

                with ddp_sync_context:
                    # Forward pass with AMP
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        outputs = self.model(inputs)
                        # TODO: Make loss calculation more flexible (allow model to return loss)
                        loss_unscaled = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                    # Check for NaN/Inf before scaling/accumulation division
                    loss_val = loss_unscaled.item()
                    is_loss_invalid = torch.isnan(loss_unscaled).any() or torch.isinf(loss_unscaled).any()

                    if is_loss_invalid:
                        self.logger.warning(f"Step {self.global_step}, Batch {i+1}/{total_batches}: NaN/Inf loss detected: {loss_val}. Skipping backward/step.")
                        # Still need to potentially zero gradients if it was a step batch
                        if should_step: self.optimizer.zero_grad(set_to_none=True)
                        continue # Skip the rest of the loop for this batch

                    # Normalize loss for gradient accumulation
                    loss = loss_unscaled / self.gradient_accumulation_steps

                # Backward pass (scaled)
                self.scaler.scale(loss).backward()

                # Accumulate metrics only for valid steps
                epoch_loss += loss_val
                num_valid_steps_in_epoch += 1
                batch_tokens = inputs.numel()
                total_tokens += batch_tokens # Example token count
                step_token_accumulator += batch_tokens # Accumulate for T/s estimate

                # Optimizer step
                if should_step:
                    # Unscale gradients before clipping/stepping
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step (scaler handles grad check/skipping)
                    self.scaler.step(self.optimizer)

                    # Update the scaler for next iteration
                    self.scaler.update()

                    # Zero gradients *after* stepping
                    self.optimizer.zero_grad(set_to_none=True)

                    # Step the scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1

                    # CORRECT PLACEMENT for max_steps check within epoch loop
                    self.logger.debug(f"Checking max_steps: global_step={self.global_step}, max_steps={self.max_steps}")
                    if self.max_steps is not None and self.global_step >= self.max_steps:
                        self.logger.info(f"Reached max_steps ({self.max_steps}). Ending epoch early.")
                        break # Exit the inner batch loop

                    # Check time every step
                    current_time = time.time()
                    time_interval_exceeded = self.time_save_interval_seconds > 0 and (current_time - self.last_time_save) > self.time_save_interval_seconds

                    # Logging (runs less frequently)
                    step_logs['loss'] = loss_val
                    if self.scheduler: step_logs['lr'] = self.optimizer.param_groups[0]['lr']

                    step_time_accumulator += (current_time - last_log_time)

                    if (self.global_step % self.log_interval == 0) or is_last_batch_step:
                        # Calculate T/s over the log interval
                        interval_tokens_sec = step_token_accumulator / step_time_accumulator if step_time_accumulator > 0 else 0
                        step_logs['T/s'] = f"{interval_tokens_sec:.0f}"

                         # Simple log message construction
                        log_msg = f"Step: {self.global_step}, Batch: {i+1}/{total_batches}, Loss: {loss_val:.4f}"
                        if 'lr' in step_logs: log_msg += f", LR: {step_logs['lr']:.2e}"
                        log_msg += f", ~T/s: {step_logs['T/s']}"
                        
                        # Use tqdm.write for console AND logger.info for file log
                        progress_bar.write(log_msg) 
                        self.logger.info(log_msg) # Uncommented to ensure logs go to file handler
                        
                        # Re-add progress bar update
                        progress_bar.set_postfix(step_logs) 

                        # Perform timed save/sample *if interval exceeded* and it's a log step
                        if time_interval_exceeded:
                            timed_checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{self.global_step}.pt')
                            self.logger.info(f"Saving timed checkpoint after {(current_time - self.last_time_save)/60:.1f} minutes (Aligned with Global Step: {self.global_step})...")
                            self.save_checkpoint(timed_checkpoint_path)
                            self.last_time_save = current_time # Reset timer after saving
                            self._generate_sample_and_log()

                        # Reset accumulators for next interval
                        step_time_accumulator = 0.0
                        step_token_accumulator = 0
                    
                    last_log_time = current_time # Update last log time

                    # Step end callback *after* optimizer/scheduler steps
                    self._callback_on_step_end(self.global_step, logs=step_logs)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self.logger.error(f"CUDA OOM encountered at step {self.global_step}, batch {i+1}. Consider reducing batch size or enabling gradient accumulation/checkpointing.")
                    # TODO: Implement more robust OOM handling (cleanup, checkpointing)
                    raise e # Re-raise for now
                else:
                    self.logger.error(f"RuntimeError at step {self.global_step}, batch {i+1}: {e}", exc_info=True)
                    raise e

        # End of epoch calculations
        avg_epoch_loss = epoch_loss / num_valid_steps_in_epoch if num_valid_steps_in_epoch > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0

        self.logger.info(f"Epoch {self.current_epoch + 1} finished in {epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}, Tokens/sec: {tokens_per_sec:.2f}")

        return {'loss': avg_epoch_loss, 'tokens_per_sec': tokens_per_sec}

    def _evaluate(self) -> Dict[str, float]:
        """Evaluates the model on the validation set."""
        if self.val_dataloader is None:
            self.logger.info("No validation dataloader provided, skipping evaluation.")
            return {}
        self.logger.info("Starting evaluation...")
        self.model.eval()
        total_loss = 0.0
        total_batches = len(self.val_dataloader)
        eval_start_time = time.time()
        total_tokens = 0 # For potential throughput calculation

        # TODO: Add evaluate_begin callback hook if needed

        progress_bar = tqdm(self.val_dataloader, total=total_batches, desc="Evaluating")

        with torch.no_grad():
            for batch in progress_bar:
                # Correctly unpack the batch dictionary
                inputs = batch['input_ids'].to(self.device, non_blocking=True)
                targets = batch['labels'].to(self.device, non_blocking=True)

                # Forward pass with AMP
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    # TODO: Make loss calculation more flexible
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

                if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    total_loss += loss.item()
                    total_tokens += inputs.numel()
                else:
                    self.logger.warning("NaN/Inf detected during evaluation. Skipping batch.")

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        eval_time = time.time() - eval_start_time
        tokens_per_sec = total_tokens / eval_time if eval_time > 0 else 0

        self.logger.info(f"Evaluation finished in {eval_time:.2f}s. Avg Loss: {avg_loss:.4f}, Tokens/sec: {tokens_per_sec:.2f}")

        # TODO: Add evaluate_end callback hook if needed

        return {'loss': avg_loss, 'tokens_per_sec': tokens_per_sec}

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Saves a checkpoint of the model and training state."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure config is serializable (convert OmegaConf if needed)
        serializable_config = self.config
        try:
            # Attempt to resolve OmegaConf to primitive types if it's OmegaConf
            # Check if it's DictConfig rather than dict
            from omegaconf import DictConfig, OmegaConf
            if isinstance(self.config, DictConfig):
                serializable_config = OmegaConf.to_container(self.config, resolve=True)
        except ImportError:
            # If OmegaConf not installed or fails, proceed with original (might raise error later)
            pass
        except Exception as e:
            self.logger.warning(f"Could not serialize OmegaConf config for checkpoint: {e}")
            # Fallback or store None?
            serializable_config = None # Avoid saving potentially problematic object

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': serializable_config # Save the potentially converted config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        try:
            torch.save(checkpoint, path)
            self.logger.info(f"Checkpoint saved successfully to {path}")
            if is_best:
                 # Optionally create a symlink or copy for 'best_model.pt'
                 best_path = os.path.join(os.path.dirname(path), 'best_model.pt')
                 # Simple copy for cross-platform compatibility
                 # Add check to prevent copying onto itself
                 if os.path.abspath(path) != os.path.abspath(best_path):
                     import shutil
                     shutil.copyfile(path, best_path)
                     self.logger.info(f"Updated best model link to {best_path}")
                 else:
                     # Log that we are saving directly as best_model.pt
                     self.logger.info(f"Saved best model directly as {best_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {path}: {e}", exc_info=True)

    def load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """Loads the trainer state from a checkpoint file.

        Returns:
            The loaded config dictionary from the checkpoint, or None if loading failed.
        """
        if not os.path.exists(path):
            self.logger.error(f"Checkpoint file not found: {path}")
            return None

        loaded_config = None
        try:
            # Load checkpoint onto the correct device directly
            checkpoint = torch.load(path, map_location=self.device)

            # Load model state
            if 'model_state_dict' in checkpoint:
                # Handle potential DataParallel/DDP wrapping
                state_dict = checkpoint['model_state_dict']
                # Simple check for keys starting with 'module.'
                if any(key.startswith('module.') for key in state_dict.keys()):
                    self.logger.info("Detected 'module.' prefix in checkpoint state_dict, attempting to load into unwrapped model.")
                    # Create a new state_dict without the prefix
                    new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(state_dict)
            else:
                self.logger.warning("Checkpoint does not contain 'model_state_dict'.")

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.logger.warning("Checkpoint does not contain 'optimizer_state_dict'. Optimizer state not loaded.")

            # Load scheduler state
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif self.scheduler:
                self.logger.warning("Checkpoint does not contain 'scheduler_state_dict'. Scheduler state not loaded.")

            # Load scaler state for AMP
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            elif self.use_amp:
                self.logger.warning("Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.")

            # Load training state (epoch, step, etc.)
            self.current_epoch = checkpoint.get('epoch', 0) # Default to 0 if not found
            self.global_step = checkpoint.get('global_step', 0) 
            self.loaded_global_step = self.global_step # Store the step we are resuming *from*
            self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
            self.metrics = checkpoint.get('metrics', {'train_loss': [], 'val_loss': []})
            # *** Store loaded config ***
            loaded_config = checkpoint.get('config')

            self.logger.info(f"Successfully loaded checkpoint from {path} at epoch {self.current_epoch}, step {self.global_step}")
            return loaded_config # Return the loaded config

        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found during load attempt: {path}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {path}: {e}", exc_info=True)
            return None
            # Optionally re-raise or exit, depending on desired behavior
            # raise e

    # --- Callback Hook Methods --- (Similar to base.py)

    def _callback_on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
             if hasattr(callback, 'on_train_begin'): callback.on_train_begin(logs=logs)

    def _callback_on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
             if hasattr(callback, 'on_train_end'): callback.on_train_end(logs=logs)

    # Add definitions for the step callbacks
    def _callback_on_step_begin(self, step, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_step_begin'): callback.on_step_begin(step, logs=logs)

    def _callback_on_step_end(self, step, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_step_end'): callback.on_step_end(step, logs=logs)

    # Need epoch begin/end hooks too
    def _callback_on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'): callback.on_epoch_begin(epoch, logs=logs)

    def _callback_on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'): callback.on_epoch_end(epoch, logs=logs)

    def _generate_sample_and_log(self):
        """Generates a text sample using the current model state and logs it."""
        self.logger.info("Attempting to generate sample...")
        # Revert checks to use vocab_path
        if not self.vocab_path:
            self.logger.warning("Cannot generate sample: vocab_path not provided to Trainer.")
            return
        if not os.path.exists(self.vocab_path):
             self.logger.warning(f"Cannot generate sample: vocab_path '{self.vocab_path}' not found.")
             return

        # Revert generation logic to use CharDataset for vocab/decode
        try:
            # Instantiate CharDataset temporarily to access vocab and decode method
            # Provide a VALID dummy file_path (e.g., vocab_path itself if text file)
            # If vocab_path points to a JSON, need a placeholder text file.
            # Let's assume vocab_path IS the text file for simplicity here, adjust if needed.
            try:
                 with open(self.vocab_path, 'r') as f:
                     _ = f.read(1) # Check if readable as text
                 dummy_file_path = self.vocab_path
            except:
                 # If vocab_path is likely JSON, create a dummy text file path reference
                 dummy_file_path = os.path.splitext(self.vocab_path)[0] + ".dummy.txt"
                 if not os.path.exists(dummy_file_path):
                      with open(dummy_file_path, 'w') as f: f.write("dummy")
                      self.logger.debug(f"Created dummy file {dummy_file_path} for CharDataset init.")
            
            temp_dataset = CharDataset(file_path=dummy_file_path, block_size=1, vocab_path=self.vocab_path)
            char_to_idx = temp_dataset.char_to_idx
            idx_to_char = temp_dataset.idx_to_char # Get idx_to_char map
            decode_method = temp_dataset.decode # Get the decode method

            # Get sampling parameters from config (Corrected Access if using self.config)
            gen_config = self.config.get('generation', {})
            start_prompt = gen_config.get('start_prompt', "The ")
            max_new_tokens = gen_config.get('max_new_tokens', 100)
            temperature = gen_config.get('temperature', 0.8)
            top_k = gen_config.get('top_k', None) # Use None for generate_text
            top_p = gen_config.get('top_p', None) # Use None for generate_text

            self.model.eval() # Set model to evaluation mode

            # Call the original generate_text function
            generated_text = generate_text(
                 model=self.model,
                 char_to_idx=char_to_idx,
                 idx_to_char=idx_to_char,
                 seed_text=start_prompt,
                 max_length=max_new_tokens, # Pass max_new_tokens as max_length
                 temperature=temperature,
                 top_k=top_k if top_k is not None else 0, # Convert None to 0 for old func
                 top_p=top_p if top_p is not None else 0.0, # Convert None to 0.0
                 device=self.device
            )
            self.logger.info(f"\n--- Generated Sample (Step {self.global_step}) ---\n{generated_text}\n-------------------------------------")

        except FileNotFoundError:
             self.logger.warning(f"Cannot generate sample: vocab_path '{self.vocab_path}' not found during temporary dataset init.")
        except Exception as e:
            self.logger.error(f"Failed to generate sample: {e}", exc_info=True)
        finally:
            self.model.train()

