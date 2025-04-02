"""
Callbacks Module
===============

This module provides a flexible callback system for training events.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Callback(ABC):
    """Base class for all callbacks."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trainer = None

    def set_trainer(self, trainer):
        """Set the trainer instance for this callback."""
        self.trainer = trainer

    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training begins."""
        pass

    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training ends."""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each training step."""
        pass

    @abstractmethod
    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each training step."""
        pass

class CallbackList:
    """Manages a list of callbacks and their execution."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks if callbacks is not None else []
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_trainer(self, trainer):
        """Set the trainer instance for all callbacks."""
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training begins."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training ends."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each training step."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_begin(step, logs=logs)

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each training step."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_end(step, logs=logs)

    def append(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def extend(self, callbacks: List[Callback]):
        """Add multiple callbacks to the list."""
        self.callbacks.extend(callbacks)

# --- Specific Callback Implementations --- #

class ReduceLROnPlateauOrInstability(Callback):
    """
    Reduces learning rate when loss plateaus or spikes, indicating instability.

    Monitors the training loss and reduces the learning rate of the optimizer
    if the loss increases significantly compared to a moving average or stays high.
    Also includes cooldown period after reduction.
    """
    def __init__(self, monitor='loss', factor=0.5, patience=10, threshold=1.5,
                 min_lr=1e-7, cooldown=5, window_size=20, verbose=True):
        """
        Args:
            monitor (str): The metric key in logs to monitor (default: 'loss').
            factor (float): Factor by which the learning rate will be reduced (new_lr = lr * factor).
            patience (int): Number of steps with increasing/high loss to wait before reducing LR.
            threshold (float): Factor increase over moving average loss to consider as instability spike.
            min_lr (float): Lower bound on the learning rate.
            cooldown (int): Number of steps to wait before resuming normal operation after LR has been reduced.
            window_size (int): Size of the moving average window for loss.
            verbose (bool): If True, prints a message when LR is reduced.
        """
        super().__init__()
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.window_size = window_size
        self.verbose = verbose

        # Internal state
        self.wait = 0           # Steps waited for improvement/stability
        self.cooldown_counter = 0 # Steps remaining in cooldown
        self.best_loss = float('inf') # Best loss observed so far (or lowest avg)
        self.recent_losses = [] # Track recent losses for moving average

    def set_trainer(self, trainer):
        """Set the trainer instance for this callback."""
        super().set_trainer(trainer)
        if hasattr(trainer, 'optimizer'):
            self.optimizer = trainer.optimizer
        else:
            self.logger.error("Trainer does not have an optimizer attribute")

    def _get_lr(self):
        """Get current learning rate from the optimizer."""
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        return None

    def _set_lr(self, new_lr):
        """Set new learning rate for the optimizer."""
        if hasattr(self, 'optimizer'):
            old_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                self.optimizer.param_groups[0]['lr'] = new_lr
                if self.verbose:
                    self.logger.warning(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            else:
                self.logger.debug(f"Attempted to set LR to {new_lr:.2e}, but it's not lower than current {old_lr:.2e}")
        else:
            self.logger.error("Optimizer not set in ReduceLROnPlateauOrInstability callback.")

    def on_step_end(self, step, logs=None):
        """Called at the end of each training step."""
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        current_lr = self._get_lr()

        if current_loss is None or current_lr is None:
            return # Cannot operate without loss and current LR

        # Add current loss to recent history
        self.recent_losses.append(current_loss)
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)

        # Calculate moving average if window is full
        moving_avg_loss = np.mean(self.recent_losses) if len(self.recent_losses) >= self.window_size else None

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0 # Reset wait counter during cooldown
            return # Do nothing during cooldown

        # --- Refined Instability Check --- #
        is_spike = False
        if moving_avg_loss is not None:
            # Update best loss if moving average improves
            if moving_avg_loss < self.best_loss:
                self.best_loss = moving_avg_loss
                self.logger.debug(f"New best average loss: {self.best_loss:.4f}")

            # Check for instability spike against the *best* loss average seen so far
            if current_loss > self.best_loss * self.threshold:
                is_spike = True
                self.wait += 1
                self.logger.debug(f"Loss spike detected ({current_loss:.4f} > {self.best_loss:.4f} * {self.threshold:.2f}). Wait: {self.wait}/{self.patience}")

        # Reset wait counter only if loss is NOT spiking relative to best loss
        if not is_spike:
            if self.wait > 0:
                self.logger.debug(f"Loss stabilized ({current_loss:.4f} <= {self.best_loss:.4f} * {self.threshold:.2f}). Resetting wait counter.")
            self.wait = 0

        # Check if patience is exceeded
        if self.wait >= self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            if new_lr < current_lr:
                self._set_lr(new_lr)
                self.cooldown_counter = self.cooldown # Start cooldown
                self.wait = 0 # Reset wait counter
                self.recent_losses = [] # Reset loss history after LR change
                self.best_loss = float('inf') # Reset best loss
            else:
                # LR already at minimum, reset wait
                self.wait = 0
                self.logger.info(f"Patience exceeded ({self.wait}/{self.patience}), but LR already at minimum ({current_lr:.2e}).")

    def on_train_begin(self, logs=None):
        """Called when training begins."""
        logs = logs or {}
        self.wait = 0
        self.cooldown_counter = 0
        self.best_loss = float('inf') # Reset best loss to infinity
        self.recent_losses = []
        self.initial_lr = self._get_lr()

    def on_train_end(self, logs=None):
        """Called when training ends."""
        logs = logs or {}
        # No cleanup needed

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        logs = logs or {}
        # No action needed at epoch start

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        # No action needed at epoch end

    def on_step_begin(self, step, logs=None):
        """Called at the beginning of each training step."""
        logs = logs or {}
        # No action needed at step start

class SampleGenerationCallback(Callback):
    """
    Generates text samples periodically during training.
    """
    def __init__(self, tokenizer, prompt="Once upon a time", sample_every_n_steps=0,
                 sample_on_epoch_end=True, max_new_tokens=50, temperature=0.7,
                 top_k=50, num_samples=1):
        """
        Args:
            tokenizer: The tokenizer for encoding the prompt and decoding the output.
            prompt (str): The text prompt to use for generation.
            sample_every_n_steps (int): Generate sample every N global steps. If 0, disable step-based sampling.
            sample_on_epoch_end (bool): Generate sample at the end of each epoch.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling parameter.
            num_samples (int): Number of samples to generate each time.
        """
        super().__init__()
        if tokenizer is None:
             raise ValueError("SampleGenerationCallback requires a tokenizer.")
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_on_epoch_end = sample_on_epoch_end
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.num_samples = num_samples
        self.device = None # Will be set from trainer

    def set_trainer(self, trainer):
        super().set_trainer(trainer)
        # Try to get device from trainer
        if hasattr(trainer, 'device'):
            self.device = trainer.device
        else:
            self.logger.warning("Could not determine device from trainer for sample generation.")
            # Fallback or raise error?
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_samples(self, step_or_epoch_info):
        """Internal method to perform sample generation and logging."""
        if not hasattr(self.trainer.model, 'generate') or not callable(self.trainer.model.generate):
            self.logger.warning(f"Model {self.trainer.model.__class__.__name__} does not have a callable 'generate' method. Skipping sample generation.")
            return

        self.logger.info(f"--- Generating Sample @ {step_or_epoch_info} ---")
        self.trainer.model.eval() # Set model to evaluation mode

        try:
            # Encode the prompt
            # Assuming tokenizer returns a dict with 'input_ids' and works like HF tokenizers
            # Add batch dimension and move to device
            inputs = self.tokenizer(self.prompt, return_tensors="pt")
            # Use dictionary access for tokenizer output
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            self.logger.info(f"Prompt: \"{self.prompt}\"...")

            # Generate samples
            with torch.no_grad():
                 # Ensure model is on the correct device (should be handled by Trainer)
                 # Call the model's generate method
                 outputs = self.trainer.model.generate(
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     max_new_tokens=self.max_new_tokens,
                     temperature=self.temperature,
                     top_k=self.top_k,
                     num_return_sequences=self.num_samples,
                     do_sample=True, # Ensure sampling is enabled
                     pad_token_id=self.tokenizer.eos_token_id # Common practice
                 )

            # Decode and log each sample
            for i, output_ids in enumerate(outputs):
                # Remove prompt tokens from the start if necessary (depends on generate output)
                # Assuming output includes prompt tokens
                output_ids_no_prompt = output_ids[input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(output_ids_no_prompt, skip_special_tokens=True)
                self.logger.info(f"Sample {i+1}: {generated_text}")

        except Exception as e:
            self.logger.error(f"Error during sample generation: {e}", exc_info=True)

        finally:
            self.trainer.model.train() # Ensure model is back in training mode
            self.logger.info("--- End Sample Generation ---")

    def on_step_end(self, step, logs=None):
        if self.sample_every_n_steps > 0 and step > 0 and step % self.sample_every_n_steps == 0:
            self._generate_samples(f"Step {step}")

    def on_epoch_end(self, epoch, logs=None):
        if self.sample_on_epoch_end:
            self._generate_samples(f"Epoch {epoch + 1} End")

    # --- Add required abstract methods (even if empty) ---
    def on_train_begin(self, logs=None):
        pass # No action needed

    def on_train_end(self, logs=None):
        pass # No action needed

    def on_epoch_begin(self, epoch, logs=None):
        pass # No action needed

    def on_step_begin(self, step, logs=None):
        pass # No action needed

class TensorBoardLogger(Callback):
    """
    Logs metrics to TensorBoard.
    """
    def __init__(self, log_dir="runs/experiment"):
        """
        Args:
            log_dir (str): Directory where TensorBoard logs will be saved.
                           Consider making this dynamic based on run timestamp/name.
        """
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self, logs=None):
        """Initialize the SummaryWriter at the start of training."""
        try:
            self.writer = SummaryWriter(self.log_dir)
            self.logger.info(f"TensorBoard logging initialized. Log directory: {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard SummaryWriter: {e}", exc_info=True)
            self.writer = None # Ensure writer is None if init fails

    def on_step_end(self, step, logs=None):
        """Log step-level metrics like loss and learning rate."""
        if self.writer and logs:
            # Log scalar values passed in logs
            loss = logs.get('loss')
            lr = logs.get('lr')
            if loss is not None:
                self.writer.add_scalar('Loss/train_step', loss, step)
            if lr is not None:
                self.writer.add_scalar('LearningRate/step', lr, step)
            # Add any other step-level metrics if available in logs

    def on_epoch_end(self, epoch, logs=None):
        """Log epoch-level metrics like train and validation loss."""
        if self.writer and logs:
            # Naming convention might need adjustment based on keys in logs
            train_loss = logs.get('loss') # train_epoch returns {'loss': avg_loss}
            val_loss = logs.get('val_loss') # evaluate returns {'loss': avg_loss}

            if train_loss is not None:
                self.writer.add_scalar('Loss/train_epoch', train_loss, epoch + 1) # Use epoch number (1-based)

            if val_loss is not None:
                self.writer.add_scalar('Loss/validation_epoch', val_loss, epoch + 1)

            # Add any other epoch-level metrics (e.g., perplexity if calculated)

    def on_train_end(self, logs=None):
        """Close the SummaryWriter at the end of training."""
        if self.writer:
            self.writer.close()
            self.logger.info("TensorBoard writer closed.")

    # --- Add required abstract methods (even if empty) ---
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        pass # No action needed

    def on_step_begin(self, step, logs=None):
        """Called at the beginning of each training step."""
        pass # No action needed

# Example of another callback:
# class TensorBoardLogger(TrainerCallback):
#     def __init__(self, log_dir):
#         super().__init__()
#         self.writer = SummaryWriter(log_dir)
#     def on_step_end(self, step, logs=None):
#         if logs:
#             self.writer.add_scalar('Loss/train_step', logs.get('loss'), step)
#             self.writer.add_scalar('LR/train_step', logs.get('lr'), step)
#     def on_epoch_end(self, epoch, logs=None):
#         if logs:
#             self.writer.add_scalar('Loss/train_epoch', logs.get('train_loss'), epoch)
#             if 'val_loss' in logs:
#                 self.writer.add_scalar('Loss/val_epoch', logs.get('val_loss'), epoch)
#     def on_train_end(self, logs=None):
#         self.writer.close() 