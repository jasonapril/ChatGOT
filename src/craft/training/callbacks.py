"""
Callbacks Module
===============

This module provides a flexible callback system for training events.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler # Assuming it's used elsewhere or will be
import os
from hydra.core.hydra_config import HydraConfig # Import HydraConfig
import datetime # Import datetime

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Abstract base class for callbacks."""
    def __init__(self):
        self.trainer = None # Initialize trainer attribute
        # Get a logger specific to the callback subclass
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_trainer(self, trainer):
        """Set the trainer instance for this callback."""
        self.trainer = trainer

    @abstractmethod
    def on_train_begin(self, trainer, logs=None):
        pass

    @abstractmethod
    def on_train_end(self, trainer, logs=None):
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, epoch, logs=None):
        pass

    @abstractmethod
    def on_step_begin(self, step: int, logs=None):
        pass

    @abstractmethod
    def on_step_end(self, step: int, logs=None):
        pass

class CallbackList:
    """Container for managing a list of callbacks."""
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks if callbacks is not None else []
        self.trainer = None # Add trainer attribute

    def set_trainer(self, trainer):
        """Set the trainer instance for this list and all contained callbacks."""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training begins."""
        if not self.trainer:
            logger.warning("CallbackList.on_train_begin called before trainer was set.")
            return # Or raise error
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(self.trainer, logs=logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training ends."""
        if not self.trainer:
            logger.warning("CallbackList.on_train_end called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(self.trainer, logs=logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        if not self.trainer:
            logger.warning("CallbackList.on_epoch_begin called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(self.trainer, epoch, logs=logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        if not self.trainer:
            logger.warning("CallbackList.on_epoch_end called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(self.trainer, epoch, logs=logs)

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each training step."""
        if not self.trainer:
            logger.warning("CallbackList.on_step_begin called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_begin(step, logs=logs)

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each training step."""
        if not self.trainer:
            logger.warning("CallbackList.on_step_end called before trainer was set.")
            return
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
        # Check for missing trainer or tokenizer first
        if not self.trainer or not self.tokenizer:
            self.logger.error("Trainer or tokenizer not set properly.")
            return

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
    """Logs training metrics to TensorBoard."""
    def __init__(self, log_dir: Optional[str] = None, experiment_id: Optional[str] = None, create_dirs: bool = True):
        self.log_dir_base = log_dir
        self.experiment_id = experiment_id
        self.create_dirs = create_dirs
        self.writer = None
        self.log_dir_absolute = None
        self.logger = logging.getLogger(self.__class__.__name__)
        # Initialize rank (default to 0 for non-distributed)
        self._rank = int(os.environ.get("RANK", "0"))

    def _initialize_writer(self):
        """Initializes the TensorBoard SummaryWriter."""
        if self._rank == 0:
            try:
                if not hasattr(self, 'log_dir_absolute') or not self.log_dir_absolute:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Use base dir and experiment ID if provided
                    if self.log_dir_base and self.experiment_id:
                        log_dir_relative = os.path.join(self.log_dir_base, self.experiment_id)
                    elif self.log_dir_base:
                        log_dir_relative = os.path.join(self.log_dir_base, timestamp)
                    else: # Default if nothing provided
                        log_dir_relative = os.path.join("outputs", "tensorboard", timestamp)

                    self.log_dir_absolute = os.path.abspath(log_dir_relative)
                    self.logger.info(f"Generated new TensorBoard log directory: {self.log_dir_absolute}")

                if self.create_dirs:
                    os.makedirs(self.log_dir_absolute, exist_ok=True)
                
                self.writer = SummaryWriter(log_dir=self.log_dir_absolute)
                self.logger.info(f"TensorBoardLogger initialized. Logging to: {self.log_dir_absolute}")
            except Exception as e:
                self.logger.error(f"Failed to initialize TensorBoard SummaryWriter in {getattr(self, 'log_dir_absolute', 'unknown_path')}: {e}", exc_info=True)
                self.writer = None

    def set_log_dir_absolute(self, path: str):
        """Allows setting the absolute log directory externally, e.g., during resume."""
        if self._rank == 0:
            self.log_dir_absolute = path
            self.log_dir_pattern = None # Indicate that a specific path was set
            self.logger.info(f"TensorBoard log directory explicitly set to: {path}")

    def on_train_begin(self, trainer, logs=None):
        """Initialize the SummaryWriter at the beginning of training."""
        if self._rank == 0:
            try:
                # If an absolute path wasn't already set (e.g., by resume), create one.
                if not hasattr(self, 'log_dir_absolute') or not self.log_dir_absolute:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Use base dir and experiment ID if provided
                    if self.log_dir_base and self.experiment_id:
                        log_dir_relative = os.path.join(self.log_dir_base, self.experiment_id)
                    elif self.log_dir_base:
                        log_dir_relative = os.path.join(self.log_dir_base, timestamp)
                    else: # Default if nothing provided
                        log_dir_relative = os.path.join("outputs", "tensorboard", timestamp)

                    self.log_dir_absolute = os.path.abspath(log_dir_relative)
                    self.logger.info(f"Generated new TensorBoard log directory: {self.log_dir_absolute}")
                # else: log_dir_absolute was set by set_log_dir_absolute

                if self.create_dirs:
                     os.makedirs(self.log_dir_absolute, exist_ok=True)
                
                self.writer = SummaryWriter(log_dir=self.log_dir_absolute)
                self.logger.info(f"TensorBoardLogger initialized. Logging to: {self.log_dir_absolute}")
            except Exception as e:
                # Use self.logger for error logging
                self.logger.error(f"Failed to initialize TensorBoard SummaryWriter in {getattr(self, 'log_dir_absolute', 'unknown_path')}: {e}", exc_info=True)
                self.writer = None # Ensure writer is None if init fails

    def on_step_end(self, step: int, logs=None):
        """Log step-level metrics."""
        if self.writer and self._rank == 0 and logs:
            loss = logs.get('loss')
            lr = logs.get('lr') # Assuming TrainingLoop adds LR to logs
            if loss is not None:
                self.writer.add_scalar('Loss/train_step', loss, step)
            if lr is not None:
                 self.writer.add_scalar('LearningRate', lr, step)
            # Add logging for other potential items in logs if needed
            # for key, value in logs.items():
            #     if key not in ['loss', 'lr'] and isinstance(value, (int, float)):
            #         self.writer.add_scalar(f'Step/{key}', value, step)

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics, logs=None):
        """Log epoch-level metrics."""
        if self.writer and self._rank == 0:
            if train_metrics:
                for key, value in train_metrics.items():
                     if isinstance(value, (int, float)):
                         self.writer.add_scalar(f'Epoch/{key}_train', value, trainer.state.epoch)
            if val_metrics:
                for key, value in val_metrics.items():
                     if isinstance(value, (int, float)):
                         self.writer.add_scalar(f'Epoch/{key}_val', value, trainer.state.epoch)
            # Optionally log histograms of weights/gradients here

    def on_train_end(self, trainer, logs=None):
        """Close the SummaryWriter at the end of training."""
        if self.writer and self._rank == 0:
            self.writer.close()
            logger.info("TensorBoardLogger writer closed.")

    def on_epoch_begin(self, trainer, epoch, logs=None):
        """Called at the beginning of each epoch."""
        pass # No action needed for TensorBoardLogger

    def on_step_begin(self, step: int, logs=None):
        """Called at the beginning of each training step."""
        pass # No action needed for TensorBoardLogger
    
    def __del__(self):
        """Ensure writer is closed if object is deleted unexpectedly."""
        if hasattr(self, 'writer') and self.writer:
            try:
                self.writer.close()
            except Exception as e:
                # Log error during cleanup if necessary
                pass # Avoid errors during garbage collection

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        monitor (str): Quantity to be monitored. Default: 'val_loss'.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0.
        patience (int): Number of epochs with no improvement after which training will be stopped. Default: 10.
        verbose (int): Verbosity mode, 0 or 1. Default: 0.
        mode (str): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity
            monitored has stopped decreasing; in 'max' mode it will stop when the quantity
            monitored has stopped increasing; in 'auto' mode, the direction is automatically inferred
            from the name of the monitored quantity. Default: 'auto'.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0.0.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best
                                     value of the monitored quantity. Default: False.
    """
    def __init__(self, monitor='val_loss', min_delta=0.0, patience=10, verbose=0, mode='auto', delta=0.0, restore_best_weights=False):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        # Use the 'delta' parameter name consistently, rename min_delta internally if needed
        # Keep min_delta for backward compatibility? Or just use delta? Let's use delta.
        self.delta = delta # Changed from min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logger.warning(f"EarlyStopping mode {mode} is unknown, fallback to auto mode.")
            mode = 'auto'

        if mode == 'auto':
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.mode = 'max'
            else:
                self.mode = 'min'
        else:
            self.mode = mode

        if self.mode == 'min':
            self.monitor_op = np.less
            # Use np.inf instead of np.Inf
            self.best = np.inf # Changed from np.Inf
        else:
            self.monitor_op = np.greater
            # Use -np.inf instead of -np.Inf
            self.best = -np.inf # Changed from -np.Inf

        # Adjust comparison based on delta and mode
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.delta)
        else: # mode == 'max'
            self.monitor_op = lambda a, b: np.greater(a, b + self.delta)

    def on_train_begin(self, trainer, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        # Use np.inf/-np.inf consistently
        self.best = np.inf if self.mode == "min" else -np.inf # Changed from np.Inf/-np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, trainer, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Early stopping conditioned on metric `{self.monitor}` which is not available. Available metrics are: {','.join(list(logs.keys()))}")
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                # TODO: Implement weight saving/loading mechanism
                # self.best_weights = self.model.get_weights()
                logger.info("Saving best model weights...")
                # Placeholder for actual weight saving
                self.best_weights = {name: param.clone().detach().cpu() for name, param in trainer.model.named_parameters()}

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        logger.info(f"Restoring model weights from the end of the best epoch ({self.best_epoch+1}).")
                    # Placeholder for actual weight loading
                    state_dict = trainer.model.state_dict()
                    state_dict.update(self.best_weights)
                    trainer.model.load_state_dict(state_dict)
                    logger.info("Restored best model weights.")


                if self.verbose > 0:
                    logger.info(f"Epoch {self.stopped_epoch + 1}: early stopping")


    def on_epoch_begin(self, trainer, epoch, logs=None):
        # No action needed on epoch begin for EarlyStopping
        pass

    def on_step_begin(self, trainer, batch_idx, logs=None):
        # No action needed on step begin for EarlyStopping
        pass

    def on_step_end(self, trainer, batch_idx, logs=None):
        # No action needed on step end for EarlyStopping
        pass

    def on_train_end(self, trainer, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
             logger.info(f"Epoch {self.stopped_epoch + 1}: early stopping")

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