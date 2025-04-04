import logging
import numpy as np
import torch
from typing import Dict, Any, Optional

from .base import Callback

logger = logging.getLogger(__name__)

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Based on Keras' EarlyStopping callback.
    """
    def __init__(self, monitor='val_loss', min_delta=0, patience=10,
                 verbose=True, mode='auto', restore_best_weights=False):
        """
        Args:
            monitor (str): Quantity to be monitored (e.g., 'val_loss', 'val_accuracy').
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (bool): If True, prints messages when stopping.
            mode (str): One of {'auto', 'min', 'max'}.
                        In 'min' mode, training stops when the quantity monitored has stopped decreasing.
                        In 'max' mode, training stops when the quantity monitored has stopped increasing.
                        In 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
                                        of the monitored quantity. If False, the model weights obtained at the
                                        last step of training are used. Requires CheckpointManager to be used.
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        if mode not in ['auto', 'min', 'max']:
            self.logger.warning(f"EarlyStopping mode {mode} is unknown, fallback to auto mode.")
            self.mode = 'auto'

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            self.min_delta *= -1
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else: # Auto mode
            if 'acc' in self.monitor:
                self.mode = 'max'
                self.monitor_op = np.greater
                self.best = -np.inf
            else: # Default to min mode for loss, etc.
                self.mode = 'min'
                self.monitor_op = np.less
                self.best = np.inf
                self.min_delta *= -1

        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_begin(self, trainer, logs=None):
        """Reset state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.best_weights = None
        self.logger.info(f"EarlyStopping enabled. Monitoring: '{self.monitor}', Mode: '{self.mode}', Patience: {self.patience}")

    def on_epoch_end(self, trainer, epoch, logs=None):
        """Check if training should stop based on the monitored metric."""
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            self.logger.warning(f"Early stopping conditioned on metric '{self.monitor}' which is not available. Available metrics are: {list(logs.keys())}")
            return

        # Check for improvement
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                if hasattr(trainer, 'checkpoint_manager'):
                    # CheckpointManager handles saving best weights based on its logic.
                    # We no longer store best_epoch directly here.
                    self.logger.debug(f"Best metric improved to {self.best:.4f} at epoch {epoch + 1}. CheckpointManager should handle saving.")
                else:
                    self.logger.warning("restore_best_weights=True requires a CheckpointManager in the Trainer.")
        else:
            self.wait += 1
            self.logger.debug(f"Metric '{self.monitor}' did not improve from {self.best:.4f}. Wait: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                # Set the trainer's stop flag
                if hasattr(trainer, '_stop_training'):
                    trainer._stop_training = True
                    self.logger.info(f"Epoch {self.stopped_epoch + 1}: early stopping triggered after {self.patience} epochs with no improvement.")
                else:
                    self.logger.error("Trainer does not have a '_stop_training' attribute. Cannot signal early stop.")

                # Restore best weights if enabled
                if self.restore_best_weights:
                    if hasattr(trainer, 'checkpoint_manager'):
                        self.logger.info(f"Attempting to restore best model weights via CheckpointManager.")
                        # Construct the expected path for the best checkpoint saved by CheckpointManager
                        # This assumes CheckpointManager saves best checkpoints with a specific naming convention.
                        # We need to know the exact path or how to retrieve it.
                        # Placeholder: Assume CheckpointManager has a method `get_best_checkpoint_path()`
                        try:
                             best_path = trainer.checkpoint_manager.get_best_checkpoint_path() # Needs implementation in CheckpointManager
                             if best_path and os.path.exists(best_path):
                                 # Reloading requires the CheckpointManager to handle state loading
                                 trainer.checkpoint_manager.load_checkpoint(best_path, load_model_only=True) # Need `load_model_only` flag
                             else:
                                 self.logger.warning("Could not find or retrieve best checkpoint path. Weights not restored.")
                        except AttributeError as e:
                             self.logger.error(f"CheckpointManager does not support retrieving best path or loading model only: {e}. Weights not restored.")
                        except Exception as e:
                            self.logger.error(f"Error restoring best weights: {e}")
                    else:
                        self.logger.warning("restore_best_weights=True but CheckpointManager not available or best epoch not recorded. Weights not restored.")

    # Implement other abstract methods as no-ops
    def on_train_end(self, trainer, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch + 1}")

    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass

    def on_step_begin(self, step: int, logs=None):
        pass

    def on_step_end(self, step: int, logs=None):
        pass 