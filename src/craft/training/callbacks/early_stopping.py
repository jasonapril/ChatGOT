import logging
import numpy as np
import torch
import os
from typing import Dict, Any, Optional, Union, Callable

from .base import Callback, TYPE_CHECKING

# Added TYPE_CHECKING block
if TYPE_CHECKING:
    from ..trainer import Trainer

logger = logging.getLogger(__name__)

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Based on Keras' EarlyStopping callback.
    """
    def __init__(self, monitor: str ='val_loss', min_delta: float = 0, patience: int = 10,
                 verbose: bool = True, mode: str ='auto', restore_best_weights: bool = False) -> None:
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
        self.monitor: str = monitor
        self.min_delta: float = min_delta
        self.patience: int = patience
        self.verbose: bool = verbose
        self.mode: str = mode
        self.restore_best_weights: bool = restore_best_weights
        # Type hint for monitor_op
        self.monitor_op: Callable[[Any, Any], bool]
        # Declare best attribute here
        self.best: float

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
        self.wait: int = 0
        self.stopped_epoch: int = 0
        # self.best_weights = None # No longer store weights directly here

    # Corrected signature
    def on_train_begin(self, **kwargs: Any) -> None:
        """Reset state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        # self.best_weights = None # Reset removed as weights are not stored
        self.logger.info(f"EarlyStopping enabled. Monitoring: '{self.monitor}', Mode: '{self.mode}', Patience: {self.patience}")

    # Corrected signature
    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Check if training should stop based on the monitored metric."""
        current = metrics.get(self.monitor)

        if current is None:
            self.logger.warning(f"Early stopping conditioned on metric '{self.monitor}' which is not available. Available metrics are: {list(metrics.keys())}")
            return

        # Ensure metric is numeric
        if not isinstance(current, (int, float)):
            self.logger.warning(f"Metric '{self.monitor}' for early stopping is not numeric ({type(current)}). Skipping check.")
            return
        current = float(current)

        # Check for improvement
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                if self.trainer and hasattr(self.trainer, 'checkpoint_manager') and self.trainer.checkpoint_manager:
                    self.logger.info(f"Attempting to restore best model weights via CheckpointManager.")
                    try:
                        self.logger.warning("Restoring best weights via EarlyStopping is currently not supported. CheckpointManager needs specific functionality (e.g., load_best_checkpoint).")
                    except Exception as e:
                        self.logger.error(f"Error during placeholder check for best weight restoration: {e}")
                else:
                    self.logger.warning("restore_best_weights=True but CheckpointManager not available or best epoch not recorded. Weights not restored.")
        else:
            self.wait += 1
            self.logger.debug(f"Metric '{self.monitor}' did not improve from {self.best:.4f}. Wait: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                # Set the trainer's stop flag
                if self.trainer and hasattr(self.trainer, '_stop_training'):
                    self.trainer._stop_training = True
                    self.logger.info(f"Epoch {self.stopped_epoch + 1}: early stopping triggered after {self.patience} epochs with no improvement.")
                else:
                    self.logger.error("Trainer does not have a '_stop_training' attribute. Cannot signal early stop.")

    # Corrected signature
    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if self.stopped_epoch > 0 and self.verbose:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch + 1}")

    # Corrected signature
    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        pass

    # Corrected signature
    def on_step_begin(self, step: int, **kwargs: Any) -> None:
        pass

    # Corrected signature
    def on_step_end(self, step: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        pass 