import logging
import numpy as np

from .base import Callback

logger = logging.getLogger(__name__)

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
                # Reached min_lr or factor didn't reduce LR
                self.logger.info(f"Patience exceeded ({self.wait}/{self.patience}) but LR not reduced (already at min_lr or factor ineffective).")
                self.wait = 0 # Reset wait counter even if LR not reduced

    def on_train_begin(self, trainer, logs=None):
        """Reset state at the beginning of training."""
        self.wait = 0
        self.cooldown_counter = 0
        self.best_loss = float('inf')
        self.recent_losses = []
        self.logger.info(f"ReduceLROnPlateauOrInstability enabled. Monitoring: '{self.monitor}'")

    # Implement other abstract methods as no-ops if not used
    def on_train_end(self, trainer, logs=None):
        pass

    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass

    def on_epoch_end(self, trainer, epoch, logs=None):
        pass

    def on_step_begin(self, step: int, logs=None):
        pass 