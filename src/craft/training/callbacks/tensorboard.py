import logging
import os
import datetime
from typing import Dict, Any, Optional
from hydra.core.hydra_config import HydraConfig
import torch

# Conditional import of SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    _TENSORBOARD_AVAILABLE = False

from .base import Callback

logger = logging.getLogger(__name__)

class TensorBoardLogger(Callback):
    """
    Callback that logs metrics to TensorBoard.

    Logs loss, learning rate, and other specified metrics during training.
    """
    def __init__(self, log_dir: Optional[str] = None, comment: str = "", flush_secs: int = 120):
        """
        Initializes the TensorBoardLogger.

        Args:
            log_dir (Optional[str]): Specific directory to save TensorBoard logs.
                                     If None, defaults to './tensorboard_logs/<timestamp>'
                                     or tries to use 'tensorboard/' under Hydra run dir.
            comment (str): Comment to append to the default log_dir.
            flush_secs (int): How often, in seconds, to flush the pending events and summaries to disk.
        """
        super().__init__()
        self.comment = comment
        self.flush_secs = flush_secs
        self.writer = None
        self.log_dir_config = log_dir # Store the configured log_dir
        self.resolved_log_dir = None # Will be set in _initialize_writer
        # self.hparams_logged = False # Removed hparams logic for simplicity

        if not _TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not found. Install tensorboard (e.g., pip install tensorboard) to use TensorBoardLogger.")

    def set_trainer(self, trainer):
        """Set the trainer and attempt to get experiment name."""
        super().set_trainer(trainer)
        # Attempt to get experiment name from the main config attached to the trainer
        if hasattr(trainer, 'config') and hasattr(trainer.config, 'experiment_name'):
            self.experiment_name = trainer.config.experiment_name
            self.logger.info(f"Retrieved experiment_name from trainer config: {self.experiment_name}")
        else:
             # Get experiment name from training config if available
            if hasattr(trainer, 'training_config') and hasattr(trainer.training_config, 'experiment_name'):
                 self.experiment_name = trainer.training_config.experiment_name
                 self.logger.info(f"Retrieved experiment_name from trainer training_config: {self.experiment_name}")
            else:
                self.logger.warning("Could not find 'experiment_name' in trainer.config or trainer.training_config. Fallback log path construction might fail.")

    def set_log_dir_absolute(self, path: str):
        """Allows external setting of the absolute log directory, primarily for resuming."""
        self.logger.info(f"Externally setting TensorBoard log directory to: {path}")
        self.log_dir_absolute = path

    def _initialize_writer(self) -> None:
        """Initializes the SummaryWriter."""
        if self.writer:
            return

        effective_log_dir = self.log_dir_config # Prioritize explicitly passed log_dir

        if effective_log_dir:
            self.logger.info(f"Using configured log_dir: {effective_log_dir}")
        else:
            # Fallback logic if no explicit log_dir was passed via config/injection
            self.logger.warning("No explicit log_dir provided.")
            hydra_run_dir = None
            try:
                hydra_run_dir = HydraConfig.get().run.dir
                effective_log_dir = os.path.join(hydra_run_dir, "tensorboard")
                self.logger.info(f"Attempting to use Hydra run directory: {effective_log_dir}")
            except Exception:
                self.logger.error("Could not get Hydra run directory. Using default log directory: ./tensorboard_logs/<timestamp>")
                effective_log_dir = os.path.join(".", "tensorboard_logs", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.comment}")
        
        # Resolve the path correctly, avoiding abspath if already absolute (e.g., from Hydra)
        if os.path.isabs(effective_log_dir):
            self.resolved_log_dir = effective_log_dir
        else:
            # If relative, make it absolute relative to the original CWD if possible
            try:
                from hydra.utils import get_original_cwd
                original_cwd = get_original_cwd()
                self.resolved_log_dir = os.path.abspath(os.path.join(original_cwd, effective_log_dir))
            except ImportError:
                 self.logger.warning("Hydra utils not found, resolving path relative to current working directory.")
                 self.resolved_log_dir = os.path.abspath(effective_log_dir)
            except Exception as e:
                 self.logger.warning(f"Could not get original CWD ({e}), resolving path relative to current working directory.")
                 self.resolved_log_dir = os.path.abspath(effective_log_dir)

        self.logger.info(f"TensorBoard logs will be saved to: {self.resolved_log_dir}")
        os.makedirs(self.resolved_log_dir, exist_ok=True)

        try:
            # Initialize the writer
            self.writer = SummaryWriter(log_dir=self.resolved_log_dir)
            self.logger.info(f"TensorBoard SummaryWriter initialized. Logging to: {self.resolved_log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SummaryWriter: {e}", exc_info=True)
            self.writer = None # Ensure writer is None if init fails

    def on_train_begin(self, trainer, logs=None):
        """Initialize the SummaryWriter at the start of training."""
        self._initialize_writer()
        
        # Log hyperparameters if available (and writer initialized)
        if self.writer and hasattr(trainer, 'config') and trainer.config:
             # Assuming trainer.config is a Pydantic model or dict-like
            try:
                # Convert Pydantic model to dict if necessary
                if hasattr(trainer.config, 'model_dump'):
                     hparams_dict = trainer.config.model_dump()
                elif isinstance(trainer.config, dict):
                    hparams_dict = trainer.config
                else:
                    hparams_dict = vars(trainer.config)
                
                # Filter out non-scalar values if necessary, or flatten
                scalar_hparams = {
                    k: v for k, v in hparams_dict.items()
                    if isinstance(v, (int, float, str, bool))
                }
                
                # Attempt to log hyperparameters
                # Note: add_hparams requires a metric_dict, which we don't have yet.
                # A common practice is to log hparams with final metrics at the end.
                # For now, we can log them as text or just store them.
                # self.writer.add_hparams(scalar_hparams, {})
                self.logger.info(f"Hyperparameters logged (to be fully associated later): {scalar_hparams}")
                self.logged_hparams = scalar_hparams # Store for later

            except Exception as e:
                self.logger.error(f"Could not log hyperparameters: {e}")


    def on_step_end(self, step, logs=None):
        """Log metrics to TensorBoard at the end of each step."""
        if not self.writer or logs is None:
            return

        global_step = logs.get('global_step', step) # Use global_step if available

        # Log training loss
        loss = logs.get('loss')
        if loss is not None:
            self.writer.add_scalar('Loss/train', loss, global_step)

        # Log learning rate
        lr = logs.get('lr')
        if lr is not None:
            self.writer.add_scalar('LearningRate', lr, global_step)
        elif hasattr(self.trainer, 'optimizer'):
             # Fallback: try to get LR directly from optimizer
            try:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LearningRate', current_lr, global_step)
            except (AttributeError, IndexError) as e:
                 self.logger.debug(f"Could not retrieve learning rate from optimizer: {e}")


    def on_epoch_end(self, trainer, epoch: int, train_metrics: dict, val_metrics: dict, logs: Optional[Dict[str, float]] = None):
        """Log metrics at the end of an epoch.

        Args:
            trainer: The Trainer instance.
            epoch: The epoch number completed.
            train_metrics: Dictionary containing training metrics for the epoch.
            val_metrics: Dictionary containing validation metrics for the epoch.
            logs: Deprecated combined logs dictionary (ignored).
        """
        if self.writer:
            global_step = trainer.global_step
            if not global_step:
                 self.logger.warning("Cannot log epoch metrics: trainer.global_step not found.")
                 return
                 
            # Log validation loss if available
            if val_metrics and 'loss' in val_metrics and val_metrics['loss'] is not None:
                self.writer.add_scalar('Loss/validation', val_metrics['loss'], global_step)

            # Log other validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    if key != 'loss' and value is not None:
                        self.writer.add_scalar(f'Metrics/val_{key}', value, global_step)

            # Log training metrics (excluding loss, which is logged per step)
            if train_metrics:
                 for key, value in train_metrics.items():
                    # Exclude train loss (logged per step) and timing info
                     if key != 'loss' and 'time' not in key and value is not None:
                         self.writer.add_scalar(f'Metrics/train_{key}', value, global_step)


    def on_train_end(self, trainer, logs: Optional[Dict] = None):
        """Close the SummaryWriter at the end of training."""
        if self.writer:
            # Log hparams with final metrics if possible
            if hasattr(self, 'logged_hparams') and logs:
                 final_metrics = {
                     f'hparam/{k}': v for k, v in logs.items()
                     if isinstance(v, (int, float))
                 }
                 # Filter hparams again to ensure they are scalars
                 scalar_hparams = {k: v for k, v in self.logged_hparams.items() if isinstance(v, (int, float, str, bool))}
                 try:
                     self.writer.add_hparams(scalar_hparams, final_metrics)
                 except Exception as e:
                     self.logger.error(f"Failed to log final hyperparameters: {e}")

            self.writer.close()
            self.logger.info("TensorBoard writer closed.")
            self.writer = None
            self.log_dir_absolute = None # Clear path

    # Implement other abstract methods as no-ops
    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass

    def on_step_begin(self, step: int, logs=None):
        pass 