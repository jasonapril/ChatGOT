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
        self.writer: Optional[SummaryWriter] = None
        self.log_dir_config = log_dir # Store the configured log_dir
        self.resolved_log_dir = None # Will be set in _initialize_writer
        self.experiment_name = None # <<< ADD attribute to store experiment name
        self.logged_hparams = None

        if not _TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not found. Install tensorboard (e.g., pip install tensorboard) to use TensorBoardLogger.")

    def _get_current_lr(self, optimizer: Optional[torch.optim.Optimizer]) -> Optional[float]:
        """Safely get the current learning rate from the optimizer."""
        if optimizer and optimizer.param_groups:
            return optimizer.param_groups[0].get('lr')
        return None

    def set_trainer(self, trainer):
        """Stores the trainer and potentially resolves the log directory."""
        super().set_trainer(trainer)
        # Resolve log directory if not provided, potentially using trainer info
        if self.log_dir_config is None and self.trainer and hasattr(self.trainer, 'checkpoint_manager'):
            # Prefer experiment-specific dir from checkpoint manager if available
            if self.trainer.checkpoint_manager and self.trainer.checkpoint_manager.log_dir:
                 self.resolved_log_dir = self.trainer.checkpoint_manager.log_dir
            else:
                # Fallback using experiment name (if available)
                experiment_name = getattr(self.trainer, 'experiment_name', 'default_experiment')
                base_output_dir = getattr(self.trainer.config, 'output_dir', 'outputs') # Get from config if possible
                self.resolved_log_dir = os.path.join(base_output_dir, experiment_name, 'logs')
        elif self.log_dir_config:
            # Use provided log_dir, make it absolute if needed
            self.resolved_log_dir = os.path.abspath(self.log_dir_config)
        else:
            # Fallback if log_dir not provided and trainer context unavailable
            self.resolved_log_dir = os.path.abspath(os.path.join('outputs', 'default_experiment', 'logs'))

        self.logger.info(f"TensorBoard logs will be saved to: {self.resolved_log_dir}")
        # Ensure the directory exists
        if self.resolved_log_dir:
             os.makedirs(self.resolved_log_dir, exist_ok=True)

        # >>> Get experiment_name directly from trainer attribute <<< 
        if hasattr(trainer, 'experiment_name') and trainer.experiment_name:
            self.experiment_name = trainer.experiment_name
            self.logger.info(f"Retrieved experiment_name from trainer: {self.experiment_name}")
        else:
            self.logger.warning("Could not find 'experiment_name' attribute on the trainer object. Log path construction might use defaults.")

    def set_log_dir_absolute(self, path: str):
        """Allows external setting of the absolute log directory, primarily for resuming."""
        self.logger.info(f"Externally setting TensorBoard log directory to: {path}")
        self.resolved_log_dir = path

    def _initialize_writer(self) -> None:
        """Initializes the SummaryWriter."""
        if self.writer:
            self.logger.debug("SummaryWriter already initialized.")
            return

        if not self.resolved_log_dir:
            self.logger.error("Cannot initialize SummaryWriter: Log directory has not been resolved.")
            # Attempt to resolve it now, maybe trainer is available?
            if self.trainer:
                self.set_trainer(self.trainer) # Try resolving again
                if not self.resolved_log_dir: # Check if it worked
                     return # Give up if still no directory
            else:
                return # Give up if no trainer context

        try:
            # Initialize the writer
            self.writer = SummaryWriter(log_dir=self.resolved_log_dir)
            self.logger.info(f"TensorBoard SummaryWriter initialized. Logging to: {self.resolved_log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SummaryWriter: {e}", exc_info=True)
            self.writer = None # Ensure writer is None if init fails

    def on_train_begin(self, **kwargs):
        """Initialize the SummaryWriter at the start of training."""
        self._initialize_writer()
        
        # Log hyperparameters if available (and writer initialized)
        if self.writer and self.trainer and hasattr(self.trainer, 'config') and self.trainer.config:
             # Assuming trainer.config is a Pydantic model or dict-like
            try:
                # Convert Pydantic model to dict if necessary
                if hasattr(self.trainer.config, 'model_dump'):
                     hparams_dict = self.trainer.config.model_dump()
                elif isinstance(self.trainer.config, dict):
                    hparams_dict = self.trainer.config
                else:
                    hparams_dict = vars(self.trainer.config)
                
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

    def on_step_begin(self, step: int, **kwargs):
        # No action needed on step begin for TensorBoard
        pass

    def on_step_end(self, step: int, global_step: int, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Log metrics after a training step."""
        if not self.writer:
            logger.debug("TensorBoard writer not initialized, skipping step logging.")
            return

        if metrics: # Only proceed if metrics are provided
            loss = metrics.get('loss', None)
            lr = metrics.get('lr', None)
            logger.debug(f"on_step_end received metrics: {metrics}")

            # Log loss if available
            if loss is not None:
                self.writer.add_scalar("Loss/train_step", loss, global_step=global_step)
                logger.debug(f"Logged Loss/train_step: {loss} at step {global_step}")

            # Log learning rate
            if lr is not None:
                logger.debug(f"Found lr in metrics dict: {lr}")
                self.writer.add_scalar("LearningRate/step", lr, global_step=global_step)
                logger.debug(f"Logged LearningRate/step: {lr} from metrics at step {global_step}")
            # Fallback for LR ONLY if lr was not in metrics
            elif self.trainer and self.trainer.optimizer:
                logger.debug("lr not found in metrics, attempting fallback to trainer.optimizer.")
                try:
                    # Assuming the first param group's LR is representative
                    current_lr = self.trainer.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("LearningRate/step", current_lr, global_step=global_step)
                    logger.debug(f"Logged LearningRate/step: {current_lr} from optimizer at step {global_step}")
                except (AttributeError, IndexError, KeyError) as e:
                    logger.warning(f"Could not retrieve learning rate from optimizer: {e}", exc_info=True)
            else:
                 logger.debug("lr not found in metrics and no trainer/optimizer for fallback.")

        else:
            logger.debug("No metrics provided for step {step}, global_step {global_step}. Nothing to log.")

    def on_epoch_begin(
        self, trainer: "Trainer", current_epoch: int, global_step: int, **kwargs: Any
    ) -> None:
        """Logs the start of an epoch."""
        self.logger.debug(f"TensorBoardLogger: Epoch {current_epoch + 1} begin at step {global_step}")
        # Optionally add epoch number to TB if desired
        # if self.writer:
        #     self.writer.add_scalar("Progress/Epoch", current_epoch + 1, global_step)

    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs):
        """Log metrics at the end of an epoch.

        Args:
            epoch: The epoch number (0-indexed).
            global_step: The global step count at the end of the epoch.
            metrics: Dictionary containing metrics from the epoch (e.g., {'loss': 0.5, 'val_loss': 0.6}).
                     Might include 'epoch_time_sec'.
            **kwargs: Catches any other keyword arguments.
        """
        if not self.writer or not metrics:
            return

        # Log scalars from the metrics dictionary
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Try to determine if it's train or val metric based on key naming
                tag_prefix = "Validation/" if "val_" in key or "eval_" in key else "Train/"
                # Clean up key name if prefixed
                clean_key = key.replace("val_", "").replace("eval_", "")
                full_tag = f"{tag_prefix}{clean_key.capitalize()}_epoch"

                try:
                    self.writer.add_scalar(full_tag, value, global_step=global_step)
                    self.logger.debug(f"Logged scalar to TensorBoard: {full_tag} = {value} (Step: {global_step})")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to log scalar '{full_tag}' to TensorBoard: {e}"
                    )

        # Log learning rate (if available and trainer exists)
        if self.trainer:
            lr = self._get_current_lr(self.trainer.optimizer)
            if lr is not None:
                try:
                    self.writer.add_scalar("LR/Learning_Rate_epoch", lr, global_step=global_step)
                    self.logger.debug(f"Logged LR to TensorBoard: {lr} (Step: {global_step})")
                except Exception as e:
                    self.logger.warning(f"Failed to log Learning Rate to TensorBoard: {e}")

        # Flush writer
        try:
            self.writer.flush()
            self.logger.debug("TensorBoard writer flushed after epoch end.")
        except Exception as e:
            self.logger.warning(f"Failed to flush TensorBoard writer: {e}")

    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Close the SummaryWriter at the end of training."""
        if self.writer:
            # Log hparams with final metrics if possible
            if hasattr(self, 'logged_hparams') and self.logged_hparams and metrics:
                 final_metrics = {
                     f'hparam/{k}': v for k, v in metrics.items()
                     if isinstance(v, (int, float))
                 }
                 # Filter hparams again to ensure they are scalars
                 scalar_hparams = {k: v for k, v in self.logged_hparams.items() if isinstance(v, (int, float, str, bool))}
                 try:
                     # Ensure final_metrics is not empty before logging
                     if final_metrics:
                         self.writer.add_hparams(scalar_hparams, final_metrics, global_step=kwargs.get('global_step', 0))
                     else:
                         self.logger.warning("No scalar final metrics found to log with hyperparameters.")
                 except Exception as e:
                     self.logger.error(f"Failed to log final hyperparameters: {e}")

            self.writer.close()
            self.logger.info("TensorBoard writer closed.")
            self.writer = None
            self.resolved_log_dir = None # Clear path

    # Implement other abstract methods as no-ops
    # Removed duplicate on_epoch_begin definition
    # Removed duplicate on_step_begin definition 