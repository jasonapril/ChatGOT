import logging
import os
import datetime
from typing import Dict, Any, Optional, Type
from hydra.core.hydra_config import HydraConfig
import torch
from pathlib import Path

# Conditional import of SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

from .base import Callback, TYPE_CHECKING

# Import Trainer within TYPE_CHECKING
if TYPE_CHECKING:
    from ..trainer import Trainer

logger = logging.getLogger(__name__)

class TensorBoardLogger(Callback):
    """
    Callback that logs metrics to TensorBoard.

    Logs loss, learning rate, and other specified metrics during training.
    """
    def __init__(self, log_dir: Optional[str] = None, comment: str = "", flush_secs: int = 120) -> None:
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
        self.writer: Optional[SummaryWriter] = None # type: ignore[assignment]
        self.log_dir_config = log_dir
        self.resolved_log_dir: Optional[str] = None
        self.experiment_name: Optional[str] = None
        self.logged_hparams: Optional[Dict[str, Any]] = None

        if not _TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard not found. Install tensorboard (e.g., pip install tensorboard) to use TensorBoardLogger.")

    def _get_current_lr(self, optimizer: Optional[torch.optim.Optimizer]) -> Optional[float]:
        """Safely get the current learning rate from the optimizer."""
        if optimizer and optimizer.param_groups:
            lr = optimizer.param_groups[0].get('lr')
            return float(lr) if isinstance(lr, (int, float)) else None
        return None

    def set_trainer(self, trainer: 'Trainer') -> None:
        """Stores the trainer and potentially resolves the log directory."""
        super().set_trainer(trainer)
        log_dir_path: Optional[str] = None
        if self.log_dir_config is None and self.trainer and hasattr(self.trainer, 'checkpoint_manager') and self.trainer.checkpoint_manager:
            ckpt_dir = getattr(self.trainer.checkpoint_manager, 'checkpoint_dir', None)
            if ckpt_dir:
                 log_dir_path = str(Path(ckpt_dir).parent / "tensorboard_logs")
            else:
                experiment_name = getattr(self.trainer, 'experiment_name', 'default_experiment')
                base_output_dir = getattr(self.trainer.config, 'output_dir', 'outputs')
                log_dir_path = os.path.join(base_output_dir, experiment_name, 'tensorboard_logs')
        elif self.log_dir_config:
            log_dir_path = os.path.abspath(self.log_dir_config)
        else:
            log_dir_path = os.path.abspath(os.path.join('outputs', 'default_experiment', 'tensorboard_logs'))

        self.resolved_log_dir = log_dir_path

        self.logger.info(f"TensorBoard logs will be saved to: {self.resolved_log_dir}")
        if self.resolved_log_dir:
             os.makedirs(self.resolved_log_dir, exist_ok=True)

        exp_name = getattr(trainer, 'experiment_name', None)
        if exp_name:
            self.experiment_name = str(exp_name)
            self.logger.info(f"Retrieved experiment_name from trainer: {self.experiment_name}")
        else:
            self.logger.warning("Could not find 'experiment_name' attribute on the trainer object. Log path construction might use defaults.")

    def set_log_dir_absolute(self, path: str) -> None:
        """Allows external setting of the absolute log directory, primarily for resuming."""
        self.logger.info(f"Externally setting TensorBoard log directory to: {path}")
        self.resolved_log_dir = path

    def _initialize_writer(self) -> None:
        """Initializes the SummaryWriter."""
        if not _TENSORBOARD_AVAILABLE:
            return
            
        if self.writer:
            self.logger.debug("SummaryWriter already initialized.")
            return

        if not self.resolved_log_dir:
            self.logger.error("Cannot initialize SummaryWriter: Log directory has not been resolved.")
            if self.trainer:
                self.set_trainer(self.trainer)
                if not self.resolved_log_dir:
                     return
            else:
                return

        try:
            self.writer = SummaryWriter(log_dir=self.resolved_log_dir)
            self.logger.info(f"TensorBoard SummaryWriter initialized. Logging to: {self.resolved_log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SummaryWriter: {e}", exc_info=True)
            self.writer = None

    def on_train_begin(self, **kwargs: Any) -> None:
        """Initialize the SummaryWriter at the start of training."""
        self._initialize_writer()
        
        if self.writer and self.trainer and hasattr(self.trainer, 'config') and self.trainer.config:
            try:
                if hasattr(self.trainer.config, 'model_dump'):
                     hparams_dict = self.trainer.config.model_dump()
                elif isinstance(self.trainer.config, dict):
                    hparams_dict = self.trainer.config
                else:
                    hparams_dict = vars(self.trainer.config)
                
                scalar_hparams = {
                    k: v for k, v in hparams_dict.items()
                    if isinstance(v, (int, float, str, bool))
                }
                
                self.logger.info(f"Hyperparameters logged (to be fully associated later): {scalar_hparams}")
                self.logged_hparams = scalar_hparams

            except Exception as e:
                self.logger.error(f"Could not log hyperparameters: {e}")

    def on_step_begin(self, step: int, **kwargs: Any) -> None:
        pass

    def on_step_end(self, step: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Log metrics after a training step."""
        if not self.writer:
            logger.debug("TensorBoard writer not initialized, skipping step logging.")
            return

        if metrics:
            loss = metrics.get('loss', None)
            lr = metrics.get('lr', None)
            logger.debug(f"on_step_end received metrics: {metrics}")

            if loss is not None:
                self.writer.add_scalar("Loss/train_step", loss, global_step=global_step)
                logger.debug(f"Logged Loss/train_step: {loss} at step {global_step}")

            if lr is not None:
                logger.debug(f"Found lr in metrics dict: {lr}")
                self.writer.add_scalar("LearningRate/step", lr, global_step=global_step)
                logger.debug(f"Logged LearningRate/step: {lr} from metrics at step {global_step}")
            elif self.trainer and self.trainer.optimizer:
                logger.debug("lr not found in metrics, attempting fallback to trainer.optimizer.")
                try:
                    current_lr = self.trainer.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("LearningRate/step", current_lr, global_step=global_step)
                    logger.debug(f"Logged LearningRate/step: {current_lr} from optimizer at step {global_step}")
                except (AttributeError, IndexError, KeyError) as e:
                    logger.warning(f"Could not retrieve learning rate from optimizer: {e}", exc_info=True)
            else:
                 logger.debug("lr not found in metrics and no trainer/optimizer for fallback.")

        else:
            logger.debug("No metrics provided for step {step}, global_step {global_step}. Nothing to log.")

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Logs the start of an epoch."""
        global_step = kwargs.get('global_step', getattr(self.trainer.state, 'global_step', -1) if self.trainer and hasattr(self.trainer, 'state') else -1)
        self.logger.debug(f"TensorBoardLogger: Epoch {epoch + 1} begin at step {global_step}")

    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
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

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tag_prefix = "Validation/" if "val_" in key or "eval_" in key else "Train/"
                clean_key = key.replace("val_", "").replace("eval_", "")
                full_tag = f"{tag_prefix}{clean_key.capitalize()}_epoch"

                try:
                    self.writer.add_scalar(full_tag, value, global_step=global_step)
                    self.logger.debug(f"Logged scalar to TensorBoard: {full_tag} = {value} (Step: {global_step})")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to log scalar '{full_tag}' to TensorBoard: {e}"
                    )

        if self.trainer:
            lr = self._get_current_lr(self.trainer.optimizer)
            if lr is not None:
                try:
                    self.writer.add_scalar("LR/Learning_Rate_epoch", lr, global_step=global_step)
                    self.logger.debug(f"Logged LR to TensorBoard: {lr} (Step: {global_step})")
                except Exception as e:
                    self.logger.warning(f"Failed to log Learning Rate to TensorBoard: {e}")

        try:
            self.writer.flush()
            self.logger.debug("TensorBoard writer flushed after epoch end.")
        except Exception as e:
            self.logger.warning(f"Failed to flush TensorBoard writer: {e}")

    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Close the SummaryWriter at the end of training."""
        if self.writer:
            if hasattr(self, 'logged_hparams') and self.logged_hparams and metrics:
                scalar_metrics = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                try:
                    self.writer.add_hparams(self.logged_hparams, scalar_metrics)
                    self.logger.info(f"Logged hyperparameters with final metrics: {scalar_metrics}")
                except Exception as e:
                    self.logger.error(f"Failed to log hparams with final metrics: {e}")
            
            try:
                self.writer.close()
                self.logger.info("TensorBoard writer closed.")
            except Exception as e:
                self.logger.warning(f"Failed to close TensorBoard writer: {e}")
            self.writer = None
            self.resolved_log_dir = None

    # Implement other abstract methods as no-ops
    # Removed duplicate on_epoch_begin definition
    # Removed duplicate on_step_begin definition 