"""
Checkpointing Module
==================

This module handles saving and loading model checkpoints, including state management
and configuration serialization.
"""

import os
import torch
import logging
import shutil
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[Any] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.callbacks = callbacks
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device

        # Ensure the main checkpoint directory exists
        if self.checkpoint_dir:
            try:
                abs_checkpoint_dir = os.path.abspath(self.checkpoint_dir)
                self.logger.info(f"Attempting to create checkpoint directory: {abs_checkpoint_dir}")
                os.makedirs(abs_checkpoint_dir, exist_ok=True)
                # Verify immediately after attempt
                if os.path.exists(abs_checkpoint_dir):
                    self.logger.info(f"Successfully confirmed checkpoint directory exists: {abs_checkpoint_dir}")
                else:
                    self.logger.error(f"FAILED TO CONFIRM checkpoint directory exists after makedirs: {abs_checkpoint_dir}")
            except Exception as e:
                 self.logger.error(f"Exception during checkpoint directory creation {self.checkpoint_dir}: {e}", exc_info=True)
                 # Consider raising error or disabling checkpointing if dir creation fails

    def _get_tensorboard_log_dir(self) -> Optional[str]:
        """Helper to find the TensorBoardLogger and get its resolved log directory."""
        if not self.callbacks:
            return None
        
        tb_logger = None
        try:
            iterator = iter(self.callbacks)
        except TypeError:
            self.logger.warning("Callbacks object is not iterable, cannot find TensorBoardLogger.")
            return None
            
        for cb in iterator:
            if cb.__class__.__name__ == 'TensorBoardLogger': 
                tb_logger = cb
                break
        
        if tb_logger and hasattr(tb_logger, 'resolved_log_dir') and tb_logger.resolved_log_dir:
            return tb_logger.resolved_log_dir
        else:
            self.logger.warning("TensorBoardLogger not found or its log directory is not set.")
            return None

    def save_checkpoint(
        self,
        path: str,
        current_epoch: int,
        global_step: int,
        best_val_metric: float,
        metrics: Dict[str, list],
        is_best: bool = False
    ) -> None:
        """Saves a checkpoint of the model and training state."""
        # Directory existence is now ensured by __init__ and path construction
        # os.makedirs(os.path.dirname(path), exist_ok=True) # No longer needed here

        # Ensure config is serializable (convert OmegaConf if needed)
        serializable_config = self.config
        try:
            # Attempt to resolve OmegaConf to primitive types if it's OmegaConf
            if isinstance(self.config, DictConfig):
                serializable_config = OmegaConf.to_container(self.config, resolve=True)
            elif isinstance(self.config, BaseModel):
                serializable_config = self.config.model_dump()
        except ImportError:
            # If OmegaConf or Pydantic not installed or fails, proceed with original
            pass
        except Exception as e:
            self.logger.warning(f"Could not serialize config for checkpoint: {e}")
            # Fallback or store None?
            serializable_config = None # Avoid saving potentially problematic object

        # Get TensorBoard log directory
        tb_log_dir = self._get_tensorboard_log_dir()
        
        checkpoint = {
            'epoch': current_epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'metrics': metrics,
            'config': serializable_config,
            'tensorboard_log_dir': tb_log_dir
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        try:
            torch.save(checkpoint, path)
            self.logger.info(f"Checkpoint saved successfully to {path}")
            
            if is_best:
                # Construct absolute path for best_model.pt
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                # Simple copy for cross-platform compatibility
                if os.path.abspath(path) != os.path.abspath(best_path):
                    shutil.copyfile(path, best_path)
                    self.logger.info(f"Updated best model link to {best_path}")
                else:
                    # Log that we are saving directly as best_model.pt
                    self.logger.info(f"Saved best model directly as {best_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {path}: {e}", exc_info=True)

    def load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """Loads the trainer state from a checkpoint file (absolute path expected)."""
        # Path is now expected to be absolute
        if not os.path.exists(path):
            self.logger.error(f"Checkpoint file not found: {path}")
            return None

        loaded_state = {}
        try:
            # Load checkpoint onto the correct device directly
            # Set weights_only=False to allow loading non-tensor objects like OmegaConf config
            self.logger.warning("Loading checkpoint with weights_only=False. Ensure the checkpoint source is trusted.")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            elif self.scaler:
                self.logger.warning("Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.")

            # Extract state information to return
            loaded_state['epoch'] = checkpoint.get('epoch', 0)
            loaded_state['global_step'] = checkpoint.get('global_step', 0)
            loaded_state['best_val_metric'] = checkpoint.get('best_val_metric', float('inf'))
            loaded_state['metrics'] = checkpoint.get('metrics', {})
            loaded_state['config'] = checkpoint.get('config')
            loaded_state['tensorboard_log_dir'] = checkpoint.get('tensorboard_log_dir')
            
            self.logger.info(f"Successfully loaded checkpoint from {path}")
            return loaded_state

        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found during load attempt: {path}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {path}: {e}", exc_info=True)
            return None 

# --- Module-Level Helper Functions ---
# Moved from checkpoint_utils.py

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the latest checkpoint or None if no checkpoint is found
    """
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
    except OSError as e:
        logging.error(f"Error listing directory {checkpoint_dir}: {e}")
        return None

    if not checkpoint_files:
        logging.warning(f"No checkpoint files (.pt or .pth) found in {checkpoint_dir}")
        return None

    try:
        # Sort checkpoint files by modification time
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    except FileNotFoundError:
        # Handle race condition: file deleted between listdir and getmtime
        logging.warning(f"File not found during sorting in {checkpoint_dir}, possibly deleted. Retrying...")
        # Simple retry: could implement more robust logic
        return get_latest_checkpoint(checkpoint_dir) 
    except Exception as e:
        logging.error(f"Error sorting checkpoint files in {checkpoint_dir}: {e}")
        return None

    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    logging.info(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

def count_checkpoints(checkpoint_dir: str) -> int:
    """
    Count the number of checkpoint files (.pt or .pth) in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Number of checkpoint files
    """
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return 0

    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
        return len(checkpoint_files)
    except OSError as e:
        logging.error(f"Error accessing checkpoint directory {checkpoint_dir}: {e}")
        return 0

def clean_old_checkpoints(checkpoint_dir: str, keep: int = 5) -> None:
    """
    Remove old checkpoint files (.pt or .pth), keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        keep: Number of most recent checkpoints to keep (must be >= 0)
    """
    if keep < 0:
        logging.warning(f"Invalid value for 'keep' ({keep}). Must be non-negative. Not cleaning checkpoints.")
        return
        
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}. Cannot clean.")
        return

    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
    except OSError as e:
        logging.error(f"Error listing directory {checkpoint_dir} for cleaning: {e}")
        return

    if len(checkpoint_files) <= keep:
        logging.info(f"Found {len(checkpoint_files)} checkpoints in {checkpoint_dir}. No cleaning needed (keeping {keep}).")
        return

    try:
        # Sort checkpoint files by modification time
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    except FileNotFoundError:
        # Handle race condition: file deleted between listdir and getmtime
        logging.warning(f"File not found during sorting in {checkpoint_dir} for cleaning, possibly deleted. Skipping cleaning cycle.")
        return
    except Exception as e:
        logging.error(f"Error sorting checkpoint files in {checkpoint_dir} for cleaning: {e}")
        return

    # Remove old checkpoint files
    files_to_remove = checkpoint_files[keep:]
    logging.info(f"Cleaning {len(files_to_remove)} old checkpoints from {checkpoint_dir} (keeping {keep}).")
    for checkpoint_file in files_to_remove:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            os.remove(checkpoint_path)
            logging.debug(f"Removed old checkpoint: {checkpoint_path}") # Use debug level for successful removal
        except FileNotFoundError:
            logging.warning(f"Attempted to remove {checkpoint_path}, but it was already deleted.")
        except Exception as e:
            logging.error(f"Failed to remove checkpoint {checkpoint_path}: {str(e)}") 