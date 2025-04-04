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
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from pathlib import Path
from ..data.tokenizers.base import BaseTokenizer

class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[List[Any]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        keep_last_n: int = 3,
        keep_best_n: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.checkpoint_dir = os.getcwd()
        self.callbacks = callbacks
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.tokenizer = tokenizer
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n

        # Ensure the main checkpoint directory exists
        self.logger.info(f"CheckpointManager using directory: {self.checkpoint_dir}")

        # Track saved checkpoints
        self.saved_checkpoints = []
        self.best_checkpoints = []

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
        state: Dict[str, Any],
        filename: str,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> None:
        """Saves a checkpoint of the model and training state using a state dict.
           Handles tokenizer saving internally using self.tokenizer.
        """
        self.logger.info(f"[CheckpointManager] Received save request for filename: {filename}")

        # Ensure the save directory exists (using self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

        save_path = Path(self.checkpoint_dir) / filename
        self.logger.info(f"[CheckpointManager] Attempting to save checkpoint state to: {save_path}")

        try:
            # Save the provided state dictionary
            torch.save(state, save_path)
            self.logger.info(f"[CheckpointManager] Successfully saved checkpoint state to {save_path}")
            
            # --- Save Tokenizer (using self.tokenizer) --- #
            if self.tokenizer is not None:
                self.logger.info(f"[CheckpointManager] Tokenizer object type: {type(self.tokenizer)}")
                tokenizer_save_path = Path(self.checkpoint_dir) / "tokenizer" 
                self.logger.info(f"[CheckpointManager] Attempting to save tokenizer using save to: {tokenizer_save_path}")
                try:
                    # Ensure the directory exists
                    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
                    # Use the .save() method appropriate for our custom tokenizers
                    self.tokenizer.save(str(tokenizer_save_path))
                    self.logger.info(f"[CheckpointManager] Successfully saved tokenizer to {tokenizer_save_path}")
                except AttributeError as ae:
                     # Log if .save() is missing, though it should exist for BaseTokenizer subclasses
                     self.logger.error(f"[CheckpointManager] Tokenizer object {type(self.tokenizer)} lacks the 'save' method. Error: {ae}", exc_info=True)
                except Exception as e:
                    self.logger.error(f"[CheckpointManager] Failed to save tokenizer to {tokenizer_save_path}: {e}", exc_info=True)
            else:
                 self.logger.warning("[CheckpointManager] self.tokenizer is None, skipping tokenizer save.")
            # --- End Save Tokenizer --- #

            # Manage saved checkpoints (track based on filename pattern)
            self._add_saved_checkpoint(str(save_path))
            # --- Simplified best checkpoint tracking --- 
            if is_best:
                best_path = Path(self.checkpoint_dir) / "best.pt"
                shutil.copyfile(save_path, best_path)
                self.logger.info(f"Saved new best checkpoint link to {best_path} based on metrics: {metrics}")
                # Update internal list of best checkpoints
                self._add_best_checkpoint(str(best_path)) # Add the best.pt link

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {filename}: {e}", exc_info=True)
            # Return None or raise? For now, log and continue if possible
            return None 

    def _add_saved_checkpoint(self, checkpoint_path: str):
        """Adds a checkpoint path to the tracking list and manages cleanup."""
        # Simple tracking: add all non-best checkpoints to one list
        if "best.pt" not in checkpoint_path:
            self.saved_checkpoints.append(checkpoint_path)
            self.saved_checkpoints.sort(key=os.path.getmtime) # Keep sorted by time
            self._manage_checkpoints(self.saved_checkpoints, self.keep_last_n)

    def _add_best_checkpoint(self, checkpoint_path: str):
        """Adds a best checkpoint path and manages cleanup."""
        # Track best checkpoints separately
        self.best_checkpoints.append(checkpoint_path)
        self.best_checkpoints.sort(key=os.path.getmtime) # Keep sorted by time
        self._manage_checkpoints(self.best_checkpoints, self.keep_best_n)

    def _manage_checkpoints(self, checkpoint_list: List[str], keep_n: int):
        """Removes older checkpoints if the list exceeds keep_n."""
        if keep_n <= 0: # Keep all if keep_n is non-positive
            return
        
        while len(checkpoint_list) > keep_n:
            path_to_remove = checkpoint_list.pop(0) # Remove the oldest
            try:
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    self.logger.info(f"Removed old checkpoint: {path_to_remove}")
                # REMOVED Tokenizer deletion logic

            except OSError as e:
                self.logger.error(f"Error removing old checkpoint {path_to_remove}: {e}") 
    # --- End Restored Methods --- 

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