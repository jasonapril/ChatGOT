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
import re # Import regex
import glob # Keep glob for potential use, though maybe not needed after refactor
from typing import Dict, Any, Optional, List, Tuple
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from pathlib import Path
from ..data.tokenizers.base import BaseTokenizer
from dataclasses import dataclass, asdict # Import dataclasses
from ..training.callbacks.base import CallbackList # Ensure CallbackList is imported if used directly

# Define custom exception for checkpoint loading errors
class CheckpointLoadError(Exception):
    """Custom exception for checkpoint loading errors."""
    pass

# Define TrainingState dataclass
@dataclass
class TrainingState:
    """Dataclass to hold the state relevant for checkpointing."""
    epoch: int
    global_step: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    scaler_state_dict: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    callbacks_state: Optional[Dict[str, Any]] = None
    tokenizer_path: Optional[str] = None
    best_val_metric: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    tensorboard_log_dir: Optional[str] = None

    # Add a class method to load from a dictionary, handling missing keys
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        return cls(
            model_state_dict=data.get('model_state_dict'), # Required, should raise error if missing later?
            optimizer_state_dict=data.get('optimizer_state_dict'), # Required
            epoch=data.get('epoch', 0),
            global_step=data.get('global_step', 0),
            scheduler_state_dict=data.get('scheduler_state_dict'),
            scaler_state_dict=data.get('scaler_state_dict'),
            config=data.get('config'),
            metrics=data.get('metrics', {}),
            best_val_metric=data.get('best_val_metric', float('inf')),
            tensorboard_log_dir=data.get('tensorboard_log_dir'),
            callbacks_state=data.get('callbacks_state'),
            tokenizer_path=data.get('tokenizer_path')
        )

class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[List[Any]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        keep_last_n: int = 3,
        keep_best_n: int = 1,
        config: Optional[Dict[str, Any]] = None, # Keep config dict for saving
        checkpoint_dir: Optional[str] = None,
        checkpoint_prefix: str = "checkpoint",
        max_checkpoints_to_keep: int = 5,
        save_best_only: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.checkpoint_dir = Path(checkpoint_dir or os.path.join(os.getcwd(), "checkpoints"))
        self.callbacks = callbacks
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.tokenizer = tokenizer
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.config = config or {} # Store config dict
        self.checkpoint_prefix = checkpoint_prefix
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.save_best_only = save_best_only

        # Ensure the main checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.info(f"CheckpointManager using directory: {self.checkpoint_dir}")

        # Track saved checkpoints
        self.saved_checkpoints: List[Tuple[str, bool]] = [] # List of (path, is_best)
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
        state: TrainingState, # Use TrainingState type hint
        filename: str, 
        metrics: Optional[Dict[str, float]] = None, # Metrics can still be passed separately for logging/best logic
        is_best: bool = False
    ) -> None:
        """Saves a checkpoint of the model and training state using a state dict.
           Handles tokenizer saving internally using self.tokenizer.
        """
        # Skip saving regular checkpoints if save_best_only is True and this is not the best
        if self.save_best_only and not is_best:
            self.logger.debug(f"Skipping saving non-best checkpoint {filename} due to save_best_only=True.")
            return
            
        self.logger.info(f"[CheckpointManager] Received save request for filename: {filename}")

        # Ensure the save directory exists (using self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")

        save_path = Path(self.checkpoint_dir) / filename
        self.logger.info(f"[CheckpointManager] Attempting to save checkpoint state to: {save_path}")

        try:
            # Convert TrainingState object to dictionary before saving
            state_dict_to_save = asdict(state) 
            # Filter out None values if desired, although torch.save handles them
            # state_dict_to_save = {k: v for k, v in asdict(state).items() if v is not None}
            torch.save(state_dict_to_save, save_path)
            self.logger.info(f"[CheckpointManager] Successfully saved checkpoint state to {save_path}")
            
            # --- Save Tokenizer (using self.tokenizer) --- #
            if self.tokenizer is not None:
                # Determine tokenizer save path based on global_step from the state
                tokenizer_save_dir_name = f"tokenizer_step_{state.global_step}" 
                tokenizer_save_path = Path(self.checkpoint_dir) / tokenizer_save_dir_name
                self.logger.info(f"[CheckpointManager] Tokenizer object type: {type(self.tokenizer)}")
                self.logger.info(f"[CheckpointManager] Attempting to save tokenizer to: {tokenizer_save_path}")
                try:
                    # Ensure the directory exists
                    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
                    # Use the .save() method appropriate for our custom tokenizers
                    self.tokenizer.save(str(tokenizer_save_path))
                    self.logger.info(f"[CheckpointManager] Successfully saved tokenizer to {tokenizer_save_path}")
                    # Add tokenizer path to the state dict *before* saving checkpoint, if possible?
                    # Or save it separately? Current logic saves checkpoint first.
                    # We need to ensure the *loaded* checkpoint *knows* about the tokenizer path.
                    # Let's modify the state *before* saving the main checkpoint file.
                    state_dict_to_save['tokenizer_path'] = tokenizer_save_dir_name
                    # Re-save the checkpoint with the tokenizer path included
                    torch.save(state_dict_to_save, save_path)
                    self.logger.info(f"[CheckpointManager] Re-saved checkpoint {save_path} to include tokenizer path: {tokenizer_save_dir_name}")

                except AttributeError as ae:
                     # Log if .save() is missing, though it should exist for BaseTokenizer subclasses
                     self.logger.error(f"[CheckpointManager] Tokenizer object {type(self.tokenizer)} lacks the 'save' method. Error: {ae}", exc_info=True)
                except Exception as e:
                    self.logger.error(f"[CheckpointManager] Failed to save tokenizer to {tokenizer_save_path}: {e}", exc_info=True)
            else:
                 self.logger.warning("[CheckpointManager] self.tokenizer is None, skipping tokenizer save.")
            # --- End Save Tokenizer --- #

            # Add to tracking list (only non-best, numbered checkpoints)
            # Also track if save_best_only=True, otherwise cleanup might remove best 
            if "best.pt" not in filename: # Check filename pattern
                 self._add_saved_checkpoint(str(save_path), is_best)
            
            # Handle best checkpoint link/copy
            if is_best:
                best_path = Path(self.checkpoint_dir) / "best.pt"
                # Use copy2 to preserve metadata if preferred, or copyfile for simplicity
                shutil.copyfile(save_path, best_path) 
                self.logger.info(f"Saved new best checkpoint link to {best_path} based on metrics: {metrics}")
                # No need to track best.pt in a separate list for cleanup

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {filename}: {e}", exc_info=True)
            # Return None or raise? For now, log and continue if possible
            return None 

    def _parse_checkpoint_name(self, filename: str) -> Optional[Tuple[int, int]]:
        """Parses epoch and step from a checkpoint filename.
           Assumes pattern like 'epoch=E-step=S.pt' or similar prefixes.
        """
        match = re.search(r"epoch=(\d+).*step=(\d+).*.pt", Path(filename).name)
        if match:
            try:
                epoch = int(match.group(1))
                step = int(match.group(2))
                return epoch, step
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse epoch/step from filename: {filename}")
        return None

    def _add_saved_checkpoint(self, checkpoint_path: str, is_best: bool):
        """Adds a checkpoint path to the tracking list and manages cleanup."""
        self.saved_checkpoints.append((checkpoint_path, is_best))
        # Sort based on parsed epoch/step, then manage cleanup
        self.saved_checkpoints.sort(key=lambda p: self._parse_checkpoint_name(p[0]) or (-1, -1))
        self._manage_checkpoints()

    def _manage_checkpoints(self):
        """Manages checkpoint retention based on epoch/step and max_checkpoints_to_keep."""
        if self.max_checkpoints_to_keep <= 0:
            return # Keep all checkpoints

        # Separate best checkpoints from regular ones for sorting
        best_checkpoints = [cp for cp in self.saved_checkpoints if cp[1]]
        regular_checkpoints = [cp for cp in self.saved_checkpoints if not cp[1]]

        # Sort regular checkpoints based on parsed epoch/step (most recent first)
        sorted_regular = sorted(
            regular_checkpoints, 
            key=lambda cp: self._parse_checkpoint_name(os.path.basename(cp[0])) or (-1, -1), # Handle None case
            reverse=True
        )

        # Determine how many regular checkpoints to keep
        num_to_keep = self.max_checkpoints_to_keep
        checkpoints_to_delete = sorted_regular[num_to_keep:]

        # Delete the oldest regular checkpoints
        for path, _ in checkpoints_to_delete:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    self.logger.info(f"Removed old checkpoint: {path}")
                    # Also remove associated tokenizer dir if it exists
                    parsed_name = self._parse_checkpoint_name(os.path.basename(path))
                    if parsed_name:
                        tokenizer_dir_name = f"tokenizer_step_{parsed_name[1]}"
                        tokenizer_dir_path = os.path.join(os.path.dirname(path), tokenizer_dir_name)
                        if os.path.isdir(tokenizer_dir_path):
                            import shutil # Use shutil for directory removal
                            shutil.rmtree(tokenizer_dir_path)
                            self.logger.info(f"Removed associated tokenizer dir: {tokenizer_dir_path}")
            except OSError as e:
                self.logger.error(f"Error removing checkpoint file {path}: {e}")

        # Update the list of saved checkpoints (keep all 'best' and the retained regular ones)
        self.saved_checkpoints = best_checkpoints + sorted_regular[:num_to_keep]

    def load_checkpoint(self, path: Optional[str] = None, map_location: Optional[str] = None) -> Optional[TrainingState]:
        """Loads the trainer state from a checkpoint file (absolute path expected).
           Raises FileNotFoundError if the path doesn't exist.
           Raises CheckpointLoadError for other loading issues.
        """
        load_path_str = path # Keep original string if provided
        # Use lower() for case-insensitive comparison for "latest"
        if load_path_str is None or load_path_str.lower() == "latest":
            # Find the latest checkpoint based on parsed name
            checkpoints = self._get_sorted_checkpoints()
            if not checkpoints:
                self.logger.warning("No checkpoints found to load.")
                return None # Explicitly return None if no checkpoints are found
            load_path_str = checkpoints[-1][0] # Path (string) of the latest checkpoint

        # Convert to Path object for internal use
        load_path = Path(load_path_str)

        try:
            if not load_path.exists(): # Use Path.exists()
                self.logger.error(f"Checkpoint file not found: {load_path}")
                # Raise FileNotFoundError instead of returning None
                raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

            self.logger.info(f"Loading checkpoint from: {load_path}")
            # Pass the Path object directly to torch.load
            checkpoint_dict = torch.load(load_path, map_location=map_location or self.device, weights_only=False)

            # Load state dicts into respective objects
            # Model
            if 'model_state_dict' in checkpoint_dict:
                # Handle potential DataParallel/DDP wrapping
                state_dict = checkpoint_dict['model_state_dict']
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
                raise CheckpointLoadError(f"Checkpoint {load_path} is missing required key: 'model_state_dict'")

            # Optimizer
            if 'optimizer_state_dict' in checkpoint_dict:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            else:
                self.logger.warning("Checkpoint does not contain 'optimizer_state_dict'. Optimizer state not loaded.")
                # Decide if this should be fatal? Maybe not if only evaluating.
                # raise CheckpointLoadError(f"Checkpoint {load_path} is missing required key: 'optimizer_state_dict'")

            # Scheduler
            if self.scheduler and 'scheduler_state_dict' in checkpoint_dict:
                self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            elif self.scheduler:
                self.logger.warning("Checkpoint does not contain 'scheduler_state_dict'. Scheduler state not loaded.")

            # Scaler
            if self.scaler and 'scaler_state_dict' in checkpoint_dict:
                self.scaler.load_state_dict(checkpoint_dict['scaler_state_dict'])
            elif self.scaler:
                self.logger.warning("Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.")

            # --- Load Tokenizer --- #
            if self.tokenizer is not None:
                tokenizer_path_in_ckpt = checkpoint_dict.get('tokenizer_path')
                if tokenizer_path_in_ckpt:
                    # Construct absolute path from checkpoint dir and relative path in checkpoint
                    tokenizer_abs_path = load_path.parent / tokenizer_path_in_ckpt
                    self.logger.info(f"Attempting to load tokenizer from path specified in checkpoint: {tokenizer_abs_path}")
                    
                    if tokenizer_abs_path.exists() and tokenizer_abs_path.is_dir():
                        try:
                            # Assuming tokenizer has a .load() method accepting the directory path
                            self.tokenizer.load(str(tokenizer_abs_path))
                            self.logger.info(f"Successfully loaded tokenizer state from {tokenizer_abs_path}")
                        except AttributeError as ae:
                            self.logger.warning(f"Tokenizer object {type(self.tokenizer)} lacks the required 'load' method. Tokenizer state not loaded. Error: {ae}", exc_info=True)
                        except Exception as e:
                            self.logger.warning(f"Failed to load tokenizer state from {tokenizer_abs_path}: {e}. Continuing checkpoint load.", exc_info=True)
                    else:
                        self.logger.warning(f"Tokenizer directory '{tokenizer_abs_path}' specified in checkpoint not found. Skipping tokenizer load.")
                else:
                    self.logger.warning("Checkpoint does not contain 'tokenizer_path'. Tokenizer state not loaded.")
            else:
                self.logger.info("No tokenizer instance in CheckpointManager, skipping tokenizer load.")
            # --- End Load Tokenizer --- #

            # Callbacks state
            if self.callbacks and "callbacks_state" in checkpoint_dict and checkpoint_dict["callbacks_state"] is not None:
                try:
                    self.callbacks.load_state_dict(checkpoint_dict["callbacks_state"])
                    self.logger.info("Loaded callbacks state.")
                except Exception as e:
                    self.logger.error(f"Failed to load callbacks state: {e}")
            elif self.callbacks and "callbacks_state" not in checkpoint_dict:
                self.logger.warning("Callbacks state not found in checkpoint.")

            # Create and return TrainingState object from the loaded dictionary
            loaded_state_obj = TrainingState.from_dict(checkpoint_dict)

            self.logger.info(f"Successfully loaded checkpoint from {load_path} into TrainingState object.")

            # --- Callbacks Hook --- #
            if self.callbacks:
                # Pass the fully constructed TrainingState object to the callback
                self.callbacks.on_load_checkpoint(trainer_state=loaded_state_obj)

            return loaded_state_obj

        except FileNotFoundError: # Specific handling for file not found
            self.logger.error(f"Checkpoint file not found during load attempt: {load_path}")
            raise # Re-raise the FileNotFoundError
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {load_path}: {e}", exc_info=True)
            # Raise CheckpointLoadError for other errors
            raise CheckpointLoadError(f"Failed to load checkpoint from {load_path}: {e}") from e

    def _get_sorted_checkpoints(self) -> List[Tuple[str, int, int]]:
        """Gets checkpoint files from the directory, sorted by epoch and step."""
        # Use Path.glob for better reliability
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pt"))

        parsed_checkpoints = []
        for f_path in checkpoint_files: # f_path is a Path object
            f = str(f_path) # Convert to string for storage/return
            parsed = self._parse_checkpoint_name(f_path.name) # Parse the name part
            if parsed: # Only include successfully parsed names
                parsed_checkpoints.append((f, parsed[0], parsed[1])) # (path_str, epoch, step)

        # Sort by epoch, then step (ascending)
        parsed_checkpoints.sort(key=lambda x: (x[1], x[2]))
        return parsed_checkpoints

    def _parse_checkpoint_name(self, filename: str) -> Optional[Tuple[int, int]]:
        """Parses epoch and step from a checkpoint filename."""
        # Matches patterns like: checkpoint_epoch_1_step_100.pt, checkpoint_step_500_best.pt, checkpoint_step_1000.pt
        # Allows optional '_best'
        match_epoch_step = re.match(rf"{self.checkpoint_prefix}_epoch_(\d+)_step_(\d+)(?:_best)?\.pt", filename)
        match_step_only = re.match(rf"{self.checkpoint_prefix}_step_(\d+)(?:_best)?\.pt", filename)
        
        if match_epoch_step:
            epoch = int(match_epoch_step.group(1))
            step = int(match_epoch_step.group(2))
            return epoch, step
        elif match_step_only:
             epoch = 0 # Assign epoch 0 if only step is present
             step = int(match_step_only.group(1))
             return epoch, step
        else:
            self.logger.debug(f"Could not parse epoch/step from filename: {filename}") # Debug level might be better
            return None