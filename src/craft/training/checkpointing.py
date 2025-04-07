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
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from ..data.tokenizers.base import Tokenizer
from dataclasses import dataclass, asdict # Import dataclasses
from ..training.callbacks.base import CallbackList # Ensure CallbackList is imported if used directly
from ..utils.logging import setup_logging
from hydra.utils import get_original_cwd
import time

# Define constants for checkpoint filenames and patterns
CHECKPOINT_FILE_PATTERN = r"checkpoint_step_(\d+)(?:_resumed)?\.pt"
RESUMED_SUFFIX = "_resumed.pt"
CHECKPOINT_CONFIG_FILENAME = "config.json" # Stores the training config
MODEL_CONFIG_FILENAME = "model_config.json" # Stores the model config
TOKENIZER_FILENAME = "tokenizer.json" # Or other relevant tokenizer files

# Define custom exception for checkpoint loading errors
class CheckpointLoadError(Exception):
    """Custom exception for checkpoint loading errors."""
    pass

class TrainingState(BaseModel):
    """Represents the state to be saved in a checkpoint."""
    epoch: int
    global_step: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    scaler_state_dict: Optional[Dict[str, Any]] = None
    best_val_metric: Optional[float] = None # Store the best validation metric achieved so far
    metrics: Optional[Dict[str, float]] = None # Last validation metrics or epoch metrics
    config: Optional[Dict[str, Any]] = None # Full experiment config
    tensorboard_log_dir: Optional[str] = None # Path to TB logs for linking
    callbacks_state: Optional[Dict[str, Any]] = None # State from callbacks

    # Allow extra fields if needed, though prefer explicit definition
    model_config = ConfigDict(extra='allow')

    # def __init__(self, **data: Any):
    #     super().__init__(**data)
    #     # Optional: Add logging here if needed, accessing attributes via self.* after super init
    #     # logger.error(f"[DEBUG TrainingState] Initializing with: step={self.global_step}, epoch={self.epoch}")

    # Example method to load state into components
    def load_state(self, model: torch.nn.Module, optimizer=None, scheduler=None, scaler=None, callbacks=None):
        """Utility method to load state dictionaries into PyTorch components."""
        model.load_state_dict(self.model_state_dict)
        if optimizer and self.optimizer_state_dict:
            optimizer.load_state_dict(self.optimizer_state_dict)
        if scheduler and self.scheduler_state_dict:
            scheduler.load_state_dict(self.scheduler_state_dict)
        if scaler and self.scaler_state_dict:
            try:
                # Safely check if the scaler has the load_state_dict method
                if hasattr(scaler, 'load_state_dict'):
                    scaler.load_state_dict(self.scaler_state_dict)
                else:
                    logger.warning("Scaler object does not have load_state_dict method.")
            except Exception as e:
                logger.error(f"Failed to load scaler state: {e}")

class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment_name: str,
        callbacks: Optional[List[Any]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Tokenizer] = None,
        keep_last_n: int = 3,
        keep_best_n: int = 1,
        config: Optional[Dict[str, Any]] = None, # Keep config dict for saving
        checkpoint_prefix: str = "checkpoint",
        max_checkpoints_to_keep: int = 5,
        save_best_only: bool = False,
        checkpoint_dir: Optional[str] = None, # <-- ADD explicit dir argument
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.callbacks = callbacks
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.tokenizer = tokenizer
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.config = config or {}
        self.checkpoint_prefix = checkpoint_prefix
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.save_best_only = save_best_only
        self.experiment_name = experiment_name
        self._MARKER_SUFFIX = "._SAVED"

        # Determine checkpoint directory
        if checkpoint_dir:
            # Use explicitly provided directory
            self.checkpoint_dir = Path(checkpoint_dir)
            self.logger.info(f"Using explicitly provided checkpoint directory: {self.checkpoint_dir}")
        else:
            # Try getting current Hydra run directory first
            try:
                from hydra.core.hydra_config import HydraConfig # Local import
                from hydra.utils import get_original_cwd # Keep for fallback
                hydra_run_dir = Path(HydraConfig.get().run.dir)
                # Construct path relative to Hydra's run dir
                self.checkpoint_dir = hydra_run_dir / "checkpoints"
                self.logger.info(f"Constructed checkpoint directory based on current Hydra run dir: {self.checkpoint_dir}")
            except Exception as hydra_e:
                 self.logger.warning(f"Could not get current Hydra run directory ({hydra_e}). Falling back to original CWD/relative path logic.")
                 # Fallback logic using get_original_cwd (as before)
                 try:
                     original_cwd = get_original_cwd()
                     self.checkpoint_dir = Path(original_cwd) / "outputs" / "experiments" / self.experiment_name / "checkpoints"
                     self.logger.info(f"Constructed checkpoint directory based on original CWD (fallback): {self.checkpoint_dir}")
                 except Exception as e:
                     self.logger.warning(f"Could not get original CWD via Hydra ({e}) as fallback. Using relative path.")
                     # Fallback to relative path from script location? Or just raise?
                     # Using a simpler relative path for now if all else fails.
                     self.checkpoint_dir = Path("checkpoints") # Simplest relative path
                     self.logger.warning(f"Using simple relative checkpoint directory (fallback): {self.checkpoint_dir.resolve()}")

        # Ensure the final checkpoint directory exists
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_e:
             self.logger.error(f"Failed to create checkpoint directory {self.checkpoint_dir}: {mkdir_e}")
             raise # Re-raise if directory creation fails critical path

        # Track saved checkpoints
        self.saved_checkpoints: List[Tuple[str, bool]] = [] # List of (path, is_best)
        self.best_checkpoints = []

        # Scan for existing checkpoints to initialize the internal list
        self._scan_for_existing_checkpoints()

    def _scan_for_existing_checkpoints(self):
        """Scans the checkpoint directory and populates self.saved_checkpoints."""
        self.logger.debug(f"Scanning {self.checkpoint_dir} for existing checkpoints...")
        found_checkpoints = []
        for ckpt_path in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            if ckpt_path.is_file():
                # Use the class-level pattern constant
                match = re.match(CHECKPOINT_FILE_PATTERN, ckpt_path.name)
                if match:
                    # Store the full path and assume it's not 'best' initially
                    # 'best.pt' is handled separately.
                    found_checkpoints.append((str(ckpt_path), False)) 
                    self.logger.debug(f"Found existing checkpoint: {ckpt_path.name}")
                else:
                    self.logger.debug(f"Ignoring file (does not match pattern): {ckpt_path.name}")

        # Sort them initially based on step number (ascending) for consistency,
        # though sorting happens again during management.
        # Access the path (first element of the tuple) for sorting.
        self.saved_checkpoints = sorted(
            found_checkpoints,
            key=lambda p: self._parse_checkpoint_name(os.path.basename(p[0])) or (0, 0) # Use basename of path p[0]
        )
        self.logger.info(f"Initialized with {len(self.saved_checkpoints)} existing checkpoints found in {self.checkpoint_dir}.")
        # Log basenames for readability
        self.logger.debug(f"Initial saved_checkpoints list (paths): {[os.path.basename(p[0]) for p in self.saved_checkpoints]}")

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
        is_best: bool = False,
        save_tokenizer: bool = True
    ) -> None:
        """Saves a checkpoint of the model and training state using a state dict.
           Optionally saves the tokenizer to a standard location within the run directory.
        """
        # Skip saving regular checkpoints if save_best_only is True and this is not the best
        if self.save_best_only and not is_best:
            self.logger.debug(f"Skipping saving non-best checkpoint {filename} due to save_best_only=True.")
            return
            
        self.logger.info(f"[CheckpointManager] Received save request for filename: {filename}")

        # --- Determine Save Path & Create Directory --- #
        # Ensure filename is just the name, not a full path initially
        filename = Path(filename).name
        # Construct the full path using the manager's checkpoint_dir
        save_path = self.checkpoint_dir / filename
        # Ensure the directory exists BEFORE trying to save
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # --- Save Tokenizer (if available and requested) --- #
        if self.tokenizer and save_tokenizer:
            # Save to a fixed location relative to the main run output dir
            # Assumes self.checkpoint_dir is like .../outputs/<run_name>/checkpoints
            tokenizer_save_dir = self.checkpoint_dir.parent / "tokenizer"
            try:
                # Save only if the directory doesn't already exist (or maybe always overwrite?)
                # Let's overwrite for simplicity, assuming tokenizer doesn't change.
                # If it could change, need a different strategy.
                tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save(str(tokenizer_save_dir))
                self.logger.info(f"Saved tokenizer to {tokenizer_save_dir}")
            except NotImplementedError:
                 self.logger.warning(f"Tokenizer type {type(self.tokenizer).__name__} does not support save(). Skipping tokenizer save.")
            except Exception as e:
                self.logger.error(f"Failed to save tokenizer to {tokenizer_save_dir}: {e}", exc_info=True)
                # Continue saving checkpoint even if tokenizer fails

        # --- Prepare State Dictionary --- #
        state_dict_to_save = state.model_dump()

        # --- Save the main state dictionary --- #
        try:
            self.logger.info(f"Saving checkpoint state for step {state.global_step} to: {save_path}")
            # Use logging level consistent with intent (DEBUG for internal steps)
            self.logger.debug(f"[DEBUG CheckpointManager] Saving state for step {state.global_step} to ABSOLUTE path {save_path.resolve()}")
            torch.save(state_dict_to_save, save_path)

            # --- Marker File --- #
            marker_path = Path(f"{save_path}._SAVED")
            try:
                # Use logging level consistent with intent (DEBUG for internal steps)
                self.logger.debug(f"[DEBUG CheckpointManager] Creating marker file at ABSOLUTE path {marker_path.resolve()}")
                marker_path.touch(exist_ok=True) # Create the marker file
                # Check existence immediately after touch(), log as DEBUG
                if marker_path.exists():
                     self.logger.debug(f"[DEBUG CheckpointManager] Marker file {marker_path.resolve()} EXISTS immediately after touch().")
                else:
                     self.logger.warning(f"[DEBUG CheckpointManager] Marker file {marker_path.resolve()} DOES NOT EXIST immediately after touch(). Potential filesystem delay? ")
            except Exception as e_marker:
                self.logger.error(f"Failed to create marker file {marker_path}: {e_marker}", exc_info=True)

            # Prune old checkpoints if enabled
            self._manage_checkpoints()

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {save_path}: {e}", exc_info=True)
            raise CheckpointLoadError(f"Failed to save checkpoint to {save_path}: {e}") from e

    def _parse_checkpoint_name(self, filename: str) -> Optional[Tuple[int, int]]:
        """Parses epoch and step from a checkpoint filename."""
        # Flexible regex: Handles epoch+step, step only, optional _time, optional _best, requires .pt
        # Example matches: 
        # - checkpoint_epoch_1_step_100.pt
        # - checkpoint_step_500.pt
        # - checkpoint_step_1000_time.pt
        # - checkpoint_step_2000_best.pt
        # - checkpoint_epoch_2_step_50_time_best.pt
        # - checkpoint_step_00002000.pt (From resume path)
        
        # Try epoch and step format first
        match_epoch_step = re.match(rf"{self.checkpoint_prefix}_epoch_(\d+)_step_(\d+).*\.pt", filename)
        if match_epoch_step:
            epoch = int(match_epoch_step.group(1))
            step = int(match_epoch_step.group(2))
            return epoch, step

        # Try step only format
        match_step_only = re.match(rf"{self.checkpoint_prefix}_step_(\d+).*\.pt", filename)
        if match_step_only:
             epoch = 0 # Assign epoch 0 if only step is present
             step = int(match_step_only.group(1))
             return epoch, step
        else:
            self.logger.debug(f"Could not parse epoch/step from filename: {filename}") # Debug level might be better
            return None

    def _add_saved_checkpoint(self, checkpoint_path: str, is_best: bool):
        """Adds a checkpoint path to the tracking list and manages cleanup."""
        self.saved_checkpoints.append((checkpoint_path, is_best))
        # Sort based on parsed epoch/step, then manage cleanup
        # Use os.path.basename for parsing consistency
        self.saved_checkpoints.sort(key=lambda p: self._parse_checkpoint_name(os.path.basename(p[0])) or (-1, -1))
        self._manage_checkpoints()

    def _manage_checkpoints(self):
        """Manages checkpoint retention based on epoch/step and max_checkpoints_to_keep."""
        self.logger.debug(f"[_manage_checkpoints] START: self.saved_checkpoints = {[os.path.basename(p[0]) for p in self.saved_checkpoints]}") # Log initial state

        if self.max_checkpoints_to_keep <= 0:
            return # Keep all checkpoints

        # Separate best checkpoints from regular ones for sorting
        best_checkpoints = [cp for cp in self.saved_checkpoints if cp[1]]
        regular_checkpoints = [cp for cp in self.saved_checkpoints if not cp[1]]
        self.logger.debug(f"[_manage_checkpoints] regular_checkpoints = {[os.path.basename(p[0]) for p in regular_checkpoints]}")

        # Sort regular checkpoints based on parsed epoch/step (most recent first)
        sorted_regular = sorted(
            regular_checkpoints, 
            key=lambda cp: self._parse_checkpoint_name(os.path.basename(cp[0])) or (-1, -1), # Handle None case
            reverse=True
        )
        self.logger.debug(f"[_manage_checkpoints] sorted_regular (descending): {[os.path.basename(p[0]) for p in sorted_regular]}")

        # Determine how many regular checkpoints to keep
        num_to_keep = self.max_checkpoints_to_keep # Use the value derived from config.keep_last
        self.logger.debug(f"[_manage_checkpoints] num_to_keep = {num_to_keep}")

        checkpoints_to_delete = sorted_regular[num_to_keep:]
        self.logger.debug(f"[_manage_checkpoints] checkpoints_to_delete = {[os.path.basename(p[0]) for p in checkpoints_to_delete]}")

        # Delete the oldest regular checkpoints
        deleted_paths = []
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
                    deleted_paths.append(path)
            except OSError as e:
                self.logger.error(f"Error removing checkpoint file {path}: {e}")
        self.logger.debug(f"[_manage_checkpoints] Actually deleted: {deleted_paths}")

        # Update the list of saved checkpoints (keep all 'best' and the retained regular ones)
        retained_regular = sorted_regular[:num_to_keep]
        self.saved_checkpoints = best_checkpoints + retained_regular
        self.logger.debug(f"[_manage_checkpoints] END: self.saved_checkpoints = {[os.path.basename(p[0]) for p in self.saved_checkpoints]}")

    def load_checkpoint(self, path: Optional[str] = None, map_location: Optional[str] = None) -> Optional[TrainingState]:
        """Loads the trainer state from a checkpoint file.
           Handles relative paths based on original CWD and absolute paths.
           Raises FileNotFoundError if the path doesn't exist.
           Raises CheckpointLoadError for other loading issues.
        """
        load_path_str = path # Keep original string if provided
        
        # --- Path Resolution Logic --- #
        if load_path_str is None:
             # Attempt to load latest from self.checkpoint_dir (relative to current Hydra run dir)
             self.logger.warning("No specific checkpoint path provided. Attempting to load latest from current run dir (might not be what you want when resuming!).")
             checkpoints = self._get_sorted_checkpoints()
             if not checkpoints:
                 self.logger.warning(f"No checkpoints found in {self.checkpoint_dir}.")
                 return None
             load_path = Path(checkpoints[-1][0]) # Use Path object of latest
             self.logger.info(f"Loading latest checkpoint found in current run dir: {load_path}")
        else:
            # Check if the provided path is absolute
            if os.path.isabs(load_path_str):
                load_path = Path(load_path_str)
                self.logger.info(f"Loading checkpoint from provided absolute path: {load_path}")
            else:
                # Resolve relative path against original working directory
                try:
                    original_cwd = Path(get_original_cwd())
                    load_path = original_cwd / load_path_str
                    self.logger.info(f"Resolved relative path {load_path_str} against original CWD ({original_cwd}) to: {load_path}")
                except Exception as e:
                    self.logger.error(f"Failed to get original CWD or resolve relative path: {e}. Falling back to current CWD.")
                    load_path = Path(load_path_str) # Fallback to CWD relative (likely wrong for resume)
        # --- End Path Resolution Logic --- #

        # Convert to Path object for internal use - load_path is now a Path object
        # load_path = Path(load_path_str) # Removed, load_path is already Path

        try:
            if not load_path.exists(): # Use Path.exists()
                self.logger.error(f"Checkpoint file not found: {load_path}")
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
                # Also check if the value is None, which is invalid if scheduler exists
                scheduler_state = checkpoint_dict['scheduler_state_dict']
                if scheduler_state is not None:
                    self.scheduler.load_state_dict(scheduler_state)
                else:
                    self.logger.error("Checkpoint contains 'scheduler_state_dict' but its value is None. Scheduler state MUST be loaded.")
                    raise CheckpointLoadError(f"Checkpoint {load_path} contains invalid None value for 'scheduler_state_dict'")
            elif self.scheduler:
                # Raise error if scheduler exists but state key is missing
                self.logger.error("Checkpoint does not contain 'scheduler_state_dict'. Scheduler state MUST be loaded.")
                raise CheckpointLoadError(f"Checkpoint {load_path} is missing required key for scheduler: 'scheduler_state_dict'")
            # else: Scheduler doesn't exist, so nothing to load

            # Scaler
            if self.scaler and 'scaler_state_dict' in checkpoint_dict:
                self.scaler.load_state_dict(checkpoint_dict['scaler_state_dict'])
            elif self.scaler:
                self.logger.warning("Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.")

            # Callbacks state
            self.logger.info(f"DEBUG LOAD: Checking callbacks state. self.callbacks is None: {self.callbacks is None}")
            if self.callbacks:
                self.logger.info(f"DEBUG LOAD: Type of self.callbacks: {type(self.callbacks)}")
                self.logger.info(f"DEBUG LOAD: callbacks_state in checkpoint_dict: {'callbacks_state' in checkpoint_dict}")
                if 'callbacks_state' in checkpoint_dict:
                    self.logger.info(f"DEBUG LOAD: checkpoint_dict['callbacks_state'] is None: {checkpoint_dict['callbacks_state'] is None}")
            
            if self.callbacks and "callbacks_state" in checkpoint_dict and checkpoint_dict["callbacks_state"] is not None:
                try:
                    self.logger.info("DEBUG LOAD: Attempting to call self.callbacks.load_state_dict...")
                    self.callbacks.load_state_dict(checkpoint_dict["callbacks_state"])
                    self.logger.info("Loaded callbacks state.")
                except Exception as e:
                    self.logger.error(f"Failed to load callbacks state: {e}")
            elif self.callbacks and "callbacks_state" not in checkpoint_dict:
                self.logger.warning("Callbacks state not found in checkpoint.")

            # Create and return TrainingState object from the loaded dictionary
            # Use model_validate for Pydantic models
            loaded_state_obj = TrainingState.model_validate(checkpoint_dict)

            self.logger.info(f"Successfully loaded checkpoint from {load_path} into TrainingState object.")

            # --- Callbacks Hook --- #
            self.logger.debug("Calling on_load_checkpoint for individual callbacks...")
            if self.callbacks and hasattr(self.callbacks, 'callbacks'):
                for cb in self.callbacks.callbacks:
                    if hasattr(cb, 'on_load_checkpoint') and callable(getattr(cb, 'on_load_checkpoint')):
                        self.logger.debug(f"  Calling on_load_checkpoint for {cb.__class__.__name__}")
                        try:
                            # Pass the state object and the filename
                            cb.on_load_checkpoint(state=loaded_state_obj, filename=load_path)
                        except Exception as cb_e:
                            self.logger.error(f"Error calling on_load_checkpoint for {cb.__class__.__name__}: {cb_e}", exc_info=True)
                            # Decide whether to continue or re-raise
            # --- End Callbacks Hook ---

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