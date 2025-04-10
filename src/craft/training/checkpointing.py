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
from typing import Dict, Any, Optional, List, Tuple, cast, Union
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError
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

class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str, # Now required
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        experiment_name: str, # Still useful for context/logging
        callbacks: Optional[List[Any]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Tokenizer] = None,
        keep_last_n: int = 3,
        keep_best_n: int = 1,
        config: Optional[Dict[str, Any]] = None, # Keep config dict for saving in state
        checkpoint_prefix: str = "checkpoint", # Keep prefix
        # max_checkpoints_to_keep: int = 5, # Superseded by keep_last_n/keep_best_n logic?
        save_best_only: bool = False,
        load_strict: bool = True, # Added argument for model state loading strictness
        # Removed checkpoint_dir Optional argument
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.callbacks = callbacks # Store for potential state saving/loading
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.tokenizer = tokenizer
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.config = config or {}
        self.checkpoint_prefix = checkpoint_prefix
        # self.max_checkpoints_to_keep = max_checkpoints_to_keep # Remove if covered by keep_last_n
        self.save_best_only = save_best_only
        self.experiment_name = experiment_name
        self._MARKER_SUFFIX = "._SAVED"
        self.load_strict = load_strict

        # --- Use provided checkpoint_dir directly --- #
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger.info(f"CheckpointManager initialized with directory: {self.checkpoint_dir}")
        # --- Remove internal directory discovery logic --- #
        # if checkpoint_dir: ...
        # else: ... (HydraConfig/get_original_cwd logic removed)

        # Ensure the final checkpoint directory exists
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_e:
             self.logger.error(f"Failed to create checkpoint directory {self.checkpoint_dir}: {mkdir_e}")
             raise # Re-raise if directory creation fails critical path

        # Track saved checkpoints
        self.saved_checkpoints: List[Tuple[str, bool]] = [] # List of (path, is_best)
        self.best_checkpoints: List[str] = [] # Store paths of best checkpoints

        # Scan for existing checkpoints
        self._scan_for_existing_checkpoints()

    def _scan_for_existing_checkpoints(self) -> None:
        """Scans the checkpoint directory and populates self.saved_checkpoints."""
        self.logger.debug(f"Scanning {self.checkpoint_dir} for existing checkpoints...")
        found_checkpoints = []
        # Reset the list before scanning to pick up new files during re-scans
        self.saved_checkpoints = [] 
        for ckpt_path in self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pt"): # Use prefix and wildcard
            self.logger.debug(f"[Scan Loop] Checking path: {ckpt_path}")
            if ckpt_path.is_file():
                self.logger.debug(f"[Scan Loop] Path is file: {ckpt_path.name}")
                # Use the class-level pattern constant
                # Simpler check: Just ensure it's a .pt file with the prefix for scanning
                if ckpt_path.name.startswith(self.checkpoint_prefix) and ckpt_path.name.endswith(".pt"): 
                    # Store the full path and assume it's not 'best' initially
                    # 'best.pt' is handled separately.
                    found_checkpoints.append((str(ckpt_path), False)) 
                    self.logger.debug(f"Found existing checkpoint during scan: {ckpt_path.name}")
                else:
                    self.logger.debug(f"Ignoring file (does not match pattern): {ckpt_path.name}")

        # Sort them initially based on step number (ascending) for consistency,
        # though sorting happens again during management.
        # Access the path (first element of the tuple) for sorting.
        found_checkpoints.sort(
            key=lambda p: self._parse_checkpoint_name(os.path.basename(p[0])) or (0, 0) # Use basename of path p[0]
        )
        # Assign the newly scanned and sorted list
        self.saved_checkpoints = found_checkpoints
        self.logger.info(
            f"Scan complete. Found {len(found_checkpoints)} matching checkpoints in {self.checkpoint_dir}."
        )
        # Correctly log the full paths found
        paths_to_log = [str(p[0]) for p in found_checkpoints]
        self.logger.debug(f"Initial saved_checkpoints list (paths): {paths_to_log}")

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
            # Explicitly cast the return value to help mypy
            return cast(Optional[str], tb_logger.resolved_log_dir)
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
        # Remove tokenizer_path as it's not used for loading anymore
        state_dict_to_save.pop('tokenizer_path', None)

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

            # --- Handle best.pt symlink --- #
            if is_best:
                best_link_path = self.checkpoint_dir / "best.pt"
                try:
                    # Remove existing link if it exists
                    if best_link_path.is_symlink() or best_link_path.exists(): # Check exists too, might be copied file
                        best_link_path.unlink()
                        self.logger.debug(f"Removed existing best.pt link/file.")
                    # Create new symlink pointing to the current best save path
                    # Use relative path for symlink if possible, depends on OS support
                    # os.symlink(save_path.name, best_link_path) # Use relative name
                    # Safer cross-platform approach: copy the file
                    shutil.copyfile(save_path, best_link_path)
                    self.logger.info(f"Updated best.pt to point to {save_path.name} (by copying)")
                except Exception as e_link:
                    self.logger.error(f"Failed to update best.pt link/file: {e_link}", exc_info=True)

            # Add the saved checkpoint to tracking list BEFORE managing/pruning
            self._add_saved_checkpoint(str(save_path), is_best)

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

    def _add_saved_checkpoint(self, checkpoint_path: str, is_best: bool) -> None:
        """Adds a checkpoint path to the tracking list and ensures uniqueness."""
        # Avoid adding duplicates
        if not any(p[0] == checkpoint_path for p in self.saved_checkpoints):
            self.saved_checkpoints.append((checkpoint_path, is_best))
            self.logger.debug(f"Added checkpoint {os.path.basename(checkpoint_path)} (is_best={is_best}) to tracking list.")
        else:
             self.logger.debug(f"Skipping duplicate add for {os.path.basename(checkpoint_path)}.")

    def _manage_checkpoints(self) -> None:
        """Removes old checkpoints based on keep_last_n and keep_best_n settings."""
        self.logger.debug("Starting checkpoint management...")
        
        # Separate best and non-best checkpoints
        best_checkpoints = sorted([p for p in self.saved_checkpoints if p[1]], key=lambda x: self._parse_checkpoint_name(os.path.basename(x[0])) or (0,0), reverse=True)
        last_checkpoints = sorted([p for p in self.saved_checkpoints if not p[1]], key=lambda x: self._parse_checkpoint_name(os.path.basename(x[0])) or (0,0), reverse=True)

        # Use self.keep_last_n instead of max_checkpoints_to_keep
        num_to_keep_last = self.keep_last_n
        num_to_keep_best = self.keep_best_n

        self.logger.debug(f"Keep last {num_to_keep_last} checkpoints.")
        self.logger.debug(f"Keep best {num_to_keep_best} checkpoints.")

        # Checkpoints to keep
        checkpoints_to_keep = set()
        if num_to_keep_best > 0 and best_checkpoints:
            checkpoints_to_keep.update([p[0] for p in best_checkpoints[:num_to_keep_best]])
        if num_to_keep_last > 0 and last_checkpoints:
            checkpoints_to_keep.update([p[0] for p in last_checkpoints[:num_to_keep_last]])
        
        self.logger.debug(f"Total unique checkpoints to keep: {len(checkpoints_to_keep)}")
        self.logger.debug(f"Checkpoints to keep (paths): {[os.path.basename(p) for p in checkpoints_to_keep]}")

        # Checkpoints to remove
        checkpoints_to_remove = []
        for ckpt_path, is_best_flag in self.saved_checkpoints:
            if ckpt_path not in checkpoints_to_keep:
                checkpoints_to_remove.append(ckpt_path)

        self.logger.debug(f"Checkpoints to remove: {[os.path.basename(p) for p in checkpoints_to_remove]}")

        # Remove the files
        removed_count = 0
        for ckpt_path in checkpoints_to_remove:
            try:
                # Remove the .pt file
                p = Path(ckpt_path)
                if p.exists():
                    p.unlink()
                    self.logger.info(f"Removed old checkpoint: {p.name}")
                    removed_count += 1
                else:
                    self.logger.debug(f"Tried to remove {p.name}, but it no longer exists.")
                
                # Remove the corresponding marker file
                marker_path = Path(f"{ckpt_path}._SAVED")
                if marker_path.exists():
                    marker_path.unlink()
                    self.logger.debug(f"Removed corresponding marker file: {marker_path.name}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {ckpt_path}: {e}")

        # Update the internal list
        self.saved_checkpoints = [p for p in self.saved_checkpoints if p[0] in checkpoints_to_keep]
        self.logger.debug(f"Removed {removed_count} old checkpoints. Current tracked: {len(self.saved_checkpoints)}")

    def load_checkpoint(
        self,
        path_specifier: str,
        # Remove component arguments: model, optimizer, scheduler, scaler, callbacks, device
    ) -> Optional[TrainingState]:
        """Loads a TrainingState object from a specified checkpoint file.

        Finds the checkpoint file based on path_specifier ('latest', 'best', or path),
        loads the dictionary, and returns a TrainingState object.
        Does NOT apply the state to components.

        Args:
            path_specifier: 'latest', 'best', a specific checkpoint filename,
                             or a full path to a checkpoint file.

        Returns:
            The loaded TrainingState object, or None if loading fails or no suitable
            checkpoint is found.
        """
        # Rescan *before* resolving keywords or relative paths based on current disk state
        if path_specifier.lower() in ['latest', 'best']:
            self.logger.debug("Rescanning directory for 'latest'/'best' specifier...")
            self._scan_for_existing_checkpoints()

        checkpoint_path_str = self._find_checkpoint_path(path_specifier)

        if not checkpoint_path_str:
            self.logger.warning(f"No checkpoint found for specifier: '{path_specifier}'")
            return None

        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.is_file():
            self.logger.error(f"Resolved checkpoint path does not exist or is not a file: {checkpoint_path}")
            return None

        self.logger.info(f"Loading checkpoint state from: {checkpoint_path}")
        loaded_dict: Optional[Dict[str, Any]] = None
        training_state: Optional[TrainingState] = None

        # --- Phase 1: Load dictionary and instantiate TrainingState --- #
        try:
            loaded_dict = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(loaded_dict, dict):
                raise CheckpointLoadError(f"File {checkpoint_path} did not contain a dictionary.")

            required_keys = {"epoch", "global_step", "model_state_dict"}
            if not required_keys.issubset(loaded_dict.keys()):
                missing_keys = required_keys - loaded_dict.keys()
                raise CheckpointLoadError(f"Checkpoint is missing required keys: {missing_keys}")

            # Instantiate TrainingState - may succeed even with invalid nested config
            training_state = TrainingState(**loaded_dict)
            self.logger.info(f"Successfully loaded and validated base TrainingState from {checkpoint_path} (Step: {training_state.global_step}).")

        except FileNotFoundError:
            # This specific error should likely propagate or be handled differently
            # Re-raising for now, as the calling context (e.g., Trainer) might handle it.
            self.logger.error(f"Checkpoint file not found during load attempt: {checkpoint_path}")
            raise # Re-raise FileNotFoundError
        except CheckpointLoadError as e:
            # Catch errors specifically from initial loading/key checks
            self.logger.error(f"Failed to load basic checkpoint structure from {checkpoint_path}: {e}")
            return None # Return None for these initial load failures
        except Exception as e:
            # Catch other potential errors during torch.load or TrainingState init
            self.logger.exception(f"Unexpected error during initial load/validation of {checkpoint_path}: {e}")
            # Wrap in CheckpointLoadError before returning None? Or just return None?
            # Returning None for consistency with CheckpointLoadError handling.
            return None

        # --- Phase 2: Validate loaded configuration (if TrainingState instantiation succeeded) --- #
        # This runs only if the basic structure was loaded correctly.
        # We now expect ValidationError or CheckpointLoadError from *this* block to propagate.
        try:
            if training_state.config and self.model and hasattr(self.model, 'config') and isinstance(getattr(self.model, 'config', None), BaseModel):
                self.logger.debug("Performing validation of loaded checkpoint configuration against current model configuration schema...")
                # Get the Pydantic model class of the current model's config
                model_config_type = type(self.model.config)
                # Validate the loaded dictionary using the model's config type
                model_config_type.model_validate(training_state.config)
                self.logger.info("Loaded checkpoint config is compatible with the current model's config schema.")
            elif training_state.config:
                self.logger.warning("Checkpoint contains configuration, but cannot validate it (model or model.config missing/invalid).")
            else:
                self.logger.info("Checkpoint does not contain a configuration dictionary ('config' key missing). Skipping config validation.")

        except ValidationError as e:
            # Log specific validation errors
            error_details = e.errors()
            self.logger.warning(
                f"Configuration mismatch detected loading checkpoint {checkpoint_path}! "
                f"The configuration saved in the checkpoint is not compatible with the "
                f"current model's configuration schema ({model_config_type.__name__}). "
                f"Validation Errors: {error_details}"
            )
            # Raise CheckpointLoadError to signal failure due to config mismatch
            raise CheckpointLoadError(
                f"Configuration mismatch: Loaded config failed validation against "
                f"{model_config_type.__name__}. Errors: {error_details}"
            ) from e
            # No broad Exception catch here, let config validation errors propagate if needed.

        # --- Phase 3: Load Component States (Using the loaded TrainingState) --- #
        # Moved state loading logic here, after config validation
        try:
            # Model State
            if self.model:
                self.logger.debug(f"Loading model state_dict (strict={self.load_strict})...")
                # Use strict mode based on manager config
                missing, unexpected = self.model.load_state_dict(training_state.model_state_dict, strict=self.load_strict) # type: ignore
                if missing: self.logger.warning(f"Missing keys when loading model state_dict: {missing}")
                if unexpected: self.logger.warning(f"Unexpected keys when loading model state_dict: {unexpected}")
            else:
                self.logger.warning("No model provided to CheckpointManager, skipping model state load.")

            # Optimizer State (if saved and optimizer provided)
            if training_state.optimizer_state_dict is not None:
                if self.optimizer:
                    self.logger.debug("Loading optimizer state_dict...")
                    self.optimizer.load_state_dict(training_state.optimizer_state_dict)
                else:
                    self.logger.warning("Optimizer state found in checkpoint, but no optimizer provided to CheckpointManager. Skipping load.")
            # Add similar blocks for scheduler, scaler, callbacks if needed here
            # Scheduler State
            if training_state.scheduler_state_dict is not None:
                if self.scheduler:
                    self.logger.debug("Loading scheduler state_dict...")
                    self.scheduler.load_state_dict(training_state.scheduler_state_dict)
                else:
                    self.logger.warning("Scheduler state found in checkpoint, but no scheduler provided. Skipping load.")

            # Scaler State
            if training_state.scaler_state_dict is not None:
                if self.scaler:
                    # Check if scaler is enabled before loading
                    if self.scaler.is_enabled():
                        self.logger.debug("Loading scaler state_dict...")
                        self.scaler.load_state_dict(training_state.scaler_state_dict)
                    else:
                        self.logger.warning("Scaler state found in checkpoint, but AMP scaler is disabled. Skipping load.")
                else:
                    self.logger.warning("Scaler state found in checkpoint, but no scaler provided. Skipping load.")

            # Callbacks State
            if training_state.callbacks_state is not None:
                if self.callbacks:
                    self.logger.debug("Loading callbacks state_dict...")
                    try:
                        # Assuming CallbackList has load_state_dict
                        self.callbacks.load_state_dict(training_state.callbacks_state)
                        # Also call the on_load_checkpoint hook on individual callbacks
                        for cb in self.callbacks.callbacks:
                            if hasattr(cb, 'on_load_checkpoint') and callable(getattr(cb, 'on_load_checkpoint')):
                                cb.on_load_checkpoint(state=training_state) # Pass the loaded state
                    except Exception as cb_load_e:
                        self.logger.error(f"Error loading callback state: {cb_load_e}", exc_info=True)
                else:
                    self.logger.warning("Callbacks state found in checkpoint, but no callbacks object provided. Skipping load.")

            # Tokenizer (Load from standard location if tokenizer provided)
            if self.tokenizer:
                # Construct the expected path relative to the checkpoint directory's parent
                tokenizer_load_path = self.checkpoint_dir.parent / "tokenizer"
                if tokenizer_load_path.is_dir():
                    self.logger.info(f"Attempting to load tokenizer from standard location: {tokenizer_load_path}...")
                    try:
                        self.tokenizer.load(str(tokenizer_load_path))
                        self.logger.info(f"Successfully loaded tokenizer from {tokenizer_load_path}.")
                    except NotImplementedError:
                        self.logger.warning(f"Tokenizer {type(self.tokenizer).__name__} does not support load(), skipping.")
                    except Exception as tok_e:
                        self.logger.warning(f"Failed to load tokenizer from {tokenizer_load_path}: {tok_e}. Continuing without loading tokenizer.")
                else:
                    self.logger.info(f"Standard tokenizer directory ({tokenizer_load_path}) not found. Skipping tokenizer load.")
            else:
                self.logger.debug("No tokenizer provided to CheckpointManager, skipping tokenizer load.")

            self.logger.info(f"Successfully loaded component states from checkpoint {checkpoint_path}.")
            return training_state

        except CheckpointLoadError as e:
            # Catch errors during component state loading
            self.logger.error(f"Error applying loaded state from checkpoint {checkpoint_path}: {e}", exc_info=True)
            return None
        except Exception as e:
            # Catch unexpected errors during component state loading
            self.logger.exception(f"Unexpected error applying state from {checkpoint_path}: {e}")
            return None

    def _find_checkpoint_path(self, path_specifier: str) -> Optional[str]:
        """Finds the actual checkpoint file path based on 'latest', 'best', filename, or full path."""
        # ... (implementation remains largely the same, just returns the path string) ...
        # Ensure it returns the full path string or None
        path_specifier = path_specifier.strip()

        # Case 1: Full path provided
        if os.path.isabs(path_specifier) and path_specifier.endswith(".pt"):
            if Path(path_specifier).is_file():
                self.logger.info(f"Using provided absolute path: {path_specifier}")
                return path_specifier
            else:
                self.logger.warning(f"Provided absolute path does not exist: {path_specifier}")
                return None # Or raise?
        
        # Case 2: Relative path (assuming relative to checkpoint_dir)
        potential_rel_path = self.checkpoint_dir / path_specifier
        if path_specifier.endswith(".pt") and potential_rel_path.is_file():
            self.logger.info(f"Using resolved relative path: {potential_rel_path}")
            return str(potential_rel_path)

        # Case 3: Keywords 'latest' or 'best'
        if path_specifier.lower() == 'latest':
            return self.find_latest_checkpoint()
        elif path_specifier.lower() == 'best':
             # Assuming find_best_checkpoint returns the path to the single best
             best_list = self.find_best_checkpoint()
             if best_list: 
                 self.logger.info(f"Found best checkpoint: {best_list[0]}")
                 return best_list[0] # Return the first (presumably only) best checkpoint path
             else:
                 self.logger.warning("'best' specified, but no best checkpoint found (or tracking disabled). Trying latest.")
                 return self.find_latest_checkpoint()
                 
        # Case 4: Just a filename (e.g., "checkpoint_step_1000.pt")
        potential_path = self.checkpoint_dir / path_specifier
        if potential_path.is_file():
            self.logger.info(f"Found checkpoint by filename in checkpoint directory: {potential_path}")
            return str(potential_path)

        # If none of the above match
        self.logger.warning(f"Could not resolve path specifier '{path_specifier}' to an existing checkpoint file.")
        return None

    def find_latest_checkpoint(self) -> Optional[str]:
        """Finds the path to the most recent checkpoint based on step number."""
        checkpoints = self._get_sorted_checkpoints() # Gets list of (path, step, time)
        if not checkpoints:
            self.logger.info("No valid checkpoints found to determine the latest.")
            return None
        latest_checkpoint_path = checkpoints[-1][0] # Last element is the latest
        self.logger.info(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path

    def find_best_checkpoint(self, metric_name: str = "val_loss") -> List[str]:
        """Finds the paths to the best checkpoint(s) based on a metric (placeholder)."""
        # Placeholder: Currently just checks for 'best.pt' symlink or file
        # TODO: Implement proper tracking based on metrics saved in TrainingState
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists(): # Could be a symlink or a direct copy
            self.logger.info(f"Found best checkpoint marker: {best_path}")
            # If it's a symlink, resolve it to the actual file
            if best_path.is_symlink():
                 try:
                     resolved_path = str(best_path.resolve(strict=True))
                     self.logger.info(f"Resolved best.pt symlink to: {resolved_path}")
                     return [resolved_path]
                 except FileNotFoundError:
                      self.logger.warning("best.pt is a broken symlink.")
                      return []
                 except Exception as e:
                      self.logger.warning(f"Error resolving best.pt symlink: {e}")
                      return [] # Treat as not found if resolution fails
            else:
                 return [str(best_path)] # It's a regular file named best.pt
        else:
            self.logger.info("No 'best.pt' checkpoint marker found.")
            return []
        # Proper implementation would involve loading metrics from saved checkpoints
        # or maintaining a separate record of best scores/paths.
        # For now, rely on simple 'best.pt' convention.

    def _get_sorted_checkpoints(self) -> List[Tuple[str, int, float]]:
        """Scans directory, parses names, and returns sorted list of (path, step, mtime)."""
        checkpoints = []
        self.logger.debug(f"Scanning {self.checkpoint_dir} for checkpoints to sort...")
        # Iterate over the current self.saved_checkpoints list populated by _scan_for_existing_checkpoints
        for ckpt_path_tuple in self.saved_checkpoints:
             ckpt_path = ckpt_path_tuple[0] # Get path string
             self.logger.debug(f"[Sort] Processing saved path: {ckpt_path}") # ADDED
             ckpt_path_obj = Path(ckpt_path)
             if ckpt_path_obj.is_file(): # Check if file still exists
                 basename = os.path.basename(ckpt_path)
                 parsed_info = self._parse_checkpoint_name(basename) # Parse basename
                 self.logger.debug(f"[Sort] Parsed info for {basename}: {parsed_info}") # ADDED
                 if parsed_info:
                     step_number = parsed_info[1]
                     try:
                         mtime = ckpt_path_obj.stat().st_mtime
                         checkpoints.append((str(ckpt_path), step_number, mtime))
                     except FileNotFoundError:
                         self.logger.warning(f"Could not stat file while sorting: {ckpt_path}")
                         continue # Skip if file vanished
                 else:
                     self.logger.debug(f"Could not parse step number from: {ckpt_path}")
             else:
                 self.logger.debug(f"Ignoring non-file item: {ckpt_path}")

        # Sort by step number descending (latest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        # Correctly log the final sorted list of tuples
        self.logger.debug(f"Sorted checkpoints found: {checkpoints}")
        return checkpoints