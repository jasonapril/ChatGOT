#!/usr/bin/env python
"""
Main Trainer Module
==================

This module provides the main Trainer class that integrates all training components.
"""

import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import time
import hashlib
from pathlib import Path
from hydra.utils import get_original_cwd
import hydra

from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .checkpointing import CheckpointManager, TrainingState, CheckpointLoadError
from .callbacks import CallbackList, TensorBoardLogger
from .callbacks.sample_generation import SampleGenerationCallback
from .generation import TextGenerator
from .progress import ProgressTracker
from ..data.tokenizers.base import Tokenizer
from ..config.schemas import TrainingConfig, DataConfig, AnyModelConfig
from ..utils.logging import setup_logging, force_flush_logs, format_time

# Helper to get a representation of state (e.g., hash of params)
def get_state_hash(component_state_dict):
    """Creates a hash of a state dict for quick comparison."""
    try:
        # Convert state dict to a string representation
        # Note: This might be slow for large states, consider sampling or norms
        state_str = str(component_state_dict)
        return hashlib.md5(state_str.encode()).hexdigest()
    except Exception as e:
        return f"ErrorHashingState: {e}"

class Trainer:
    """Main trainer class that coordinates all training components."""
    
    def __init__(
        self,
        model_config: Union[DictConfig, dict], # Accept model config dict
        config: TrainingConfig, # Expect TrainingConfig Pydantic model directly
        experiment_config: Optional[DictConfig] = None, # Accept full OmegaConf for instantiation
        experiment_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        compile_model: bool = False,  # Added compile_model argument
        **kwargs: Any # Allow extra arguments for flexibility
    ):
        # Basic setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trainer...")
        self.config = self._validate_training_config(config)
        self.experiment_config = experiment_config # Store potentially needed OmegaConf version
        self.device = kwargs.get('device') or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name or "default"
        self.resume_from_checkpoint = resume_from_checkpoint
        self.compile_model = compile_model

        # --- Instantiate components using Hydra from experiment_config --- #
        if not self.experiment_config:
             raise ValueError("Experiment configuration (OmegaConf DictConfig) is required for Trainer setup.")

        # 1. Instantiate Tokenizer (optional)
        self._setup_tokenizer()

        # 2. Instantiate DataLoaders
        self._setup_dataloaders()

        # 3. Instantiate Model (uses self.tokenizer, handles vocab_size injection)
        self._setup_model(model_config)

        # 4. Instantiate Optimizer & Scheduler (uses self.model)
        self._setup_optimizer_scheduler()

        # 5. Instantiate Evaluator (uses self.model, self.val_dataloader)
        self._setup_evaluator()

        # 6. Instantiate Callbacks
        self._setup_callbacks(kwargs.get('callbacks'))

        # 7. Setup Checkpointing (uses model, optimizer, scheduler, tokenizer, callbacks)
        self._setup_checkpointing()

        # Training state initialization
        self._initialize_training_state()

        # Resume from checkpoint if specified (needs components initialized)
        self._resume_if_needed()

        # Final setup (e.g., TensorBoard dir after potential resume)
        self._finalize_setup()

        self.logger.info("Trainer initialization complete.")

    def _validate_training_config(self, config: Any) -> TrainingConfig:
        """Validate or parse the main training configuration."""
        if isinstance(config, TrainingConfig):
            return config
        elif isinstance(config, (dict, DictConfig)):
            self.logger.warning("Received dict/DictConfig for TrainingConfig, attempting parse.")
            try:
                # Convert OmegaConf to dict if necessary before Pydantic parsing
                if _OMEGACONF_AVAILABLE and isinstance(config, DictConfig):
                    config_dict = OmegaConf.to_container(config, resolve=True)
                else:
                    config_dict = config
                return TrainingConfig(**config_dict)
            except Exception as e:
                self.logger.error(f"Failed to parse provided config dict into TrainingConfig: {e}")
                raise ValueError("Invalid training config provided.") from e
        else:
             raise TypeError(f"Expected config to be TrainingConfig, dict, or DictConfig, got {type(config)}")

    def _setup_tokenizer(self):
        """Instantiates the tokenizer using Hydra if configured."""
        self.tokenizer = None
        if hasattr(self.experiment_config, 'data') and hasattr(self.experiment_config.data, 'tokenizer') and self.experiment_config.data.tokenizer:
            self.logger.info("Instantiating tokenizer from data config...")
            try:
                # Add check for _target_ before instantiation
                if not OmegaConf.select(self.experiment_config.data.tokenizer, '_target_', default=None):
                     self.logger.warning("data.tokenizer config is present but missing '_target_'. Cannot instantiate.")
                     return
                self.tokenizer = hydra.utils.instantiate(self.experiment_config.data.tokenizer)
                if self.tokenizer:
                     self.logger.info(f"Tokenizer {type(self.tokenizer).__name__} instantiated successfully.")
                else:
                     # Instantiate might return None if config resolves to null
                     self.logger.warning("Hydra instantiation for tokenizer returned None.")
            except Exception as e:
                self.logger.error(f"Failed to instantiate tokenizer from config: {e}", exc_info=True)
                # Decide if this is critical - maybe proceed without tokenizer?
        else:
            self.logger.info("No tokenizer configuration found in experiment_config.data.tokenizer")

    def _setup_dataloaders(self):
        """Instantiates dataloaders using Hydra, potentially via a factory target."""
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None # Add placeholder for test loader

        if hasattr(self.experiment_config, 'data') and self.experiment_config.data:
            self.logger.info("Instantiating dataloaders from data config...")
            try:
                # Instantiate the entire data config node.
                # This assumes data_config's _target_ points to a function like
                # create_dataloaders(cfg: DataConfig, tokenizer: Tokenizer) -> Dict[str, DataLoader]
                # OR that the nested dataset targets handle tokenizer injection.
                if not OmegaConf.select(self.experiment_config.data, '_target_', default=None):
                     self.logger.warning("experiment_config.data is present but missing '_target_'. Cannot instantiate dataloaders automatically. Manual setup required if Trainer is used directly.")
                     # Attempt to manually create from datasets if structure matches old way
                     # This is a fallback / transitional step
                     if hasattr(self.experiment_config.data, 'datasets'):
                         self.logger.warning("Attempting manual dataloader creation from data.datasets...")
                         self._manual_dataloader_creation(self.experiment_config.data)
                     return # Exit if no target and no datasets sub-config

                # Pass the tokenizer we potentially created in _setup_tokenizer
                # Hydra will pass it as a kwarg if the target function accepts it.
                dataloaders_dict = hydra.utils.instantiate(
                    self.experiment_config.data,
                    tokenizer=self.tokenizer,
                    _convert_="partial"
                 ) # hydra instantiate handles the structure

                if not isinstance(dataloaders_dict, dict):
                     raise TypeError(f"Instantiating data config did not return a dictionary of dataloaders, got {type(dataloaders_dict)}")

                self.train_dataloader = dataloaders_dict.get('train')
                self.val_dataloader = dataloaders_dict.get('val')
                self.test_dataloader = dataloaders_dict.get('test') # Assign test if present

                if self.train_dataloader:
                    self.logger.info("Train dataloader instantiated.")
                else:
                     # This should likely be an error
                     raise ValueError("Dataloader instantiation did not produce a 'train' dataloader.")
                if self.val_dataloader:
                     self.logger.info("Validation dataloader instantiated.")
                else:
                     self.logger.info("Validation dataloader not found or configured.")
                if self.test_dataloader:
                     self.logger.info("Test dataloader instantiated.")

            except Exception as e:
                self.logger.error(f"Failed to instantiate dataloaders from config: {e}", exc_info=True)
                raise ValueError("Dataloader instantiation failed.") from e
        else:
            raise ValueError("Data configuration (experiment_config.data) is required for Trainer.")

    # Fallback for manual creation if data._target_ is missing
    def _manual_dataloader_creation(self, data_cfg: DictConfig):
        # Avoid circular import, import hydra if not already available
        try:
            import hydra.utils
        except ImportError:
            self.logger.error("Hydra is required for manual dataset instantiation fallback.")
            raise

        # from .creation import create_dataset, create_dataloader # Old import
        from .utils import create_dataloader # Import create_dataloader from new location

        dataloaders = {}
        required_splits = ['train', 'val']
        if not hasattr(data_cfg, 'datasets') or not data_cfg.datasets:
             self.logger.error("Manual dataloader creation fallback failed: data_cfg missing 'datasets' attribute.")
             return

        for split_name, split_config in data_cfg.datasets.items():
            if not split_config: continue
            # Ensure dataset config exists and has a target
            if not hasattr(split_config, 'dataset') or not split_config.dataset:
                 self.logger.warning(f"Skipping manual creation for split '{split_name}': Missing 'dataset' config.")
                 continue
            if not OmegaConf.select(split_config.dataset, '_target_', default=None):
                 self.logger.warning(f"Skipping manual creation for split '{split_name}': Dataset config missing '_target_'. Config: {split_config.dataset}")
                 continue

            try:
                # dataset = create_dataset(split_config.dataset, tokenizer=self.tokenizer) # Old way
                # Use Hydra instantiate directly
                self.logger.info(f"Manually instantiating dataset for split '{split_name}' using Hydra...")
                dataset = hydra.utils.instantiate(
                    split_config.dataset,
                    tokenizer=self.tokenizer,
                    _convert_="partial"
                 )
                if not isinstance(dataset, Dataset):
                     raise TypeError(f"Instantiated dataset for {split_name} is not a torch Dataset.")

                # Prepare dataloader config (combine common and split-specific)
                # Ensure dl_cfg is a standard dict for create_dataloader
                common_dl_params = {k: v for k, v in data_cfg.items() if k not in ['datasets', 'tokenizer', '_target_']}
                split_dl_params = OmegaConf.to_container(split_config.get('dataloader', {}), resolve=True)
                dl_cfg = {**common_dl_params, **split_dl_params} # Split params override common

                # Use the imported create_dataloader
                loader = create_dataloader(dataset, dl_cfg) # Pass standard dict
                dataloaders[split_name] = loader
                self.logger.info(f"Successfully created manual dataloader for split '{split_name}'.")
            except Exception as e:
                 self.logger.error(f"Manual creation failed for split '{split_name}': {e}", exc_info=True)
                 if split_name in required_splits: raise # Re-raise if essential split fails
        # Assign loaders
        self.train_dataloader = dataloaders.get('train')
        self.val_dataloader = dataloaders.get('val')
        self.test_dataloader = dataloaders.get('test')
        self.logger.info("Manual dataloader creation fallback attempt complete.")

    def _setup_model(self, model_config: Union[DictConfig, dict]):
        """Instantiates the model using Hydra, injecting vocab_size if available."""
        self.logger.info("Setting up model...")

        # Convert OmegaConf DictConfig to standard dict if necessary for manipulation
        if isinstance(model_config, DictConfig):
            model_dict = OmegaConf.to_container(model_config, resolve=True)
            if not isinstance(model_dict, dict):
                 raise TypeError(f"Resolved model_config is not a dict: {type(model_dict)}")
        elif isinstance(model_config, dict):
            model_dict = model_config.copy()
        else:
            raise TypeError(f"model_config must be DictConfig or dict, got {type(model_config)}")

        # --- Inject vocab_size from tokenizer --- #
        if self.tokenizer is not None and hasattr(self.tokenizer, 'get_vocab_size'):
            try:
                vocab_size = self.tokenizer.get_vocab_size()
                if vocab_size is not None:
                    if 'vocab_size' in model_dict and model_dict['vocab_size'] != vocab_size:
                        self.logger.warning(
                            f"Model config vocab_size ({model_dict['vocab_size']}) differs from "
                            f"tokenizer vocab_size ({vocab_size}). Using tokenizer size."
                        )
                    elif 'vocab_size' not in model_dict or model_dict['vocab_size'] is None:
                        self.logger.info(f"Injecting vocab_size={vocab_size} from tokenizer into model config.")
                    model_dict['vocab_size'] = vocab_size
                else:
                     self.logger.warning("tokenizer.get_vocab_size() returned None.")
            except Exception as e:
                self.logger.error(f"Failed to get vocab_size from tokenizer: {e}")
        elif 'vocab_size' not in model_dict or model_dict['vocab_size'] is None:
             self.logger.warning("Tokenizer not available or has no vocab_size, and model config vocab_size is not set. This might cause issues.")
        # -------------------------------------- #

        # --- Instantiate using Hydra --- #
        try:
            # Ensure _target_ exists
            if "_target_" not in model_dict:
                 # Maybe try to get it from experiment_config.model if model_config was just params?
                 # Or rely on schema validation to have enforced it earlier.
                 # For now, assume model_config passed to Trainer should have _target_.
                 # Check if architecture exists and try to infer target? Seems fragile.
                 if "architecture" in model_dict:
                      self.logger.warning(f"Model config is missing '_target_'. Attempting to infer from experiment_config based on architecture: {model_dict['architecture']}...")
                      # Try to find the target in the original experiment config
                      if self.experiment_config and OmegaConf.select(self.experiment_config, f"model.architecture") == model_dict['architecture']:
                          target_path = OmegaConf.select(self.experiment_config, "model._target_")
                          if target_path:
                              self.logger.info(f"Found target '{target_path}' in experiment_config.model. Injecting it.")
                              model_dict["_target_"] = target_path
                          else:
                               raise ValueError("Model config missing '_target_' and couldn't find it in experiment_config.model.")
                      else:
                           raise ValueError("Model config missing '_target_' and architecture mismatch or experiment_config unavailable.")
                 else:
                    raise ValueError("Model configuration must include '_target_'.")

            self.logger.info(f"Instantiating model with _target_={model_dict.get('_target_')}...")
            # Use _convert_="partial" to allow nested instantiation if needed later
            # Hydra automatically handles passing the dictionary keys as kwargs
            self.model = hydra.utils.instantiate(model_dict, _convert_="partial")

            if not isinstance(self.model, nn.Module):
                self.logger.warning(f"Instantiated model {type(self.model)} is not an nn.Module subclass.")

            self.logger.info(f"Model {type(self.model).__name__} instantiated successfully.")

        except Exception as e:
            self.logger.error(f"Failed to instantiate model from config: {e}", exc_info=True)
            self.logger.debug(f"Model config dictionary used for instantiation: {model_dict}")
            raise ValueError("Model instantiation failed.") from e

        # --- Move model to device --- #
        if self.model and self.device:
            try:
                self.model.to(self.device)
                self.logger.info(f"Moved model to device: {self.device}")
            except Exception as e:
                self.logger.error(f"Failed to move model to device {self.device}: {e}")
                # Decide if this is critical - potentially raise?
        # --------------------------- #

    def _setup_optimizer_scheduler(self):
        """Creates the optimizer and scheduler using Hydra instantiation."""
        # --- Optimizer --- #
        self.logger.info("Creating optimizer...")
        self.optimizer = None
        if not self.experiment_config or not hasattr(self.experiment_config, 'optimizer'):
             raise ValueError("Optimizer configuration missing in experiment config.")
        try:
            self.optimizer = hydra.utils.instantiate(
                self.experiment_config.optimizer,
                params=self.model.parameters(),
                _convert_="partial"
            )
            self.logger.info(f"Optimizer {type(self.optimizer).__name__} created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to instantiate optimizer from config: {e}", exc_info=True)
            raise ValueError("Optimizer instantiation failed.") from e

        # --- Scheduler --- #
        self.logger.info("Creating scheduler...")
        self.scheduler = None
        if hasattr(self.experiment_config, 'scheduler') and self.experiment_config.scheduler:
            scheduler_cfg_node = self.experiment_config.scheduler
            scheduler_cfg_copy = OmegaConf.to_container(scheduler_cfg_node, resolve=False)
            if not isinstance(scheduler_cfg_copy, dict):
                raise TypeError("Resolved scheduler config is not a dict")

            # Calculate/verify T_max
            t_max_key = 'T_max' # Common key, adjust if needed
            max_steps = getattr(self.config, 'max_steps', None)
            num_epochs = getattr(self.config, 'num_epochs', None)
            current_t_max = scheduler_cfg_copy.get(t_max_key)

            if max_steps is not None:
                 if current_t_max != max_steps:
                     self.logger.warning(f"Scheduler T_max ({current_t_max}) will be overridden by training max_steps ({max_steps}).")
                     scheduler_cfg_copy[t_max_key] = max_steps
            elif num_epochs is not None and current_t_max is None or current_t_max == '???':
                 if self.train_dataloader:
                     try:
                         steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
                         total_steps = num_epochs * steps_per_epoch
                         self.logger.info(f"Calculated total_steps for scheduler T_max: {total_steps}")
                         scheduler_cfg_copy[t_max_key] = total_steps
                     except Exception as e:
                         raise ValueError("Scheduler T_max calculation failed.") from e
                 else:
                     raise ValueError("Cannot calculate scheduler T_max: train_loader missing.")
            elif current_t_max is None or current_t_max == '???':
                 raise ValueError(f"Scheduler config requires '{t_max_key}', but it was not set and could not be determined from training config.")

            try:
                self.scheduler = hydra.utils.instantiate(
                    scheduler_cfg_copy,
                    optimizer=self.optimizer,
                    _convert_="partial"
                )
                self.logger.info(f"Scheduler {type(self.scheduler).__name__} created successfully.")
            except Exception as e:
                self.logger.error(f"Failed to instantiate scheduler from config: {e}", exc_info=True)
                raise ValueError("Scheduler instantiation failed.") from e
        else:
            self.logger.info("No scheduler configuration provided.")

    def _setup_evaluator(self):
        """Initializes the Evaluator if a validation dataloader is provided."""
        self.evaluator = None
        if self.val_dataloader:
            self.logger.info("Validation dataloader provided. Initializing Evaluator.")
            try:
                self.evaluator = Evaluator(
                    model=self.model,
                    val_dataloader=self.val_dataloader,
                    device=self.device,
                    config=self.config
                )
            except Exception as e:
                 self.logger.error(f"Failed to initialize Evaluator: {e}", exc_info=True)
                 # Decide if this is critical - maybe proceed without evaluator?
                 self.evaluator = None
        else:
            self.logger.info("No validation dataloader provided. Skipping Evaluator initialization.")

    def _setup_callbacks(self, callbacks_list: Optional[List[Any]]):
        """Initializes the CallbackList and instantiates callbacks via Hydra if needed."""
        self.callbacks = CallbackList([]) # Start with empty list
        if callbacks_list: # If pre-instantiated list is passed
             self.logger.info(f"Using {len(callbacks_list)} pre-instantiated callbacks.")
             self.callbacks = CallbackList(callbacks_list)
        elif self.experiment_config and hasattr(self.experiment_config, 'callbacks') and self.experiment_config.callbacks:
            # Instantiate from config if list wasn't provided
            raw_callbacks_cfg = self.experiment_config.callbacks
            try:
                logger.info("Instantiating callbacks using Hydra...")
                instantiated_callbacks_dict = hydra.utils.instantiate(raw_callbacks_cfg)
                self.callbacks = CallbackList(list(instantiated_callbacks_dict.values()))
                logger.info(f"Instantiated callbacks: {[cb.__class__.__name__ for cb in self.callbacks.callbacks]}")
            except Exception as e:
                logger.error(f"Error instantiating callbacks via Hydra: {e}", exc_info=True)
                # Proceeding without callbacks from config
        else:
            logger.info("No callbacks provided or configured.")

        # Always set the trainer instance on the callback list
        self.callbacks.set_trainer(self)

    def _setup_checkpointing(self):
        """Initializes the CheckpointManager."""
        self.logger.info("Setting up Checkpoint Manager...")
        try:
            # Use original CWD as base for persistent path
            try:
                original_cwd = Path(get_original_cwd())
            except Exception:
                self.logger.warning("Could not get original CWD, using current CWD for checkpoint path.")
                original_cwd = Path.cwd()
            # Construct path relative to original CWD/outputs/experiment_name
            persistent_checkpoint_dir = original_cwd / "outputs" / self.experiment_name / "checkpoints"
            self.logger.info(f"Using persistent checkpoint directory: {persistent_checkpoint_dir}")

            # Access checkpoint settings from TrainingConfig
            checkpoint_dir_from_config = getattr(self.config, 'checkpoint_dir', None)
            if checkpoint_dir_from_config:
                 self.logger.warning(f"Ignoring calculated checkpoint dir {persistent_checkpoint_dir}. Using dir from config: {checkpoint_dir_from_config}")
                 checkpoint_dir_to_use = Path(checkpoint_dir_from_config)
            else:
                 checkpoint_dir_to_use = persistent_checkpoint_dir

            self.checkpoint_manager = CheckpointManager(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                experiment_name=self.experiment_name,
                callbacks=self.callbacks,
                tokenizer=self.tokenizer,
                config=self.config.model_dump(), # Save Pydantic config state
                checkpoint_dir=str(checkpoint_dir_to_use), # Pass determined path
                max_checkpoints_to_keep=getattr(self.config, 'keep_last', 5)
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize CheckpointManager: {e}", exc_info=True)
            raise ValueError("CheckpointManager initialization failed.") from e

    def _initialize_training_state(self):
        """Initializes training state variables."""
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.metrics = {}
        self.loaded_tb_log_dir = None # For resuming TB logs
        self.loaded_state: Optional[TrainingState] = None # Store loaded state
        # Initialize scaler here for AMP
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_amp)

    def _resume_if_needed(self):
        """Handles resuming from checkpoint if configured."""
        if self.resume_from_checkpoint:
            self.logger.info(f"Trainer: Proceeding with resume attempt using path: {self.resume_from_checkpoint}")
            self._resume_from_checkpoint(self.resume_from_checkpoint)
        else:
            self.logger.info("Trainer: No checkpoint path provided for resume.")

        # Log state AFTER potential resume attempt
        self.logger.info(f"[Trainer Init After Resume Check] Model State Hash: {get_state_hash(self.model.state_dict())}")
        if self.optimizer:
            self.logger.info(f"[Trainer Init After Resume Check] Optimizer State Hash: {get_state_hash(self.optimizer.state_dict())}")

    def _finalize_setup(self):
        """Perform final setup steps, e.g., setting TensorBoard directory after resume."""
        # TensorBoard setup after potential resume
        if self.loaded_tb_log_dir:
            for cb in self.callbacks.callbacks:
                if cb.__class__.__name__ == 'TensorBoardLogger':
                    if hasattr(cb, 'set_log_dir_absolute') and callable(getattr(cb, 'set_log_dir_absolute')):
                        cb.set_log_dir_absolute(self.loaded_tb_log_dir)
                        self.logger.info(f"Resuming TensorBoard logging to: {self.loaded_tb_log_dir}")
                    else:
                        self.logger.warning("TensorBoardLogger found, but lacks set_log_dir_absolute method. Will create new log dir.")
                    break # Assume only one TB logger

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Loads state from a checkpoint file using CheckpointManager."""
        if not self.checkpoint_manager:
            self.logger.error("CheckpointManager is not initialized. Cannot resume.")
            return

        try:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.loaded_state = self.checkpoint_manager.load_checkpoint(checkpoint_path)

            if self.loaded_state:
                self.epoch = self.loaded_state.epoch + 1 # Start next epoch
                self.global_step = self.loaded_state.global_step
                self.best_val_metric = self.loaded_state.best_val_metric
                self.loaded_tb_log_dir = self.loaded_state.tensorboard_log_dir
                self.logger.info(f"Resumed from Epoch {self.loaded_state.epoch}, Global Step {self.global_step}")
                self.logger.info(f"Loaded Best Validation Metric: {self.best_val_metric}")
                if self.loaded_tb_log_dir:
                    self.logger.info(f"Loaded previous TensorBoard log dir: {self.loaded_tb_log_dir}")
                    # Find the TensorBoardLogger and set its path
                    for cb in self.callbacks.callbacks:
                        if isinstance(cb, TensorBoardLogger):
                            self.logger.info(f"Setting absolute log dir for TensorBoardLogger: {self.loaded_tb_log_dir}")
                            cb.set_log_dir_absolute(self.loaded_tb_log_dir)
                            break
                else:
                    self.logger.warning("No TensorBoard log directory found in the loaded checkpoint state.")

            else:
                self.logger.warning(f"Checkpoint loading returned None for path: {checkpoint_path}. Starting fresh.")
                self.resume_from_checkpoint = None # Clear flag if load fails

        except FileNotFoundError as e:
             # Specific handling for file not found
             self.logger.error(f"Checkpoint file not found during resume: {e}")
             self.resume_from_checkpoint = None
             self.logger.warning("Cleared resume_from_checkpoint flag.")
             raise e # Re-raise the error so the test can catch it
        except CheckpointLoadError as e:
            self.logger.error(f"Checkpoint load error during resume: {e}")
            self.resume_from_checkpoint = None 
            self.logger.warning("Cleared resume_from_checkpoint flag due to load failure.")
            raise e # Re-raise the error
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during resume from checkpoint: {e}", exc_info=True)
            self.resume_from_checkpoint = None
            self.logger.warning("Cleared resume_from_checkpoint flag due to unexpected error.")
            raise e # Re-raise unexpected errors

    def train(self):
        """Main training loop."""
        self.logger.info("--- Starting Training Run ---")
        self.callbacks.on_train_begin() # Don't pass self
        self._stop_training = False # Flag to signal training should stop

        # Instantiate the training loop
        self.training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            train_dataloader=self.train_dataloader,
            device=self.device,
            config=self.config, # Pass Pydantic TrainingConfig
            experiment_config=self.experiment_config, # Pass OmegaConf DictConfig
            scheduler=self.scheduler,
            use_amp=self.config.use_amp,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            log_interval=self.config.log_interval,
            callbacks=self.callbacks,
            checkpoint_manager=self.checkpoint_manager, # Pass the manager
            save_steps_interval=self.config.save_steps_interval, # Pass interval
            max_steps=self.config.max_steps, # Pass max steps
            time_save_interval_seconds=self.config.time_save_interval_seconds,
        )
        # Ensure model is on the correct device *before* initializing TrainingLoop
        # self.model.to(self.device) # Already done in Trainer.__init__

        # --- Handle immediate actions after resume ---
        # REMOVED Immediate Save Block
        # if self._just_resumed_trigger_save:
        #    ...

        if self._just_resumed_trigger_sample:
            self.logger.info("Performing immediate sample after resuming.")
            # Attempt to directly find and call the callback
            sample_cb = None
            if self.callbacks and hasattr(self.callbacks, 'callbacks'): # Check if CallbackList and its list exist
                 # Find the specific callback (assuming it's the correct type)
                 try:
                     # Ensure import is available
                     from .callbacks.sample_generation import SampleGenerationCallback
                     # Corrected indentation for the loop and its content
                     for cb in self.callbacks.callbacks:
                          if isinstance(cb, SampleGenerationCallback):
                               sample_cb = cb
                               break
                 except ImportError:
                      self.logger.error("Could not import SampleGenerationCallback for direct check.")
                 except Exception as e:
                      self.logger.error(f"Error finding SampleGenerationCallback instance: {e}")

            if sample_cb:
                try:
                    self.logger.debug(f"[Resume Sample] Found callback via direct check: {type(sample_cb).__name__}. Calling generate_samples...")
                    sample_cb.generate_samples(trigger_event=f"resume at step {self.global_step}")
                    # Reset the TrainingLoop timer if possible
                    if hasattr(self.training_loop, 'last_time_based_sample'):
                        self.training_loop.last_time_based_sample = time.time()
                except Exception as e:
                    self.logger.error(f"Error calling generate_samples on found callback: {e}", exc_info=True)
                else:
                    self.logger.warning("Could not find SampleGenerationCallback instance for immediate post-resume sample.")
            self._just_resumed_trigger_sample = False # Reset flag regardless of success
        # --- End Handle immediate actions ---

        # --- DEBUG: Log state before epoch loop ---
        self.logger.debug(f"[Train Start] Just Resumed Flags: Save={self._just_resumed_trigger_save}, Sample={self._just_resumed_trigger_sample}")
        self.logger.debug(f"[Train Start] Callbacks object: {type(self.callbacks).__name__}, Number of callbacks: {len(self.callbacks.callbacks) if self.callbacks and hasattr(self.callbacks, 'callbacks') else 'N/A'}")
        # --- END DEBUG --- 

        # Log state JUST BEFORE epoch loop starts
        self.logger.info(f"[Trainer Train] PRE-LOOP: Model State Hash: {get_state_hash(self.model.state_dict())}")
        if self.optimizer:
             self.logger.info(f"[Trainer Train] PRE-LOOP: Optimizer State Hash: {get_state_hash(self.optimizer.state_dict())}")
        
        max_steps_reached = False # Initialize here
        try:
            # --- Evaluate before training starts (if eval_before_train is True) ---
            if hasattr(self.config, 'eval_before_train') and self.config.eval_before_train and self.val_dataloader and self.evaluator:
                self.logger.info("Evaluating model before training starts...")
                # Use self.evaluator instead of self.evaluate
                val_metrics_before = self.evaluator.evaluate()
                if val_metrics_before:
                    self.logger.info(f"Pre-training evaluation metrics: {val_metrics_before}")
            # --- End Pre-Train Evaluation --- 

            # Calculate total steps across epochs for progress bar
            total_steps_for_epochs = self.num_epochs * len(self.train_dataloader)
            # Use max_steps if defined and smaller than total epoch steps
            effective_total_steps = min(self.max_steps, total_steps_for_epochs) if self.max_steps else total_steps_for_epochs
            
            # Initialize Progress Tracker (using effective total steps if max_steps is limiting)
            progress_tracker = ProgressTracker(
                total_steps=effective_total_steps,
                initial_step=self.global_step # Start from potentially resumed step
            )

            # Determine starting epoch (handle resume)
            resume_epoch = 0
            if self.resume_from_checkpoint and self.global_step > 0:
                resume_epoch = self.global_step // len(self.train_dataloader)
                self.logger.info(f"Resuming from global step {self.global_step}, starting at calculated epoch {resume_epoch + 1}")
            start_epoch = resume_epoch # Start from resumed epoch
            
            # Flag to track if initial sample has been generated after resuming
            initial_sample_generated = False

            # Main epoch loop
            for epoch in range(start_epoch, self.num_epochs):
                self.epoch = epoch
                # Standardize the call: Pass trainer, current_epoch, global_step
                self.callbacks.on_epoch_begin(trainer=self, current_epoch=epoch, global_step=self.global_step)
                self.logger.info(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
                
                # Perform immediate sample generation if just resumed
                if self._just_resumed_trigger_sample and not initial_sample_generated:
                    self.logger.info("Performing immediate sample after resuming.")
                    self._trigger_sample_generation(f"resume at step {self.global_step}")
                    self._just_resumed_trigger_sample = False # Reset flag
                    initial_sample_generated = True # Mark as done
                    
                # --- Debug state hashes before epoch loop --- 
                self.logger.info(f"[Trainer Train] PRE-LOOP: Model State Hash: {get_state_hash(self.model.state_dict())}")
                if self.optimizer:
                    self.logger.info(f"[Trainer Train] PRE-LOOP: Optimizer State Hash: {get_state_hash(self.optimizer.state_dict())}")
                # --- End debug --- 
                
                epoch_metrics = self.training_loop.train_epoch(
                    trainer=self, # Pass trainer instance
                    current_epoch=epoch,
                    global_step=self.global_step,
                    loaded_global_step=self.loaded_state.global_step if self.loaded_state else None,
                    progress=progress_tracker # Pass progress tracker
                )
                # Update global step based on what TrainingLoop finished at
                new_global_step = epoch_metrics.get('final_global_step', self.global_step)
                self.logger.debug(f"[Trainer Epoch End] Before update: self.global_step={self.global_step}, epoch_metrics={epoch_metrics}")
                self.global_step = new_global_step
                self.logger.debug(f"[Trainer Epoch End] After update: self.global_step={self.global_step}")

                # --- Perform Validation ---
                final_val_metrics = None # Ensure it exists even if validation doesn't run
                if self.val_dataloader and self.evaluator and self.eval_interval > 0 and (self.global_step % self.eval_interval == 0 or (self.max_steps is not None and self.global_step >= self.max_steps)):
                    self.logger.info(f"Evaluation interval reached at step {self.global_step}. Evaluating...")
                    # Use self.evaluator instead of self.evaluate
                    final_val_metrics = self.evaluator.evaluate()

                # Combine metrics into a single logs dict for callbacks
                epoch_logs = {**epoch_metrics, **(final_val_metrics or {})} 
                self.callbacks.on_epoch_end(epoch=epoch, global_step=self.global_step, metrics=epoch_logs)

                # Check if max_steps reached based on the updated self.global_step
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    self.logger.info(f"Max steps condition met ({self.global_step} >= {self.max_steps}). Breaking epoch loop.")
                    max_steps_reached = True
                    final_step = self.global_step # Record final step when condition met
                    break # Exit epoch loop

            # If loop finished normally (no break)
            if not max_steps_reached:
                 self.logger.info("Training finished after completing specified epochs/steps.")

        except RuntimeError as e:
            # Catch specific runtime errors first
            self.logger.exception(f"Training interrupted due to runtime error: {e}")
            # Optionally re-raise or handle differently?
            # For now, just log and proceed to finally
        except Exception as e:
            self.logger.exception(f"Training interrupted due to unexpected error: {e}")
            # Any other exception, log and proceed to finally
        finally:
            # This block always executes, regardless of breaks or exceptions
            self.callbacks.on_train_end()
            self.logger.info("--- Training Run Ended ---")
            # --- DEBUG LOGGING --- #
            self.logger.info(f"[Trainer Train End] Final self.global_step: {self.global_step}")
            # --- END DEBUG LOGGING --- #
            
        return self.metrics

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        use_beam_search: bool = False,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> List[str]:
        """Generate text using the trained model."""
        generator = TextGenerator(
            model=self.model,
            device=self.device,
            config=self.config
        )
        return generator.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            use_beam_search=use_beam_search,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping
        )

