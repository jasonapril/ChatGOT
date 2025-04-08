#!/usr/bin/env python
"""
Main Trainer Module
==================

This module provides the main Trainer class that integrates all training components.
"""

import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import time
import hashlib
from pathlib import Path
from hydra.utils import get_original_cwd, instantiate
import hydra
from pydantic import ValidationError

from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .checkpointing import CheckpointManager, TrainingState, CheckpointLoadError
from .callbacks import CallbackList, TensorBoardLogger
from .callbacks.base import Callback
from .callbacks.sample_generation import SampleGenerationCallback
from .generation import TextGenerator
from .progress import ProgressTracker
from ..data.tokenizers.base import Tokenizer
from ..config.schemas import TrainingConfig, DataConfig, AnyModelConfig, ExperimentConfig, LanguageModelConfig
from ..utils.logging import setup_logging, force_flush_logs, format_time
from ..utils.common import set_seed, setup_device

# Helper to get a representation of state (e.g., hash of params)
def get_state_hash(component_state_dict: Dict[str, Any]) -> str:
    """Creates a hash of a state dict for quick comparison."""
    try:
        # Convert state dict to a string representation
        # Note: This might be slow for large states, consider sampling or norms
        state_str = str(component_state_dict)
        return hashlib.md5(state_str.encode()).hexdigest()
    except Exception as e:
        return f"ErrorHashingState: {e}"

class Trainer:
    """
    Main trainer class that coordinates the training loop and evaluation,
    instantiating components from a Hydra configuration object.
    """

    # Type hints for attributes initialized in __init__
    config: TrainingConfig # Validated Pydantic config
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader]
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    tokenizer: Optional[Tokenizer]
    callbacks: CallbackList
    evaluator: Optional[Evaluator]
    checkpoint_manager: Optional[CheckpointManager]
    scaler: Optional[torch.cuda.amp.GradScaler]
    device: torch.device
    resume_from_checkpoint: Optional[str] # From CLI
    compile_model: bool # From CLI
    experiment_name: Optional[str] # From CLI
    raw_config: DictConfig # Store the raw hydra config

    # Internal state attributes
    epoch: int
    global_step: int
    start_time: Optional[float]
    total_train_time: float
    best_val_metric: Optional[float]
    _just_resumed_trigger_eval: bool
    _just_resumed_trigger_save: bool
    _just_resumed_trigger_sample: bool
    _stop_training: bool

    def __init__(
        self,
        cfg: DictConfig, # Main Hydra configuration object
        resume_from_checkpoint: Optional[str] = None,
        compile_model: bool = False,
        # experiment_name: Optional[str] = None, # Removed, get from cfg.experiment
        # Removed pre-instantiated component arguments
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trainer from configuration...")
        self.raw_config = cfg # Store raw config

        # --- Access the nested experiment config ---
        if not cfg.get('experiment'):
            raise ValueError("Configuration is missing the required 'experiment' block.")
        exp_cfg = cfg.experiment # Use this for accessing data, model, training etc.
        # --- End Access --- #

        # 1. Validate Core Training Config
        try:
            # Convert the training section from the experiment config
            training_cfg_node = exp_cfg.get('training')
            if not training_cfg_node:
                raise ValueError("Experiment configuration is missing the 'training' block.")

            # Convert OmegaConf node to dict and parse with Pydantic
            self.logger.debug(f"Attempting to parse TrainingConfig from:\n{OmegaConf.to_yaml(training_cfg_node)}")
            training_dict = OmegaConf.to_container(training_cfg_node, resolve=True)
            if not isinstance(training_dict, dict):
                 raise TypeError(f"OmegaConf.to_container did not return a dict, got {type(training_dict)}")
            self.config = TrainingConfig(**training_dict)

        except ValidationError as e: # Catch Pydantic validation error specifically
             self.logger.error(f"Pydantic validation failed for TrainingConfig: {e}", exc_info=True)
             # Log the dict that failed validation for easier debugging
             try:
                 failed_dict = OmegaConf.to_container(exp_cfg.training, resolve=True)
                 self.logger.debug(f"Training config dict that failed validation:\n{failed_dict}")
             except Exception as log_e:
                 self.logger.debug(f"Could not log failing dict: {log_e}")
             raise
        except Exception as e:
             # Catch other errors during conversion/access
             self.logger.error(f"Failed to convert or validate TrainingConfig from exp_cfg.training: {e}", exc_info=True)
             raise

        # Get resume_from flag: CLI > training config > experiment config > top-level config
        self.resume_from_checkpoint = (
            resume_from_checkpoint # Highest priority: CLI flag passed to __init__
            or self.config.resume_from_checkpoint # Next: from validated TrainingConfig
            or exp_cfg.get("resume_from") # Fallback: from experiment config
            or cfg.get("resume_from") # Lowest priority: from top-level config
        )
        self.compile_model = compile_model # From CLI flag
        self.experiment_name = exp_cfg.get("experiment_name", "default_experiment") # Get from experiment config

        # 2. Setup Device (Prefer experiment level, fallback to top level)
        device_pref = exp_cfg.get('device', cfg.get('device', 'auto'))
        self.device = self._setup_device(device_pref)

        # 3. Instantiate Core Components using Hydra (accessing via exp_cfg)
        try:
            # --- Tokenizer (Optional but often needed) ---
            self.tokenizer = None
            tokenizer_cfg_node = exp_cfg.get("data") and exp_cfg.data.get("tokenizer")
            if tokenizer_cfg_node:
                self.logger.info("Instantiating tokenizer...")
                if isinstance(tokenizer_cfg_node, DictConfig):
                    self.tokenizer = instantiate(tokenizer_cfg_node)
                    self.logger.info(f"Instantiated tokenizer: {type(self.tokenizer).__name__}")
                else:
                    self.logger.warning(f"Tokenizer config found but is not an OmegaConf node ({type(tokenizer_cfg_node)}). Skipping instantiation.")
            else:
                self.logger.info("No tokenizer configuration found in data config.")

            # --- Model ---
            self.logger.info("Instantiating model...")
            model_cfg_node = exp_cfg.get('model')
            if not model_cfg_node:
                 raise ValueError("Experiment configuration is missing the 'model' block.")

            # --- Explicitly instantiate nested Model Config --- #
            config_sub_node = model_cfg_node.get('config')
            if not config_sub_node:
                raise ValueError("Model configuration ('experiment.model.config') is missing.")
            try:
                self.logger.debug(f"Instantiating nested model config ({type(config_sub_node)})...")
                config_dict = OmegaConf.to_container(config_sub_node, resolve=True)
                if not isinstance(config_dict, dict):
                     raise TypeError(f"OmegaConf.to_container(model.config) did not return dict, got {type(config_dict)}")
                # Use the correct schema based on architecture if needed, here assuming LanguageModelConfig
                # TODO: Use the discriminated union AnyModelConfig for validation/parsing here?
                # For now, assume it should be LanguageModelConfig based on the experiment file.
                model_config_obj = LanguageModelConfig(**config_dict) # Parse/validate with Pydantic
                self.logger.debug(f"Nested model config instantiated: {type(model_config_obj)}")
            except ValidationError as e:
                self.logger.error(f"Pydantic validation failed for model config: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to instantiate nested model config: {e}", exc_info=True)
                raise
            # --- End Nested Config Instantiation --- #

            model_instantiate_kwargs = {}
            if self.tokenizer and model_config_obj.vocab_size is None:
                 # Call get_vocab_size() method instead of accessing attribute
                 tokenizer_vocab_size = self.tokenizer.get_vocab_size() # type: ignore
                 if tokenizer_vocab_size:
                      model_config_obj = model_config_obj.model_copy(update={'vocab_size': tokenizer_vocab_size})
                      self.logger.info(f"Injecting vocab_size={tokenizer_vocab_size} from tokenizer into model config object.")
                 else:
                     self.logger.warning("Model config expects vocab_size but tokenizer does not provide it or it's invalid.")

            # Instantiate the main model directly, passing the pre-instantiated config object.
            main_model_target_path = model_cfg_node.get('_target_')
            if not main_model_target_path:
                 raise ValueError("Model configuration ('experiment.model') is missing '_target_'.")
            try:
                 ModelClass = hydra.utils.get_class(main_model_target_path)
                 self.logger.info(f"Located model class: {ModelClass}")
                 # TransformerModel.__init__ only takes the config object
                 self.model = ModelClass(config=model_config_obj)
            except ImportError as e:
                 self.logger.error(f"Could not import model class {main_model_target_path}: {e}")
                 raise
            except Exception as e:
                 self.logger.error(f"Error instantiating model {main_model_target_path} directly: {e}", exc_info=True)
                 raise

            self.model.to(self.device)
            self.logger.info(f"Instantiated model: {type(self.model).__name__}")

            # --- Dataloaders ---
            self.logger.info("Instantiating dataloaders...")
            data_cfg_node = exp_cfg.get('data')
            if not data_cfg_node or not data_cfg_node.get("datasets"):
                 raise ValueError("Configuration section 'experiment.data.datasets' is missing.")
            self.train_dataloader = None
            self.val_dataloader = None
            # Train dataloader
            train_dl_cfg_node = data_cfg_node.datasets.get("train")
            if train_dl_cfg_node:
                # ... (rest of dataloader instantiation logic remains largely the same,
                #      but should source batch_size/num_workers from data_cfg_node)
                dl_kwargs = {"tokenizer": self.tokenizer} if self.tokenizer else {}
                if train_dl_cfg_node.get('dataloader_target'):
                     dataset_instance = instantiate(train_dl_cfg_node.dataset, **dl_kwargs)
                     # Get dataloader params from the correct node
                     dl_params = train_dl_cfg_node.dataloader_params if hasattr(train_dl_cfg_node, 'dataloader_params') else {}
                     self.train_dataloader = instantiate(train_dl_cfg_node.dataloader, dataset=dataset_instance, **dl_params)
                else:
                     self.train_dataloader = instantiate(train_dl_cfg_node, **dl_kwargs)
                     if isinstance(self.train_dataloader, torch.utils.data.Dataset):
                          batch_size = data_cfg_node.get('batch_size', 1)
                          num_workers = data_cfg_node.get('num_workers', 0)
                          self.logger.info(f"Instantiated train dataset ({type(self.train_dataloader).__name__}), wrapping in DataLoader (batch={batch_size}, workers={num_workers})...")
                          self.train_dataloader = DataLoader(
                              dataset=self.train_dataloader,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True
                          )
            else:
                raise ValueError("cfg.experiment.data.datasets.train configuration is missing.")
            # Val dataloader
            val_dl_cfg_node = data_cfg_node.datasets.get("val")
            if val_dl_cfg_node:
                 dl_kwargs = {"tokenizer": self.tokenizer} if self.tokenizer else {}
                 if val_dl_cfg_node.get('dataloader_target'):
                      dataset_instance = instantiate(val_dl_cfg_node.dataset, **dl_kwargs)
                      dl_params = val_dl_cfg_node.dataloader_params if hasattr(val_dl_cfg_node, 'dataloader_params') else {}
                      self.val_dataloader = instantiate(val_dl_cfg_node.dataloader, dataset=dataset_instance, **dl_params)
                 else:
                      self.val_dataloader = instantiate(val_dl_cfg_node, **dl_kwargs)
                      if isinstance(self.val_dataloader, torch.utils.data.Dataset):
                           batch_size = data_cfg_node.get('batch_size', 1)
                           num_workers = data_cfg_node.get('num_workers', 0)
                           self.logger.info(f"Instantiated val dataset ({type(self.val_dataloader).__name__}), wrapping in DataLoader (batch={batch_size}, workers={num_workers})...")
                           self.val_dataloader = DataLoader(
                                dataset=self.val_dataloader,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False
                           )
            self.logger.info(f"Train Dataloader: {type(self.train_dataloader).__name__}")
            self.logger.info(f"Validation Dataloader: {type(self.val_dataloader).__name__ if self.val_dataloader else 'None'}")

            # --- Optimizer ---
            self.logger.info("Instantiating optimizer...")
            optimizer_cfg_node = exp_cfg.get('optimizer')
            if not optimizer_cfg_node:
                 raise ValueError("Experiment configuration is missing the 'optimizer' block.")
            self.optimizer = instantiate(optimizer_cfg_node, params=self.model.parameters())
            self.logger.info(f"Instantiated optimizer: {type(self.optimizer).__name__}")

            # --- Scheduler (Optional) ---
            self.scheduler = None
            scheduler_cfg_node = exp_cfg.get("scheduler")
            if scheduler_cfg_node:
                self.logger.info("Instantiating scheduler...")
                self.scheduler = instantiate(scheduler_cfg_node, optimizer=self.optimizer)
                self.logger.info(f"Instantiated scheduler: {type(self.scheduler).__name__}")
            else:
                 self.logger.info("No scheduler configuration found.")

            # --- AMP GradScaler ---
            self.logger.info(f"Setting up GradScaler (AMP enabled: {self.config.use_amp})...")
            # Use self.config here as it's the validated TrainingConfig
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

            # --- Callbacks ---
            self.logger.info("Instantiating callbacks...")
            callbacks_list: List[Callback] = []
            callbacks_conf_node = exp_cfg.get("callbacks")
            if callbacks_conf_node:
                 if isinstance(callbacks_conf_node, (DictConfig, dict)):
                     for name, cb_conf in callbacks_conf_node.items():
                         if cb_conf is not None and hasattr(cb_conf, '_target_'):
                             try:
                                 callback_instance = instantiate(cb_conf, trainer=self)
                                 if isinstance(callback_instance, Callback):
                                     callbacks_list.append(callback_instance)
                                     self.logger.info(f"  Instantiated callback '{name}': {type(callback_instance).__name__}")
                                 else:
                                     self.logger.warning(f"Instantiated object for callback '{name}' is not a Callback subclass ({type(callback_instance).__name__}). Skipping.")
                             except Exception as e:
                                 self.logger.error(f"Failed to instantiate callback '{name}': {e}", exc_info=True)
                         else:
                              self.logger.warning(f"Callback configuration '{name}' is missing or invalid (_target_). Skipping.")
                 else:
                      self.logger.warning(f"Callbacks config section is not a dictionary/mapping ({type(callbacks_conf_node)}). No callbacks instantiated.")
            self.callbacks = CallbackList(callbacks_list)
            self.callbacks.set_trainer(self)
            self.logger.info(f"CallbackList setup with {len(self.callbacks.callbacks)} callbacks.")

            # --- Evaluator (Optional) ---
            self.evaluator = None
            eval_cfg_node = exp_cfg.get("eval")
            if eval_cfg_node and self.val_dataloader:
                self.logger.info("Instantiating evaluator...")
                try:
                     # Pass components explicitly
                     self.evaluator = instantiate(
                         eval_cfg_node,
                         model=self.model,
                         val_dataloader=self.val_dataloader,
                         device=self.device,
                         tokenizer=self.tokenizer # Pass tokenizer if evaluator needs it
                     )
                     self.logger.info(f"Instantiated evaluator: {type(self.evaluator).__name__}")
                except Exception as e:
                     self.logger.error(f"Failed to instantiate evaluator: {e}", exc_info=True)
                     self.logger.warning("Proceeding without evaluator.")
            elif not self.val_dataloader:
                 self.logger.info("No validation dataloader, evaluator not instantiated.")
            else:
                 self.logger.info("No evaluator configuration ('eval') found.")

            # --- Checkpoint Manager (Optional) ---
            self.checkpoint_manager = None
            # Look for checkpointing config under experiment first
            checkpointing_cfg_node = exp_cfg.get("checkpointing", cfg.get("checkpointing"))
            if checkpointing_cfg_node:
                 self.logger.info("Instantiating checkpoint manager...")
                 try:
                      # Pass necessary components and potentially the specific checkpointing config node
                      self.checkpoint_manager = instantiate(
                          checkpointing_cfg_node,
                          model=self.model,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          scaler=self.scaler,
                          config=self.config, # Pass validated TrainingConfig
                          experiment_name=self.experiment_name # Pass resolved experiment name
                          # Add other necessary args based on CheckpointManager.__init__
                      )
                      self.logger.info(f"Instantiated checkpoint manager: {type(self.checkpoint_manager).__name__}")
                 except Exception as e:
                      self.logger.error(f"Failed to instantiate checkpoint manager: {e}", exc_info=True)
                      self.logger.warning("Proceeding without checkpoint manager.")
            else:
                self.logger.info("No checkpointing configuration found.")

        except Exception as e:
            self.logger.error(f"Error during component instantiation: {e}", exc_info=True)
            raise

        # 4. Compile Model (Optional)
        if self.compile_model:
            self._compile_model()

        # 5. Handle Checkpoint Resuming (Call helper method)
        # Initialize training state attributes before potentially overwriting from checkpoint
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = None
        self.total_train_time = 0.0
        self._stop_training = False
        
        loaded_global_step = self._handle_resume_from_checkpoint() # Call the new helper

        # 6. Instantiate Evaluator (Optional, requires val_dataloader)
        if self.val_dataloader and cfg.get('evaluation'):
            self.logger.info("Instantiating Evaluator...")
            try:
                self.evaluator = instantiate(
                    cfg.eval,
                    model=self.model,
                    val_dataloader=self.val_dataloader,
                    device=self.device,
                )
                self.logger.info(f"Instantiated Evaluator: {type(self.evaluator).__name__}")
            except Exception as e:
                self.logger.error(f"Failed to instantiate evaluator: {e}", exc_info=True)
                self.logger.warning("Proceeding without evaluator.")

        # 7. Finalize Setup
        self._finalize_setup()

        self.logger.info("Trainer initialization complete.")

    def _handle_resume_from_checkpoint(self) -> Optional[int]:
        """Handles loading state from a checkpoint if specified."""
        self._just_resumed_trigger_eval = False
        self._just_resumed_trigger_save = False
        self._just_resumed_trigger_sample = False
        loaded_global_step = None # Track the step loaded from checkpoint

        if self.resume_from_checkpoint and self.checkpoint_manager:
            self.logger.info(f"Attempting to resume training from checkpoint: {self.resume_from_checkpoint}")
            try:
                # Load state includes model, optimizer, scheduler, scaler, epoch, step, etc.
                # Call load_checkpoint with only the path specifier
                loaded_state = self.checkpoint_manager.load_checkpoint(
                    path_specifier=self.resume_from_checkpoint # <<< FIXED: Pass only path_specifier
                    # REMOVED arguments: model, optimizer, scheduler, scaler, device, callbacks
                )
                if loaded_state:
                    self.logger.info(f"Successfully loaded state from checkpoint at global step {loaded_state.global_step}. Resuming epoch {loaded_state.epoch + 1}.")
                    # Restore training state variables
                    self.epoch = loaded_state.epoch
                    self.global_step = loaded_state.global_step
                    self.best_val_metric = loaded_state.best_val_metric
                    # Store the loaded step for potential dataloader skipping
                    loaded_global_step = self.global_step
                    # Update total train time if available
                    if loaded_state.total_train_time:
                        self.total_train_time = loaded_state.total_train_time
                        self.logger.info(f"Resumed total training time: {format_time(self.total_train_time)}")

                    # Check if loaded config matches current config (basic check)
                    # TODO: Enhance config diff logic if needed
                    if loaded_state.config:
                        try:
                            # Attempt conversion assuming loaded_state.config is dict-like
                            loaded_conf_dict = dict(loaded_state.config)
                            current_conf_dict = OmegaConf.to_container(self.raw_config, resolve=True)
                            # Simple comparison; might need deep diff for complex cases
                            if loaded_conf_dict != current_conf_dict:
                                self.logger.warning("Configuration in checkpoint differs from current configuration. Differences may cause issues.")
                                # Consider adding a more detailed diff log here if necessary
                        except Exception as diff_err:
                            self.logger.warning(f"Could not compare checkpoint config with current config: {diff_err}")
                    
                    # Set flags to trigger actions after resuming, respecting intervals
                    self._just_resumed_trigger_eval = True
                    self._just_resumed_trigger_save = True
                    self._just_resumed_trigger_sample = True

                else:
                    self.logger.warning(f"Checkpoint specified ({self.resume_from_checkpoint}) but loading returned no state. Starting fresh.")

            except CheckpointLoadError as e:
                self.logger.error(f"Failed to load checkpoint: {e}. Training will start from scratch.", exc_info=True)
            except FileNotFoundError:
                self.logger.error(f"Checkpoint file not found: {self.resume_from_checkpoint}. Training will start from scratch.")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred during checkpoint loading: {e}. Training will start from scratch.", exc_info=True)
        else:
            if self.resume_from_checkpoint:
                 self.logger.warning(f"Resume path '{self.resume_from_checkpoint}' provided, but CheckpointManager is not available. Starting training from scratch.")
            else:
                 self.logger.info("No checkpoint specified or CheckpointManager disabled. Starting training from scratch.")
        
        return loaded_global_step

    def _setup_device(self, device_preference: Optional[str] = None) -> torch.device:
        """Sets up the appropriate torch device based on config and availability."""
        self.logger.info(f"Requested device preference: '{device_preference}'")
        if isinstance(device_preference, torch.device):
            self.logger.info(f"Using provided torch.device object: {device_preference}")
            return device_preference
        elif isinstance(device_preference, str):
             # Handle 'auto' explicitly
             if device_preference.lower() == 'auto':
                  selected_device_str = "cuda" if torch.cuda.is_available() else "cpu"
                  self.logger.info(f"'auto' detected, selecting: {selected_device_str}")
                  return torch.device(selected_device_str)
             else:
                  # Try to create device from the provided string (e.g., 'cuda:0', 'cpu')
                  try:
                      device = torch.device(device_preference)
                      self.logger.info(f"Using specified device string: {device}")
                      return device
                  except RuntimeError as e:
                      self.logger.error(f"Invalid device string '{device_preference}': {e}. Falling back to auto-detection.", exc_info=True)
                      # Fallthrough to auto-detection below
        # Auto-detection if preference is None, invalid, or not specified
        selected_device_str = "cuda" if torch.cuda.is_available() else "cpu"
        final_device = torch.device(selected_device_str)
        self.logger.info(f"No valid device specified or fallback required, auto-detected: {final_device}")
        return final_device

    def _compile_model(self) -> None:
         """Compiles the model using torch.compile if requested."""
         if not getattr(self, 'compile_model', False):
             return
         self.logger.info("Compiling the model... (takes a ~minute)")
         try:
             compiled_model: nn.Module = torch.compile(self.model) # type: ignore[assignment]
             self.model = compiled_model
             self.logger.info("Model compiled successfully.")
         except Exception as e:
             self.logger.warning(f"Model compilation failed: {e}. Proceeding without compilation.", exc_info=True)

    def _finalize_setup(self) -> None:
         """Final setup steps after component initialization and potential resume."""
         if self.checkpoint_manager:
              # Ensure checkpoint_dir exists before accessing parent
              ckpt_dir = getattr(self.checkpoint_manager, 'checkpoint_dir', None)
              if ckpt_dir and isinstance(ckpt_dir, Path) and ckpt_dir.parent:
                  run_dir = ckpt_dir.parent
                  # Iterate through callbacks to find the TensorBoardLogger
                  tb_logger = None
                  # Import TensorBoardLogger at the top of the file or within the method if necessary
                  from .callbacks.tensorboard import TensorBoardLogger # Assuming this is the correct import path
                  for callback in self.callbacks: # CallbackList is iterable
                      if isinstance(callback, TensorBoardLogger):
                          tb_logger = callback
                          break

                  if tb_logger:
                      if hasattr(tb_logger, 'set_log_dir_absolute') and callable(getattr(tb_logger, 'set_log_dir_absolute')):
                          tb_logger.set_log_dir_absolute(str(run_dir))
              else:
                    self.logger.warning("Could not determine run directory from CheckpointManager.")
         self.logger.debug("Trainer final setup steps complete.")

    def train(self) -> Dict[str, Any]:
        self.logger.info("Starting training...")
        self.start_time = time.time()
        self._stop_training = False

        # Parameters are now accessed directly from self.config within TrainingLoop
        # Removed extraction lines:
        # use_amp = self.config.use_amp
        # grad_accum_steps = self.config.gradient_accumulation_steps
        # max_grad_norm = self.config.max_grad_norm
        # log_interval = self.config.log_interval
        # save_steps_interval = self.config.save_interval if self.config.save_interval is not None else 0
        # time_save_interval = self.config.time_save_interval_seconds if self.config.time_save_interval_seconds is not None else 0
        # max_steps = self.config.max_steps

        # Instantiate TrainingLoop with simplified arguments
        training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            train_dataloader=self.train_dataloader,
            device=self.device,
            config=self.config, # Pass the Pydantic TrainingConfig
            scheduler=self.scheduler,
            callbacks=self.callbacks.callbacks, # Pass the inner list
            checkpoint_manager=self.checkpoint_manager,
        )

        # Removed initial_state_for_loop creation

        final_state = {}
        try:
            self.callbacks.on_train_begin()

            # Call run without initial state or resume flags
            final_state = training_loop.run()

            self.total_train_time = final_state.get("total_train_time", 0.0)
            # Update trainer's final state from loop results if needed
            self.global_step = final_state.get("global_step", self.global_step)
            self.epoch = final_state.get("epoch", self.epoch)
            self.best_val_metric = final_state.get("best_val_metric", self.best_val_metric)

            self.logger.info("Training finished.")
            final_metrics = final_state.get("metrics", {})
            self.logger.info(f"Final Metrics: {final_metrics}")

            self.callbacks.on_train_end(metrics=final_metrics) # Global train end hook

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user (KeyboardInterrupt).")
            # Optionally save checkpoint on interrupt?
            if self.checkpoint_manager:
                self.logger.info("Attempting to save final state on interrupt...")
                # Gather current state manually or get from training_loop if possible
                # This part needs careful implementation if required
                pass
            final_state["status"] = "interrupted"

        except Exception as e:
            self.logger.error(f"An error occurred during training: {e}", exc_info=True)
            final_state["status"] = "failed"
            self.callbacks.on_train_error(e) # Global error hook
            # Re-raise maybe? Or just return failed state?
            raise

        finally:
            # Ensure logs are flushed etc.
            force_flush_logs()
            if self.start_time is not None:
                 elapsed_time = time.time() - self.start_time
                 self.logger.info(f"Total Training Time (Trainer perspective): {format_time(elapsed_time)}")

        return final_state # Return dict with metrics, final step/epoch, status

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        # TODO: Add other generation parameters (top_p, etc.)
        # Consider accepting a GenerationConfig object?
    ) -> str:
        # ... (generate_text implementation - likely needs minor adjustments) ...
        # Should check self.model and self.tokenizer are available
        self.logger.info(f"Generating text with prompt: '{prompt[:50]}...'")
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not available for text generation.")
        if not hasattr(self.model, 'generate'): # Check if model has generate method
             raise RuntimeError(f"The current model ({type(self.model).__name__}) does not support text generation via a 'generate' method.")

        # Simple wrapper around model.generate or a TextGenerator class
        # Ensure model is in eval mode and on the correct device
        self.model.eval()
        self.model.to(self.device)

        generator = TextGenerator(model=self.model, tokenizer=self.tokenizer, device=self.device)

        try:
            with torch.no_grad():
                # Assuming generator.generate_text takes these params
                # Adapt if the signature is different
                results = generator.generate_text(
                    start_prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    # Pass other params
                )
            generated_text = results[0] if results else "" # Assuming list output
            self.logger.info("Text generation complete.")
            return generated_text
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}", exc_info=True)
            return f"Error during generation: {e}"
        finally:
             # Ensure model is returned to train mode if necessary (though Trainer typically ends after train)
             # self.model.train() # Or manage this state within train() method
             pass

