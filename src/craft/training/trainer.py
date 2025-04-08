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
from hydra.utils import get_original_cwd
import hydra

from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .checkpointing import CheckpointManager, TrainingState, CheckpointLoadError
from .callbacks import CallbackList, TensorBoardLogger
from .callbacks.base import Callback
from .callbacks.sample_generation import SampleGenerationCallback
from .generation import TextGenerator
from .progress import ProgressTracker
from ..data.tokenizers.base import Tokenizer
from ..config.schemas import TrainingConfig, DataConfig, AnyModelConfig
from ..utils.logging import setup_logging, force_flush_logs, format_time

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
    using pre-instantiated components.
    """

    # Add type hints for attributes initialized in __init__
    config: TrainingConfig
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
    resume_from_checkpoint: Optional[str]
    # compile_model: bool # Already implicitly typed by init default
    # experiment_name: Optional[str] # Already implicitly typed by init default

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
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        # --- Optional Components ---
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[Tokenizer] = None,
        callbacks: Optional[List[Callback]] = None,
        evaluator: Optional[Evaluator] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: Optional[Union[str, torch.device]] = None,
        resume_from_checkpoint: Optional[str] = None,
        compile_model: bool = False,
        experiment_name: Optional[str] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trainer with pre-instantiated components...")

        # 1. Validate and store core config
        self.config = self._validate_training_config(config) # Keep validation

        # 2. Assign pre-instantiated components
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.scaler = scaler # Assign pre-created scaler

        # 3. Setup Device
        self.device = self._setup_device(device)
        self.model.to(self.device)
        # TODO: Consider moving optimizer state to device if needed during resume

        # 4. Setup Callbacks
        self.callbacks = self._setup_callbacks(callbacks) # Wrap in CallbackList

        # 5. Setup Evaluator (Assign pre-built one)
        self.evaluator = evaluator # Directly assign
        if self.evaluator:
             self.logger.info(f"Using provided Evaluator: {type(self.evaluator).__name__}")
        elif self.val_dataloader:
             self.logger.warning("Validation dataloader provided but no Evaluator instance was passed to Trainer.")
             # Decide: Create a default one here, or rely on user passing it?
             # For stricter separation, rely on user passing it.

        # 6. Setup Checkpointing (Assign pre-built one)
        self.checkpoint_manager = checkpoint_manager # Directly assign
        if self.checkpoint_manager:
             self.logger.info(f"Using provided CheckpointManager: {type(self.checkpoint_manager).__name__}")
        else:
             self.logger.warning("No CheckpointManager provided. Checkpoint saving/loading will be disabled.")
             # Disable resume if no manager
             if resume_from_checkpoint:
                 self.logger.warning("`resume_from_checkpoint` was specified, but no CheckpointManager provided. Resuming is disabled.")
                 resume_from_checkpoint = None # Override

        # 7. Compile Model (Optional)
        if compile_model:
             self._compile_model() # Keep internal compilation logic

        # 8. Initialize Training State Variables
        self._initialize_training_state()

        # 9. Handle Resuming (Needs CheckpointManager)
        self.resume_from_checkpoint = resume_from_checkpoint # Store original request
        self._resume_if_needed() # Logic now relies on self.checkpoint_manager

        # 10. Finalize Setup (Potentially linking callbacks to trainer state)
        self._finalize_setup()

        self.logger.info("Trainer initialization complete.")

    def _validate_training_config(self, config: Any) -> TrainingConfig:
        """Validate the main training configuration."""
        if isinstance(config, TrainingConfig):
            return config
        else:
             raise TypeError(f"Expected config to be TrainingConfig instance, got {type(config)}")

    def _setup_device(self, device_option: Optional[Union[str, torch.device]]) -> torch.device:
        """Determines and sets up the torch device."""
        if isinstance(device_option, torch.device):
            self.logger.info(f"Using provided device: {device_option}")
            return device_option
        elif isinstance(device_option, str):
             self.logger.info(f"Setting device based on string: '{device_option}'")
             return torch.device(device_option)
        else:
             # Auto-detection
             selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             self.logger.info(f"Auto-detected device: {selected_device}")
             return selected_device

    def _setup_callbacks(self, callbacks_list: Optional[List[Callback]]) -> CallbackList:
        """Sets up the CallbackList, ensuring essential callbacks if needed."""
        if callbacks_list is None:
             callbacks_list = []
        # Example: Could potentially add default progress logger if none provided
        # has_progress_logger = any(isinstance(cb, ProgressLogger) for cb in callbacks_list)
        # if not has_progress_logger:
        #    callbacks_list.append(ProgressLogger())
        self.logger.info(f"Setting up CallbackList with {len(callbacks_list)} provided callbacks.")
        # Pass trainer instance to callbacks AFTER basic init is done
        callback_handler = CallbackList(callbacks_list, trainer=self)
        return callback_handler

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

    def _initialize_training_state(self) -> None:
        """Initializes variables tracking training progress."""
        self.epoch = 0
        self.global_step = 0
        self.start_time = None
        self.total_train_time = 0.0
        self.best_val_metric = None
        self._just_resumed_trigger_eval = False
        self._just_resumed_trigger_save = False
        self._just_resumed_trigger_sample = False
        self._stop_training = False

    def _resume_if_needed(self) -> None:
         """Handles loading state if resume_from_checkpoint is set."""
         if self.resume_from_checkpoint and self.checkpoint_manager:
             self.logger.info(f"Attempting to resume training from: {self.resume_from_checkpoint}")
             try:
                 # Load ONLY the state object from the checkpoint manager
                 loaded_state: Optional[TrainingState] = self.checkpoint_manager.load_checkpoint(
                     path_specifier=self.resume_from_checkpoint
                     # No component arguments passed here anymore
                 )
                 
                 if loaded_state:
                     self.logger.info(f"Successfully loaded TrainingState from checkpoint (Step: {loaded_state.global_step}). Applying state...")
                     
                     # --- Apply state to components --- #
                     # Model state
                     if loaded_state.model_state_dict:
                         # Handle potential DataParallel/DDP wrapping
                         state_dict = loaded_state.model_state_dict
                         if any(key.startswith('module.') for key in state_dict.keys()):
                             self.logger.info("Detected 'module.' prefix in checkpoint state_dict, removing before loading.")
                             state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                         self.model.load_state_dict(state_dict)
                         self.logger.info("Applied model state.")
                     else:
                          # This should ideally be caught by CheckpointManager load validation
                          raise CheckpointLoadError("Loaded state is missing 'model_state_dict'.")
                     
                     # Optimizer state
                     if self.optimizer and loaded_state.optimizer_state_dict:
                         self.optimizer.load_state_dict(loaded_state.optimizer_state_dict)
                         self.logger.info("Applied optimizer state.")
                         # TODO: Move optimizer state to self.device if needed
                         # for state in self.optimizer.state.values():
                         #     for k, v in state.items():
                         #         if isinstance(v, torch.Tensor):
                         #             state[k] = v.to(self.device)
                     elif self.optimizer:
                         self.logger.warning("Optimizer exists but no optimizer state found in checkpoint.")
                         # Consider if this should be an error depending on use case
                     
                     # Scheduler state
                     if self.scheduler and loaded_state.scheduler_state_dict:
                         self.scheduler.load_state_dict(loaded_state.scheduler_state_dict)
                         self.logger.info("Applied scheduler state.")
                     elif self.scheduler:
                         self.logger.warning("Scheduler exists but no scheduler state found in checkpoint.")
                         # Consider if this should be an error

                     # Scaler state
                     if self.scaler and loaded_state.scaler_state_dict:
                         if hasattr(self.scaler, 'load_state_dict'):
                             self.scaler.load_state_dict(loaded_state.scaler_state_dict)
                             self.logger.info("Applied GradScaler state.")
                         else:
                             self.logger.warning("GradScaler object does not have load_state_dict method.")
                     elif self.scaler:
                         self.logger.warning("GradScaler exists but no scaler state found in checkpoint.")

                     # Callbacks state (if CallbackList supports it)
                     if self.callbacks and loaded_state.callbacks_state:
                          if hasattr(self.callbacks, 'load_state_dict') and callable(getattr(self.callbacks, 'load_state_dict')):
                              self.callbacks.load_state_dict(loaded_state.callbacks_state)
                              self.logger.info("Applied callbacks state via CallbackList.load_state_dict.")
                          else:
                              self.logger.warning("Callbacks state found in checkpoint, but CallbackList does not support load_state_dict.")
                     elif self.callbacks and hasattr(self.callbacks, 'load_state_dict'):
                          self.logger.warning("No callbacks state found in checkpoint to load.")
                     # ------------------------------------ #

                     # Update trainer state variables from loaded state
                     self.global_step = loaded_state.global_step
                     self.epoch = loaded_state.epoch # Resume from the END of the completed epoch
                     self.best_val_metric = loaded_state.best_val_metric

                     # Set flags to skip immediate actions after resuming
                     # (Maybe adjust this strategy later if needed)
                     self._just_resumed_trigger_eval = True
                     self._just_resumed_trigger_save = True
                     self._just_resumed_trigger_sample = True

                     self.logger.info(f"Successfully resumed trainer state to Step={self.global_step}, Epoch={self.epoch}, BestMetric={self.best_val_metric}")

                 else:
                     # load_checkpoint returned None
                     self.logger.warning(f"Checkpoint loading via CheckpointManager for '{self.resume_from_checkpoint}' failed or returned no state. Starting fresh.")
                     self.resume_from_checkpoint = None # Clear flag

             except CheckpointLoadError as e:
                 # Error during loading or state application
                 self.logger.error(f"Checkpoint loading or state application failed: {e}. Starting training from scratch.", exc_info=True)
                 self.resume_from_checkpoint = None # Clear flag
             except Exception as e:
                 self.logger.error(f"An unexpected error occurred during checkpoint resume: {e}. Starting training from scratch.", exc_info=True)
                 self.resume_from_checkpoint = None # Clear flag

         elif self.resume_from_checkpoint and not self.checkpoint_manager:
              self.logger.error("Resume requested but CheckpointManager is not available. Cannot resume.")
         else:
             self.logger.info("Starting training from scratch (no resume specified or possible).")

    def _finalize_setup(self) -> None:
         """Final setup steps after component initialization and potential resume."""
         if self.checkpoint_manager:
              # Ensure checkpoint_dir exists before accessing parent
              ckpt_dir = getattr(self.checkpoint_manager, 'checkpoint_dir', None)
              if ckpt_dir and isinstance(ckpt_dir, Path) and ckpt_dir.parent:
                  run_dir = ckpt_dir.parent
                  tb_logger = self.callbacks.get_callback(TensorBoardLogger)
                  if tb_logger:
                      if hasattr(tb_logger, 'set_log_dir_absolute') and callable(getattr(tb_logger, 'set_log_dir_absolute')):
                          tb_logger.set_log_dir_absolute(str(run_dir))
                      # Removed fallback to set_log_dir as it might not be intended for absolute paths
                      else:
                          self.logger.warning("TensorBoardLogger does not have set_log_dir_absolute method. Cannot update log path after resume.")
              else:
                    self.logger.warning("Could not determine run directory from CheckpointManager.")
         self.logger.debug("Trainer final setup steps complete.")

    def train(self) -> Dict[str, Any]:
        self.logger.info("Starting training...")
        self.start_time = time.time()
        self._stop_training = False

        # Extract parameters from self.config for TrainingLoop
        # Using default values from TrainingConfig schema if available
        use_amp = self.config.use_amp
        grad_accum_steps = self.config.gradient_accumulation_steps
        max_grad_norm = self.config.max_grad_norm
        log_interval = self.config.log_interval
        # Provide default 0 if Optional[int] is None
        save_steps_interval = self.config.save_steps_interval if self.config.save_steps_interval is not None else 0
        time_save_interval = self.config.time_save_interval_seconds if self.config.time_save_interval_seconds is not None else 0
        max_steps = self.config.max_steps

        # Instantiate TrainingLoop with corrected arguments
        training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            train_dataloader=self.train_dataloader,
            device=self.device,
            config=self.config,
            experiment_config=None,
            scheduler=self.scheduler,
            use_amp=use_amp,
            gradient_accumulation_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            log_interval=log_interval,
            callbacks=self.callbacks.callbacks, # Pass the inner list
            checkpoint_manager=self.checkpoint_manager,
            save_steps_interval=save_steps_interval,
            time_save_interval_seconds=time_save_interval,
            max_steps=max_steps
        )

        # Removed initial_state_for_loop creation
        # Removed run_kwargs dictionary

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

