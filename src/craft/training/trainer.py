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
import contextlib # Added for torch.no_grad()
from tqdm.auto import tqdm # Use auto version for notebook/terminal compatibility
import sys

from .training_loop import TrainingLoop, get_current_lr, get_cuda_memory_stats # Import helpers too
from .evaluation import Evaluator # Ensure Evaluator is imported
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

# --- Import Initialization Helpers ---
from .initialization import (
    initialize_device,
    initialize_tokenizer,
    initialize_model,
    initialize_dataloaders,
    initialize_optimizer,
    initialize_scheduler,
    initialize_amp_scaler,
    initialize_callbacks,
    initialize_checkpoint_manager,
    initialize_evaluator,
    compile_model_if_enabled,
)

# Helper to get a representation of state (e.g., hash of params)
def get_state_hash(component_state_dict: Dict[str, Any]) -> str:
    """Creates a hash of a state dict for quick comparison."""
    try:
        # Convert state dict to a string representation
        state_str = str(component_state_dict)
        return hashlib.md5(state_str.encode()).hexdigest()
    except Exception as e:
        return f"ErrorHashingState: {e}"

class Trainer:
    """
    Main trainer class that coordinates the training loop and evaluation,
    instantiating components from a Hydra configuration object.
    Uses helper functions from `initialization.py` for component setup.
    """

    # Type hints for attributes initialized in __init__
    config: TrainingConfig # Validated Pydantic config
    raw_config: DictConfig # Store the raw hydra config (root)
    experiment_cfg: DictConfig # Convenience access to cfg.experiment node

    device: torch.device
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader]
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    tokenizer: Optional[Tokenizer]
    callbacks: CallbackList # Manages the list of callbacks
    evaluator: Optional[Evaluator]
    checkpoint_manager: Optional[CheckpointManager]
    scaler: Optional[torch.cuda.amp.GradScaler]

    resume_from_checkpoint: Optional[str] # Determined from CLI or config
    compile_model: bool # Determined from CLI or config
    experiment_name: Optional[str] # From experiment config

    # Internal state attributes
    epoch: int
    global_step: int
    start_time: Optional[float]
    total_train_time: float
    best_val_metric: Optional[float] # Use Optional[float]
    _just_resumed_trigger_eval: bool
    _stop_training: bool

    def __init__(
        self,
        cfg: DictConfig, # Main Hydra configuration object (root)
        resume_from_checkpoint: Optional[str] = None, # Allow override from CLI
        compile_model: Optional[bool] = None, # Allow override from CLI
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trainer...")
        self.raw_config = cfg # Store raw config

        # --- Configuration Handling ---
        if not cfg.get('experiment'):
            raise ValueError("Configuration is missing the required 'experiment' block.")
        self.experiment_cfg = cfg.experiment # Shortcut to experiment config node

        # 1. Validate Core Training Config (Pydantic)
        try:
            training_cfg_node = self.experiment_cfg.get('training')
            if not training_cfg_node:
                raise ValueError("Experiment configuration is missing the 'training' block.")
            training_dict = OmegaConf.to_container(training_cfg_node, resolve=True, throw_on_missing=True) # Ensure it resolves fully
            if not isinstance(training_dict, dict):
                 raise TypeError(f"Resolved training config is not a dict, got {type(training_dict)}")
            self.config = TrainingConfig(**training_dict) # Validated training config
            self.logger.info("TrainingConfig successfully parsed and validated.")
        except ValidationError as e:
             self.logger.error(f"Pydantic validation failed for TrainingConfig: {e}", exc_info=True)
             # Log the failing dict for debugging if possible
             try:
                 failed_dict = OmegaConf.to_container(self.experiment_cfg.training, resolve=True)
                 self.logger.debug(f"Training config dict that failed validation:\\n{failed_dict}")
             except Exception as log_e:
                 self.logger.debug(f"Could not log failing dict: {log_e}")
             raise
        except Exception as e:
             self.logger.error(f"Failed to resolve or validate TrainingConfig: {e}", exc_info=True)
             raise

        # FIXED: Use __init__ args directly, config schema doesn't have these
        self.resume_from_checkpoint = resume_from_checkpoint
        # FIXED: Default compile_model based on config if not provided via CLI
        # self.compile_model = compile_model if compile_model is not None else self.config.compile_model
        # FIXED AGAIN: Use the init argument directly
        self.compile_model = compile_model

        # Extract experiment name
        # self.experiment_name = self.config.experiment_name or "default_experiment"
        # FIXED: Get experiment name from the experiment config node, not the parsed TrainingConfig
        self.experiment_name = self.experiment_cfg.get("name", "default_experiment")
        self.logger.info(f"Experiment Name: {self.experiment_name}")

        # --- Initialize State Variables (Before potential resume) ---
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = None
        self.start_time = None
        self.total_train_time = 0.0
        self._stop_training = False
        self._just_resumed_trigger_eval = False

        # --- Component Initialization (using helpers) ---
        try:
            # Get config nodes needed for initialization helpers
            data_cfg_node = self.experiment_cfg.get("data")
            model_cfg_node = self.experiment_cfg.get("model")
            optimizer_cfg_node = self.experiment_cfg.get("optimizer")
            scheduler_cfg_node = self.experiment_cfg.get("scheduler")
            callbacks_cfg_node = self.experiment_cfg.get("callbacks")
            checkpoint_cfg_node = self.experiment_cfg.get("checkpointing")
            eval_cfg_node = self.experiment_cfg.get("evaluation")
            compile_options_cfg = self.raw_config.get("torch_compile_options") # From root config

            # Device
            self.device = initialize_device(self.experiment_cfg.get("device", "cpu"))

            # Tokenizer (optional)
            self.tokenizer = initialize_tokenizer(data_cfg_node)

            # Model
            if not model_cfg_node: raise ValueError("cfg.experiment.model is required.")
            self.model = initialize_model(model_cfg_node, self.device, self.tokenizer)

            # Dataloaders
            if not data_cfg_node: raise ValueError("cfg.experiment.data is required for dataloaders.")
            self.train_dataloader, self.val_dataloader = initialize_dataloaders(
                data_cfg_node, self.device, self.tokenizer
            )

            # Optimizer
            if not optimizer_cfg_node: raise ValueError("cfg.experiment.optimizer is required.")
            self.optimizer = initialize_optimizer(optimizer_cfg_node, self.model)

            # Scheduler (optional)
            self.scheduler = initialize_scheduler(scheduler_cfg_node, self.optimizer)

            # AMP Scaler
            self.scaler = initialize_amp_scaler(self.config.use_amp, self.device)

            # Callbacks (get raw list first)
            # Pass trainer=self if callbacks need it during their __init__
            raw_callbacks = initialize_callbacks(callbacks_cfg_node)
            self.callbacks = CallbackList(raw_callbacks)
            self.callbacks.set_trainer(self) # Set trainer instance on callbacks AFTER list is created
            self.logger.info(f"Initialized CallbackList with {len(self.callbacks.callbacks)} callbacks.")

            # Checkpoint Manager (optional)
            self.checkpoint_manager = initialize_checkpoint_manager(
                checkpoint_cfg_node=checkpoint_cfg_node,
                full_app_config=self.raw_config, # Pass root config
                experiment_name=self.experiment_name,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=self.scheduler,
                callbacks=self.callbacks, # Pass CallbackList instance
                tokenizer=self.tokenizer,
            )

            # Evaluator (optional, requires val_dataloader)
            self.evaluator = initialize_evaluator(
                eval_cfg_node=eval_cfg_node,
                model=self.model,
                val_dataloader=self.val_dataloader,
                device=self.device,
                use_amp=self.config.use_amp,
                callbacks=self.callbacks # Pass CallbackList instance
            )

        except Exception as e:
            self.logger.exception(f"Error during component initialization: {e}")
            # Consider more specific error handling or re-raising
            raise

        # --- Resume from Checkpoint (if specified) ---
        # Needs to happen *after* all components are instantiated but *before* compilation
        loaded_global_step = 0 # Track step loaded from checkpoint for potential use
        if self.resume_from_checkpoint:
            loaded_state = self._resume_from_checkpoint(self.resume_from_checkpoint)
            if loaded_state:
                 loaded_global_step = loaded_state.global_step
        else:
             self.logger.info("Starting training from scratch (no checkpoint specified).")


        # --- Compile Model (Optional, *after* resume) ---
        # The model instance might have been replaced by checkpoint loading, so compile here.
        self.model = compile_model_if_enabled(
             self.model,
             self.compile_model,
             compile_options_cfg
        )

        # --- Setup Training Loop Object ---
        self.training_loop = TrainingLoop(
            model=self.model, # Pass potentially compiled model
            optimizer=self.optimizer,
            train_dataloader=self.train_dataloader,
            device=self.device,
            config=self.config, # Pass validated TrainingConfig
            scheduler=self.scheduler,
            callbacks=self.callbacks, # Pass initialized CallbackList
            checkpoint_manager=self.checkpoint_manager,
        )

        self.logger.info("Trainer initialization complete.")

    def _resume_from_checkpoint(self, path_specifier: str) -> Optional[TrainingState]:
        """
        Loads training state from a checkpoint.
        Returns the loaded TrainingState object if successful, otherwise None.
        Updates self.epoch, self.global_step, self.best_val_metric, and component states.
        """
        self.logger.info(f"Attempting to resume training from checkpoint: {path_specifier}")
        if not self.checkpoint_manager:
            self.logger.error("Cannot resume: CheckpointManager was not initialized.")
            return None

        try:
            # Load the state object using CheckpointManager
            loaded_state = self.checkpoint_manager.load_checkpoint(path_specifier)
            if not loaded_state:
                # load_checkpoint should log internally if file not found or load fails
                self.logger.warning(f"Checkpoint load returned None for specifier '{path_specifier}'. Starting training from scratch.")
                self._reset_training_state() # Ensure state is reset
                return None

            self.logger.info(f"Successfully loaded checkpoint metadata: Step {loaded_state.global_step}, Epoch {loaded_state.epoch}")

            # --- Load State into Components ---
            # Model state is handled by CheckpointManager.load_checkpoint internally now

            # Optimizer state is handled by CheckpointManager.load_checkpoint internally now

            # Scheduler state is handled by CheckpointManager.load_checkpoint internally now

            # AMP Scaler state is handled by CheckpointManager.load_checkpoint internally now

            # --- Load Training Progress ---
            self.epoch = loaded_state.epoch
            self.global_step = loaded_state.global_step
            self.best_val_metric = loaded_state.best_val_metric # Load best metric
            self.logger.info(f"Resuming from Epoch: {self.epoch}, Global Step: {self.global_step}, Best Val Metric: {self.best_val_metric}")

            # --- Load Callback State ---
            # Callbacks state is handled by CheckpointManager.load_checkpoint internally now

            # Trigger evaluation after resuming if not resuming at step 0 and evaluator exists
            self._just_resumed_trigger_eval = self.global_step > 0 and self.evaluator is not None
            self.logger.info(f"Training resumed successfully from step {self.global_step}.")
            return loaded_state # Return the loaded state object

        except CheckpointLoadError as e:
            self.logger.error(f"CheckpointLoadError while resuming from '{path_specifier}': {e}. Starting training from scratch.", exc_info=True)
            self._reset_training_state()
            return None
        except FileNotFoundError as e:
             self.logger.error(f"Checkpoint file not found for specifier '{path_specifier}': {e}. Starting training from scratch.", exc_info=True)
             self._reset_training_state()
             return None
        except Exception as e:
            # Catch any other unexpected errors during the resume process
            self.logger.exception(f"An unexpected error occurred during checkpoint loading/state restoration: {e}. Starting training from scratch.")
            self._reset_training_state()
            return None

    def _reset_training_state(self):
        """Resets internal state variables for starting from scratch."""
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = None
        # Reset optimizer/scheduler/scaler states?
        # This is tricky. If resume fails, we might want to start fresh.
        # However, components are already initialized. Restarting their state
        # might require re-initialization which is complex here.
        # Assumption: If resume fails, we proceed with the initially configured
        # component states (e.g., fresh optimizer).
        self.logger.warning("Resetting trainer epoch, global_step, and best_val_metric to initial values.")

    def _should_run_eval(self, current_global_step: int, prev_global_step: int) -> bool:
        """Determines if evaluation should run based on step interval or resume trigger."""
        if not self.evaluator or not self.config.eval_interval or self.config.eval_interval <= 0:
            return False

        # Add type check for robustness in testing with mocks
        if not isinstance(current_global_step, int) or not isinstance(prev_global_step, int):
            self.logger.warning(f"_should_run_eval received non-integer steps: current={type(current_global_step)}, prev={type(prev_global_step)}. Skipping eval check.")
            return False

        # Check if the interval boundary was crossed since the last check
        # Handle edge case: prev_global_step might be 0 at the start.
        start_eval_count = (prev_global_step -1) // self.config.eval_interval if prev_global_step > 0 else -1
        end_eval_count = (current_global_step - 1) // self.config.eval_interval if current_global_step > 0 else -1
        crossed_boundary = end_eval_count > start_eval_count

        # Also run if resuming and haven't evaluated yet for this state
        run_after_resume = self._just_resumed_trigger_eval

        if run_after_resume:
            self.logger.debug("Evaluation triggered immediately after resuming.")
            self._just_resumed_trigger_eval = False # Reset trigger after checking

        if crossed_boundary:
             self.logger.debug(f"Evaluation triggered by interval: Step {current_global_step} crossed boundary for interval {self.config.eval_interval}.")

        return crossed_boundary or run_after_resume

    def train(self) -> Dict[str, Any]:
        """Main training entry point."""
        self.logger.info(f"Starting training run: {self.experiment_name}")
        self.start_time = time.time()
        start_epoch = self.epoch
        start_step = self.global_step

        # --- Set Seed --- #
        seed = self.raw_config.get("seed")
        if seed is not None:
             set_seed(seed)
             self.logger.info(f"Set random seed to: {seed}")

        # --- Callbacks: Train Begin --- #
        self.callbacks.on_train_begin(global_step=self.global_step)

        # --- Delegate to TrainingLoop --- #
        training_result = {}
        try:
            self.logger.info(f"Calling TrainingLoop.run (start_epoch={start_epoch}, start_step={start_step})...")
            training_result = self.training_loop.run(
                start_epoch=start_epoch,
                start_step=start_step
            )
            self.logger.info("TrainingLoop.run finished.")

            # --- Update Trainer State from Result --- #
            self.epoch = training_result.get("epoch", self.epoch)
            self.global_step = training_result.get("final_global_step", self.global_step)
            # Potentially update best_val_metric if TrainingLoop returns it
            if "best_val_metric" in training_result:
                 self.best_val_metric = training_result["best_val_metric"]

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user (KeyboardInterrupt). Performing cleanup...")
            self._stop_training = True # Set flag for callbacks
            # TrainingLoop.run should handle its own cleanup, Trainer just logs
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during TrainingLoop.run: {e}")
            # Optionally re-raise or set error state
            raise # Re-raise for now
        finally:
            # --- Cleanup and Final Callbacks ---
            self.total_train_time = time.time() - self.start_time if self.start_time else 0
            self.logger.info(f"Training run finished. Total training time: {format_time(self.total_train_time)}")

            # Update final metrics dict for on_train_end
            final_metrics = {
                 "final_step": self.global_step,
                 "final_epoch": self.epoch,
                 "best_val_metric": self.best_val_metric,
                 "total_train_time_seconds": self.total_train_time,
            }
            # Include metrics returned by training_loop.run if available
            if training_result:
                 final_metrics.update({f"loop_{k}":v for k,v in training_result.items()}) # Prefix loop metrics

            # Pass exception info if one occurred
            exception_info = sys.exc_info()[1] if 'e' in locals() else None

            # Add exception to metrics if it exists
            if exception_info:
                final_metrics["exception"] = str(exception_info)

            self.callbacks.on_train_end(metrics=final_metrics)
            force_flush_logs() # Ensure logs are written

        # Return final state/metrics accumulated by Trainer
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_metric": self.best_val_metric,
            "total_train_time": self.total_train_time,
            # Optionally include raw training_result from loop
            "loop_result": training_result
        }

def _is_jupyter(): # Simple check if running in a notebook
    """Checks if the code is running in a Jupyter-like environment (notebook, qtconsole)."""
    try:
        # Using get_ipython is a common way, though fragile. Check class name.
        shell = get_ipython().__class__.__name__ # type: ignore
        if 'zmqshell' in shell.lower(): # Covers ZMQInteractiveShell in notebook/qtconsole
            return True
        # elif shell == 'TerminalInteractiveShell': # Terminal IPython isn't Jupyter UI
        #     return False
        else: # Other shells or non-IPython environments
            return False
    except NameError:
        return False      # Standard Python interpreter
    except Exception:
        return False      # Any other issue, assume not Jupyter
