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
from datetime import datetime, timedelta
from typing import cast

from .training_loop import TrainingLoop, get_current_lr, get_cuda_memory_stats # Import helpers too
from .evaluation import Evaluator # Ensure Evaluator is imported
from .checkpointing import CheckpointManager, TrainingState, CheckpointLoadError
from .callbacks import CallbackList, TensorBoardLogger
from .callbacks.base import Callback
from .callbacks.sample_generation import SampleGenerationCallback
from .generation import TextGenerator
from .progress import ProgressTracker
from ..data.tokenizers.base import Tokenizer
from ..config.schemas import TrainingConfig, DataConfig, AnyModelConfig, ExperimentConfig, LanguageModelConfig, CheckpointingConfig
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
    scaler: Optional[torch.amp.GradScaler]

    resume_from_checkpoint: Optional[str] # Determined from CLI or config
    compile_model: Optional[bool] # Determined from CLI or config
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
            # Cast dict before passing to Pydantic
            self.config = TrainingConfig(**cast(Dict[str, Any], training_dict))
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

        # Use __init__ args directly
        self.resume_from_checkpoint = resume_from_checkpoint
        # Handle Optional[bool] assignment
        self.compile_model = compile_model # Keep as Optional[bool] consistent with __init__ arg

        # Extract experiment name from experiment config node
        self.experiment_name = self.experiment_cfg.get("experiment_name", "default_experiment")
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
            data_cfg_node       = self.experiment_cfg.get("data")
            model_cfg_node      = self.experiment_cfg.get("model")
            optimizer_cfg_node  = self.experiment_cfg.get("optimizer")
            scheduler_cfg_node  = self.experiment_cfg.get("scheduler")
            callbacks_cfg_node  = self.experiment_cfg.get("callbacks")
            checkpoint_cfg_node: Optional[DictConfig] = self.experiment_cfg.get("checkpointing")
            eval_cfg_node       = self.experiment_cfg.get("evaluation")
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

            # AMP Scaler - Ensure type consistency
            # The function returns torch.amp.GradScaler, matches attribute type hint
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
                # Pass experiment name safely, providing a default if None
                experiment_name=self.experiment_name or "default_experiment",
                model=self.model,
                optimizer=self.optimizer,
                # Pass scaler (Optional[torch.amp.GradScaler])
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

            self._just_resumed_trigger_eval = False

            # --- Initialize Progress Tracker --- #
            self.progress = ProgressTracker(
                total_steps=self.config.max_steps, # Use max_steps from validated config
                log_interval=self.config.log_interval
                # Add other relevant params if ProgressTracker needs them
            )

        except Exception as e:
            self.logger.exception(f"Error during component initialization: {e}")
            # Consider more specific error handling or re-raising
            raise

        # --- Resume from Checkpoint (if specified) --- #
        # Needs to happen *after* all components are instantiated but *before* compilation
        loaded_global_step = 0 # Track step loaded from checkpoint for potential use
        if self.resume_from_checkpoint:
            loaded_state = self._resume_from_checkpoint(self.resume_from_checkpoint)
            if loaded_state:
                 loaded_global_step = loaded_state.global_step
                 # Update progress tracker if resuming
                 self.progress.update_step(loaded_global_step)
                 self._just_resumed_trigger_eval = True # Trigger eval after resuming if needed
                 self.logger.info(f"Training resumed successfully from step {loaded_global_step}.")
        else:
             self.logger.info("Starting training from scratch (no checkpoint specified).")

        # --- Compile Model (Optional, *after* potential resume) ---
        # Compile here after the model state might have been updated by checkpoint loading.
        self.model = compile_model_if_enabled(
             self.model,
             # Pass Optional[bool], defaulting to False if None
             self.compile_model or False,
             compile_options_cfg
        )

        # --- Setup Training Loop Object --- #
        self.training_loop = TrainingLoop(
            model=self.model, # Pass potentially compiled model
            optimizer=self.optimizer,
            train_dataloader=self.train_dataloader,
            device=self.device,
            config=self.config, # Pass validated TrainingConfig
            scheduler=self.scheduler,
            # Pass the inner list of callbacks
            callbacks=self.callbacks.callbacks, # Pass List[Callback]
            checkpoint_manager=self.checkpoint_manager,
        )

        self.logger.info("Trainer initialization complete.")

    def _prepare_training_state(self, epoch: int, global_step: int) -> TrainingState:
        """Gathers component states and creates a TrainingState object."""
        # Get state dicts safely, handling None components
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict() if self.optimizer else None
        scheduler_state = self.scheduler.state_dict() if self.scheduler else None
        scaler_state = self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None
        callbacks_state = self.callbacks.state_dict() if self.callbacks else None

        # Get TB log dir (handle potential absence of callback)
        tb_log_dir = None
        if self.callbacks:
            tb_logger = self.callbacks.get_callback(TensorBoardLogger)
            if tb_logger and hasattr(tb_logger, 'resolved_log_dir'):
                tb_log_dir = tb_logger.resolved_log_dir

        # Get serializable config (using the validated TrainingConfig for now)
        serializable_config = self.config.model_dump() if self.config else None

        state = TrainingState(
            epoch=epoch,
            global_step=global_step,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            scheduler_state_dict=scheduler_state,
            scaler_state_dict=scaler_state,
            best_val_metric=self.best_val_metric, # Include current best metric
            config=serializable_config,
            tensorboard_log_dir=tb_log_dir,
            callbacks_state=callbacks_state,
            # Include last eval metrics if available?
            metrics=getattr(self, 'current_val_metrics', None), # Save last known eval metrics
            # Tokenizer path is handled by CheckpointManager during save now
        )
        return state

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

    def _reset_training_state(self) -> None:
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
        """Main training loop coordination."""
        self.logger.info("Starting training run...")
        self.start_time = time.time()
        start_epoch = self.epoch
        start_step = self.global_step

        # Use checkpointing config for metric name and mode (needed for initial eval best check)
        # Safely access checkpointing config and its fields
        chkpt_cfg: Optional[CheckpointingConfig] = None
        if self.experiment_cfg and isinstance(self.experiment_cfg, DictConfig):
             chkpt_node = self.experiment_cfg.get("checkpointing")
             if chkpt_node:
                  # Assuming CheckpointingConfig is the Pydantic model
                  try:
                      chkpt_cfg = CheckpointingConfig(**OmegaConf.to_container(chkpt_node, resolve=True)) # type: ignore
                  except Exception:
                       self.logger.warning("Could not parse checkpointing config node for initial eval.")

        # Access attributes directly
        val_metric_name = chkpt_cfg.val_metric if chkpt_cfg else "val_loss"
        mode = chkpt_cfg.mode if chkpt_cfg else "min"
        save_best_only = chkpt_cfg.save_best_only if chkpt_cfg else False

        # --- Initial Evaluation After Resume --- #
        # Get validation config early
        # Check evaluator exists before trying to run eval
        if self.evaluator and self._just_resumed_trigger_eval:
            self.logger.info(f"Running initial evaluation after resume at global step {self.global_step}...")
            self.callbacks.on_validation_begin(epoch=self.epoch, global_step=self.global_step) # Use current epoch
            val_metrics = self.evaluator.evaluate()
            self.last_eval_time = time.time() # Update last eval time
            self._just_resumed_trigger_eval = False # Reset trigger immediately
            if val_metrics:
                self.logger.info(f"Initial validation metrics: {val_metrics}")
                # Log validation metrics via callbacks
                self.callbacks.on_validation_end(
                    epoch=self.epoch, # Use current epoch
                    global_step=self.global_step,
                    metrics=val_metrics
                )
                # Keep track of the latest validation metrics
                self.current_val_metrics = val_metrics.copy()
                # --- Checkpointing (Best) - After Initial Resume Validation --- #
                if self.checkpoint_manager:
                    is_best = False
                    current_metric_val = None
                    # Use checkpointing config for metric name and mode
                    chkpt_cfg = self.experiment_cfg.get("checkpointing")
                    val_metric_name = chkpt_cfg.val_metric if chkpt_cfg else "val_loss"
                    mode = chkpt_cfg.mode if chkpt_cfg else "min"

                    if val_metric_name and val_metrics:
                        current_metric_val = val_metrics.get(val_metric_name)
                        # Initialize best_val_metric if it's None
                        if self.best_val_metric is None:
                            self.best_val_metric = float('inf') if mode == 'min' else float('-inf')

                        # Check if current metric is better based on mode
                        if current_metric_val is not None and self.best_val_metric is not None:
                            if mode == 'min' and current_metric_val < self.best_val_metric:
                                self.best_val_metric = current_metric_val
                                is_best = True
                            elif mode == 'max' and current_metric_val > self.best_val_metric:
                                self.best_val_metric = current_metric_val
                                is_best = True

                    if is_best:
                        self.logger.info(f"New best validation metric (post-resume): {val_metric_name} = {self.best_val_metric}")
                        # Prepare state and filename for saving
                        state_to_save = self._prepare_training_state(epoch=self.epoch, global_step=self.global_step)
                        # Update metrics in state just before saving
                        state_to_save.metrics = val_metrics
                        state_to_save.best_val_metric = self.best_val_metric # Ensure latest best is saved

                        filename = f"{self.checkpoint_manager.checkpoint_prefix}_epoch_{state_to_save.epoch}_step_{state_to_save.global_step}_best.pt"

                        self.checkpoint_manager.save_checkpoint(
                            state=state_to_save,
                            filename=filename,
                            metrics=val_metrics, # Pass metrics for logging/potential internal use
                            is_best=True # Explicitly True here
                        )
            else:
                self.logger.warning("Initial evaluation after resume finished but returned no metrics.")

        # --- Delegate to TrainingLoop --- #
        training_result: Dict[str, Any] = {}
        exception_object = None # Initialize exception tracking variable
        self._stop_training = False # Ensure stop flag is reset

        try:
            # Check if num_epochs is set before using in range
            if self.config.num_epochs is None:
                if self.config.max_steps is None:
                     self.logger.warning("Neither num_epochs nor max_steps is set. Training loop will not run.")
                     # Decide behavior: return early or let loop handle 0 iterations?
                     # Returning early for clarity
                     return {"final_epoch": self.epoch, "final_global_step": self.global_step, "status": "No training duration defined"}
                else:
                     # Run indefinitely until max_steps is hit inside the loop
                     # A large number can simulate this, or refactor loop condition
                     # Using a very large number for simplicity here
                     effective_num_epochs = sys.maxsize
                     self.logger.info(f"num_epochs not set, running until max_steps ({self.config.max_steps}) is reached.")
            else:
                 effective_num_epochs = self.config.num_epochs

            self.logger.info(f"Entering training loop (start_epoch={start_epoch}, effective_num_epochs={effective_num_epochs})...")
            for epoch in range(start_epoch, effective_num_epochs):
                self.epoch = epoch # Update current epoch
                # Initialize val_metrics for this epoch loop evaluation
                val_metrics_this_epoch: Optional[Dict[str, float]] = None # Rename to avoid redefinition clash

                self.logger.info(f"--- Starting Epoch {epoch + 1}/{effective_num_epochs} --- Global Step: {self.global_step}")

                # Callbacks: Epoch Begin
                self.callbacks.on_epoch_begin(epoch=epoch, global_step=self.global_step)

                # Run the training epoch via TrainingLoop
                epoch_metrics = self.training_loop.train_epoch(
                    trainer=self, # Pass self (Trainer) as context
                    current_epoch=epoch,
                    global_step=self.global_step, # Pass current global step
                    progress=self.progress, # Pass progress tracker
                    loaded_global_step=start_step if epoch == start_epoch else None # Pass resume step for first epoch
                )

                # Sync global step from progress tracker after epoch simulation
                if self.progress:
                     current_progress_step = getattr(self.progress, 'current_step', self.global_step)
                     if current_progress_step > self.global_step:
                         self.logger.debug(f"Updating Trainer global_step from {self.global_step} to {current_progress_step} based on ProgressTracker.")
                         self.global_step = current_progress_step
                     elif hasattr(self.progress, 'current_step') and current_progress_step < self.global_step:
                          self.logger.warning(f"ProgressTracker step ({current_progress_step}) is behind Trainer step ({self.global_step}). Not updating.")

                # Callbacks: Epoch End
                self.callbacks.on_epoch_end(
                    epoch=epoch, metrics=epoch_metrics, val_metrics=val_metrics_this_epoch
                )

                # --- Evaluation (Scheduled or End of Epoch) --- #
                # TODO: Revisit prev_step calculation - requires info from TrainingLoop
                # prev_step = self.global_step - (steps_in_epoch if not is_resuming_this_epoch else (steps_in_epoch - (resume_batch_offset+1)))
                # Pass global_step as prev_step for now, revisit triggering logic
                should_evaluate = self._should_run_eval(current_global_step=self.global_step, prev_global_step=self.global_step - 1)

                if should_evaluate and self.evaluator: # Also check if evaluator exists
                    self.logger.info(f"Running evaluation at global step {self.global_step}...")
                    self.callbacks.on_validation_begin(epoch=epoch, global_step=self.global_step)
                    # Evaluate using the evaluator instance
                    current_eval_metrics = self.evaluator.evaluate()
                    self.last_eval_time = time.time() # Update last eval time
                    if current_eval_metrics:
                        self.logger.info(f"Validation metrics at step {self.global_step}: {current_eval_metrics}")
                        # Log validation metrics via callbacks (e.g., TensorBoard)
                        self.callbacks.on_validation_end(
                            epoch=epoch,
                            global_step=self.global_step,
                            metrics=current_eval_metrics
                        )
                        # Keep track of the latest validation metrics
                        self.current_val_metrics = current_eval_metrics.copy() # Update current val metrics here as well
                    else:
                        self.logger.warning("Evaluation finished but returned no metrics.")
                        # Ensure renamed variable is set to None
                        current_eval_metrics = None
                else:
                     if should_evaluate and not self.evaluator:
                          self.logger.warning("Evaluation interval reached, but no Evaluator configured. Skipping evaluation.")

                # --- Checkpointing (Best and Interval) --- #
                should_save = False
                is_best = False # Reset for this evaluation check
                current_metric_val: Optional[float] = None # Type hint for clarity

                # Determine best based on validation metric
                if current_eval_metrics and self.checkpoint_manager:
                    # Use checkpointing config for metric name and mode
                    # Access attributes directly
                    val_metric_name = chkpt_cfg.val_metric if chkpt_cfg else "val_loss"
                    mode = chkpt_cfg.mode if chkpt_cfg else "min"

                    if val_metric_name:
                        current_metric_val = current_eval_metrics.get(val_metric_name)
                        if self.best_val_metric is None: # Initialize if first eval
                            self.best_val_metric = float('inf') if mode == 'min' else float('-inf')

                        if current_metric_val is not None and self.best_val_metric is not None:
                            if mode == 'min' and current_metric_val < self.best_val_metric:
                                self.best_val_metric = current_metric_val
                                is_best = True
                                self.logger.info(f"New best validation metric: {val_metric_name} = {self.best_val_metric}")
                            elif mode == 'max' and current_metric_val > self.best_val_metric:
                                self.best_val_metric = current_metric_val
                                is_best = True

                # Check save interval (only if not saving best or save_best_only is false)
                chkpt_save_interval = self.config.save_interval
                # Use save_best_only from checkpointing config
                # Access attribute directly
                save_best_only = chkpt_cfg.save_best_only if chkpt_cfg else False

                time_based_save = False
                # TODO: Implement time-based saving interval check
                # time_save_interval = chkpt_cfg.get("time_save_interval_seconds") if chkpt_cfg else None
                # if time_save_interval and time_save_interval > 0:
                #    if (time.time() - self.last_save_time) >= time_save_interval:
                #        time_based_save = True

                step_based_save = (
                    chkpt_save_interval is not None and
                    chkpt_save_interval > 0 and
                    self.global_step > 0 and
                    self.global_step % chkpt_save_interval == 0
                )

                # Determine if we should save this checkpoint
                if self.checkpoint_manager:
                    if is_best:
                        should_save = True # Always save if it's the best
                    elif not save_best_only and (step_based_save or time_based_save):
                        should_save = True # Save if interval reached and not save_best_only
                    # Add any other conditions for saving?

                if should_save and self.checkpoint_manager:
                    # Prepare state and filename for saving
                    state_to_save = self._prepare_training_state(epoch=epoch, global_step=self.global_step)
                    # Update metrics in state just before saving
                    # Type hint current_eval_metrics where it's used here
                    current_eval_metrics_typed: Optional[Dict[str, float]] = current_eval_metrics
                    state_to_save.metrics = current_eval_metrics_typed or epoch_metrics # Use eval metrics if available, else epoch
                    state_to_save.best_val_metric = self.best_val_metric # Ensure latest best is saved

                    # Add _best suffix if it is the best checkpoint
                    suffix = "_best" if is_best else ""
                    filename = f"{self.checkpoint_manager.checkpoint_prefix}_epoch_{state_to_save.epoch}_step_{state_to_save.global_step}{suffix}.pt"

                    self.logger.info(f"Saving checkpoint at step {self.global_step} (epoch {epoch+1}). Filename: {filename}, Is Best: {is_best}")
                    self.checkpoint_manager.save_checkpoint(
                        state=state_to_save,
                        filename=filename,
                        # Pass the renamed val_metrics variable
                        # Type hint here as well
                        metrics=(current_eval_metrics_typed), # Pass current eval metrics for context
                        is_best=is_best
                    )
                    # self.last_save_time = time.time() # Update time if time-based implemented

                # --- Max Steps Check (inside epoch loop) --- #
                # Check if max_steps is reached AFTER the epoch finishes and AFTER callbacks
                # FIXED: Access max_steps from the config object
                if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                    self.logger.info(
                        f"Reached max_steps ({self.config.max_steps}), stopping training."
                    )
                    self._stop_training = True # Signal to exit outer loop

                # Check for stopping condition
                if self._stop_training:
                    self.logger.info(f"Stopping training early at epoch {epoch + 1} due to signal.")
                    break

            if not self._stop_training:
                 self.logger.info("Completed all planned epochs.")

            # Aggregate final metrics if needed (simple placeholder)
            training_result = {"final_epoch": self.epoch, "final_global_step": self.global_step}

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user (KeyboardInterrupt).")
            self._stop_training = True # Set flag
            exception_object = sys.exc_info()[1] # Store exception
            # Remove incorrect assignment that was previously commented out
            # Call on_exception for callbacks
            try:
                self.callbacks.on_exception(exception=exception_object)
            except Exception as cb_e:
                 self.logger.error(f"Error during callback on_exception for KeyboardInterrupt: {cb_e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during TrainingLoop run: {e}", exc_info=True)
            exception_object = e # Store exception
            self._stop_training = True # Ensure cleanup runs
            # Call on_exception for callbacks
            try:
                self.callbacks.on_exception(exception=exception_object)
            except Exception as cb_e:
                 self.logger.error(f"Error during callback on_exception: {cb_e}", exc_info=True)

        finally:
            # --- Callbacks: Train End --- #
            # Pass final metrics if available
            # Ensure final_metrics has the correct type compatible with callback
            # Explicitly handle type for mypy clarity
            final_metrics: Optional[Dict[str, Any]] = None
            if isinstance(training_result, dict):
                # We know training_result is Dict[str, Any], which matches Optional[Dict[str, Any]]
                final_metrics = training_result
            elif training_result is not None:
                # Log if training_result is somehow not a dict (shouldn't happen)
                self.logger.warning(f"Unexpected type for training_result: {type(training_result)}. Passing None to on_train_end.")
            try:
                # Pass the explicitly typed final_metrics
                self.callbacks.on_train_end(metrics=final_metrics, exception=exception_object)
            except Exception as cb_e:
                 self.logger.error(f"Error during callback on_train_end: {cb_e}", exc_info=True)

            # Log final state
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            self.logger.info(f"[Trainer Train End] Final self.global_step: {self.global_step}")
            self.logger.info(f"Training run finished. Total training time: {format_time(elapsed_time)}")
            force_flush_logs() # Ensure all logs are written

        return training_result

def _is_jupyter() -> bool: # Simple check if running in a notebook
    """Check if the code is running in a Jupyter/IPython environment."""
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
