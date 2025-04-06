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

from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .checkpointing import CheckpointManager, TrainingState, CheckpointLoadError
from .callbacks import CallbackList, TensorBoardLogger
from .generation import TextGenerator
from .progress import ProgressTracker
from craft.data.tokenizers.base import BaseTokenizer
from ..config.schemas import TrainingConfig
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
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainingConfig, # Expect TrainingConfig Pydantic model directly
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[Union[str, torch.device]] = None,
        callbacks: Optional[List[Any]] = None,
        tokenizer: Optional[BaseTokenizer] = None, # Added tokenizer param
        experiment_config: Optional[DictConfig] = None, # <-- Add this parameter
        experiment_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None, # Keep resume flag
        compile_model: bool = False,  # Added compile_model argument
        **kwargs: Any # Allow extra arguments for flexibility
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trainer...")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config # Store the Pydantic config object
        self.experiment_config = experiment_config # <-- Store the experiment config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer # Store tokenizer
        self.resume_from_checkpoint = resume_from_checkpoint # Store resume flag
        self.compile_model = compile_model # Store compile_model flag
        self.experiment_name = experiment_name or "default" # <-- Store experiment_name

        # Use default config if none provided
        if self.config is None:
            self.logger.warning("No TrainingConfig provided, using defaults.")
            self.config = TrainingConfig()
        elif not isinstance(self.config, TrainingConfig):
            # If a dict-like object is passed, try to parse it
            try:
                self.config = TrainingConfig(**config)
            except Exception as e:
                self.logger.error(f"Failed to parse provided config: {e}")
                raise ValueError("Invalid config provided to Trainer.") from e

        # --- Derived Attributes from Config ---
        # Access Pydantic model attributes directly
        self.num_epochs = self.config.num_epochs
        self.max_steps = self.config.max_steps
        self.use_amp = self.config.use_amp
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.max_grad_norm = self.config.max_grad_norm
        self.log_interval = self.config.log_interval
        self.eval_interval = self.config.eval_interval
        # Access checkpoint-related attributes directly from config
        self.save_interval = self.config.save_interval
        self.save_steps_interval = self.config.save_steps_interval
        self.time_save_interval_seconds = self.config.time_save_interval_seconds
        # self.checkpoint_dir = self.config.checkpoint_dir # REMOVED - Doesn't exist on TrainingConfig

        # --- Define Persistent Checkpoint Directory --- 
        # Use original CWD as base to ensure consistency across runs
        try:
            original_cwd = Path(get_original_cwd())
        except Exception:
            self.logger.warning("Could not get original CWD, using current CWD for checkpoint path.")
            original_cwd = Path.cwd()
        persistent_checkpoint_dir = original_cwd / "outputs" / self.experiment_name / "checkpoints"
        self.logger.info(f"Using persistent checkpoint directory: {persistent_checkpoint_dir}")
        # --- End Define Checkpoint Directory ---

        # Initialize scaler for mixed precision *before* CheckpointManager
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Log initial state BEFORE resume attempt
        self.logger.info(f"[Trainer Init] BEFORE Resume: Model State Hash: {get_state_hash(self.model.state_dict())}")
        if self.optimizer: # Check if optimizer exists
            self.logger.info(f"[Trainer Init] BEFORE Resume: Optimizer State Hash: {get_state_hash(self.optimizer.state_dict())}")
        else:
             self.logger.info("[Trainer Init] BEFORE Resume: No optimizer yet.")

        # Initialize components
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)
        
        # Setup Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            experiment_name=self.experiment_name, # Pass experiment_name
            # scaler=self.scaler, # Assuming GradScaler handled elsewhere
            callbacks=self.callbacks, # Pass callbacks for state saving/loading
            tokenizer=self.tokenizer,
            config=self.config.model_dump() if hasattr(self.config, 'model_dump') else OmegaConf.to_container(self.config, resolve=True) if self.config else None,
            checkpoint_dir=self.config.checkpoint_dir if hasattr(self.config, 'checkpoint_dir') else None, # <-- PASS checkpoint_dir
            max_checkpoints_to_keep=self.config.keep_last if hasattr(self.config, 'keep_last') and self.config.keep_last is not None else 5
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.metrics = {}
        self.loaded_tb_log_dir = None
        
        # Flags to trigger actions immediately after resuming
        self._just_resumed_trigger_save = False
        self._just_resumed_trigger_sample = False

        # Move model to device
        self.model.to(self.device)
        
        # Resume from checkpoint if specified
        self.resume_from_checkpoint = resume_from_checkpoint # Store resume flag FIRST
        if self.resume_from_checkpoint:
            self.logger.info(f"Attempting to resume from checkpoint: {self.resume_from_checkpoint}")
            self._resume_from_checkpoint()
            # Log state IMMEDIATELY AFTER resume attempt
            self.logger.info(f"[Trainer Init] AFTER Resume: Model State Hash: {get_state_hash(self.model.state_dict())}")
            if self.optimizer:
                self.logger.info(f"[Trainer Init] AFTER Resume: Optimizer State Hash: {get_state_hash(self.optimizer.state_dict())}")
        else:
             self.logger.info("No checkpoint specified, starting fresh.")
        
        # TensorBoard setup after potential resume
        tb_logger = None
        for cb in self.callbacks.callbacks: # Iterate through the list
            if cb.__class__.__name__ == 'TensorBoardLogger':
                tb_logger = cb
                break
        
        if tb_logger:
            if self.loaded_tb_log_dir:
                if hasattr(tb_logger, 'set_log_dir_absolute') and callable(getattr(tb_logger, 'set_log_dir_absolute')):
                     tb_logger.set_log_dir_absolute(self.loaded_tb_log_dir)
                     self.logger.info(f"Resuming TensorBoard logging to: {self.loaded_tb_log_dir}")
                else:
                     self.logger.warning("TensorBoardLogger found, but lacks set_log_dir_absolute method. Will create new log dir.")

    def _resume_from_checkpoint(self):
        """Resume training from a checkpoint."""
        # --- Add check for 'latest' --- 
        if self.resume_from_checkpoint is not None and self.resume_from_checkpoint.lower() == "latest":
            self.logger.error("Resuming with 'latest' is currently not supported due to directory ambiguity.")
            self.logger.error("Please provide an explicit path to a checkpoint file.")
            self.resume_from_checkpoint = None # Clear flag
            raise ValueError("Resuming with 'latest' checkpoint is not supported. Provide explicit path.")
        # --- End check ---
            
        try:
            # We now expect self.resume_from_checkpoint to be an explicit path or None
            state = self.checkpoint_manager.load_checkpoint(self.resume_from_checkpoint) 
            if state:
                # Update Trainer state from loaded TrainingState
                self.epoch = state.epoch
                self.global_step = state.global_step
                self.best_val_metric = state.best_val_metric if state.best_val_metric is not None else float('inf')
                self.metrics = state.metrics or {}
                self.loaded_tb_log_dir = state.tensorboard_log_dir
                # Set flags to trigger actions on first loop iteration after resume
                self._just_resumed_trigger_save = True 
                self._just_resumed_trigger_sample = True
                self.logger.info(f"Resumed trainer state from checkpoint at epoch {self.epoch}, step {self.global_step}")
                if self.loaded_tb_log_dir:
                    self.logger.info(f"Loaded TensorBoard log directory: {self.loaded_tb_log_dir}")
            else:
                 # If load returns None (e.g., no checkpoints found for 'latest')
                 self.logger.error(f"Checkpoint load returned None. Cannot resume state from {self.resume_from_checkpoint}.")
                 self.resume_from_checkpoint = None # Clear flag so we start fresh
                 raise CheckpointLoadError("Checkpoint load returned None state.") # Raise an error to indicate failure
        except FileNotFoundError as e: # Catch specifically FileNotFoundError
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
        self.callbacks.on_train_begin(self) # Pass self
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
        if self._just_resumed_trigger_save:
            self.logger.info("Performing immediate time-based save after resuming.")
            filename = f"checkpoint_step_{self.global_step:06d}_resumed.pt"
            # Create state (similar logic to loop, consider refactoring)
            serializable_config = self.config.model_dump() if isinstance(self.config, TrainingConfig) else self.config
            tb_log_dir = None
            if self.callbacks:
               for cb in self.callbacks.callbacks:
                   if cb.__class__.__name__ == 'TensorBoardLogger' and hasattr(cb, 'resolved_log_dir'):
                       tb_log_dir = cb.resolved_log_dir
                       break
            state = TrainingState(
                epoch=self.epoch,
                global_step=self.global_step,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                scaler_state_dict=self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None,
                best_val_metric=self.best_val_metric,
                metrics=self.metrics,
                config=serializable_config,
                tensorboard_log_dir=tb_log_dir,
            )
            self.checkpoint_manager.save_checkpoint(state=state, filename=filename, metrics=state.metrics, is_best=False)
            # Reset the TrainingLoop timer
            if hasattr(self.training_loop, 'last_time_based_save'):
                self.training_loop.last_time_based_save = time.time()
            self._just_resumed_trigger_save = False # Reset flag

        if self._just_resumed_trigger_sample:
            self.logger.info("Performing immediate time-based sample after resuming.")
            sample_cb_found = False
            if self.callbacks:
                for cb in self.callbacks.callbacks:
                     if cb.__class__.__name__ == 'SampleGenerationCallback':
                         cb.generate_samples(self, trigger_event=f"resume at step {self.global_step}") 
                         sample_cb_found = True
                         break # Keep the break after finding the callback
                if sample_cb_found:
                     # Reset the TrainingLoop timer
                     if hasattr(self.training_loop, 'last_time_based_sample'):
                         self.training_loop.last_time_based_sample = time.time()
                else:
                    self.logger.warning("Triggered immediate sample after resume, but SampleGenerationCallback not found.")
                self._just_resumed_trigger_sample = False # Reset flag
        # --- End Handle immediate actions ---

        # Log state JUST BEFORE epoch loop starts
        self.logger.info(f"[Trainer Train] PRE-LOOP: Model State Hash: {get_state_hash(self.model.state_dict())}")
        if self.optimizer:
             self.logger.info(f"[Trainer Train] PRE-LOOP: Optimizer State Hash: {get_state_hash(self.optimizer.state_dict())}")
        
        try:
            for epoch in range(self.epoch, self.num_epochs):
                self.epoch = epoch
                self.callbacks.on_epoch_begin(epoch=epoch)

                # Create progress tracker, considering max_steps for display
                # max_steps = getattr(self.config, 'max_steps', None) # Now self.max_steps
                steps_in_epoch = len(self.train_dataloader) # Full steps in this epoch

                # Calculate the number of steps remaining in the *entire run*
                remaining_run_steps = float('inf') # Default to infinite if max_steps is None
                if self.max_steps is not None:
                    remaining_run_steps = max(0, self.max_steps - self.global_step)

                # Determine the number of steps to actually run *in this specific epoch*
                steps_this_epoch = min(steps_in_epoch, remaining_run_steps)

                progress = ProgressTracker(
                    total_steps=steps_this_epoch, # Display total for *this epoch*
                    log_interval=self.log_interval,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}"
                )

                # Use the single training_loop instance created outside the loop
                train_metrics = self.training_loop.train_epoch(
                    current_epoch=epoch,
                    global_step=self.global_step,
                    progress=progress,
                    loaded_global_step=self.global_step if self.resume_from_checkpoint and epoch == (self.global_step // steps_in_epoch) else None
                )

                # Update global step based on the final step reached in train_epoch
                final_step_in_epoch = train_metrics.get('final_global_step', self.global_step) # Default to current if key missing
                self.global_step = final_step_in_epoch

                # Check if max_steps reached after the epoch completed
                if self.max_steps is not None and self.global_step >= self.max_steps:
                     self.logger.info(f"Reached max_steps ({self.max_steps}) after epoch {epoch+1}. Stopping training.")
                     break # Exit the epoch loop
                
                # Evaluation block
                is_best = False # Default to False if no evaluation
                val_metrics = {} # Default to empty if no evaluation
                if self.val_dataloader is not None and self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                    evaluator = Evaluator(
                        model=self.model,
                        val_dataloader=self.val_dataloader,
                        device=self.device,
                        config=self.config,
                        use_amp=self.use_amp,
                        callbacks=self.callbacks
                    )
                    val_metrics = evaluator.evaluate()
                    current_val_loss = val_metrics.get('loss', float('inf'))
                    is_best = current_val_loss < self.best_val_metric
                    if is_best:
                        self.best_val_metric = current_val_loss
                        self.logger.info(f"New best validation metric: {self.best_val_metric:.4f} at epoch {epoch+1}, step {self.global_step}")
                        # No separate save call here anymore

                # --- Save Checkpoint based on Interval (Simplified) --- #
                # Call save if interval is met. Pass the is_best status determined during eval.
                if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
                    filename = f"checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"
                    self.logger.info(f"Save interval reached. Saving checkpoint {filename} (is_best={is_best})")
                    
                    # --- Create TrainingState object --- #
                    current_state = TrainingState(
                        model_state_dict=self.model.state_dict(), # Get current model state
                        optimizer_state_dict=self.optimizer.state_dict() if self.optimizer else None,
                        scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                        scaler_state_dict=self.scaler.state_dict() if self.scaler and self.scaler.is_enabled() else None,
                        epoch=self.epoch,
                        global_step=self.global_step,
                        best_val_metric=self.best_val_metric,
                        metrics={**train_metrics, **val_metrics}, # Combined metrics
                        config=self.config.model_dump() if self.config else {},
                        # TODO: Add other relevant states like RNG states, callback states?
                    )
                    
                    # --- Call CheckpointManager with the state --- #
                    self.checkpoint_manager.save_checkpoint(
                        state=current_state,
                        filename=filename, 
                        is_best=is_best
                    )
                # --- End Simplified Save Check --- #

                self.callbacks.on_epoch_end(epoch=epoch, logs=train_metrics)
                
                # Check for stop signal from callbacks
                if self._stop_training:
                    self.logger.info("Stopping training based on callback signal.")
                    break
                    
            # After loop finishes normally
            if not self._stop_training:
                 self.logger.info("Training finished after completing specified epochs/steps.")

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True) # Log traceback
            raise
        finally:
            self.callbacks.on_train_end()
            self.logger.info("--- Training Run Ended ---")
            
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

