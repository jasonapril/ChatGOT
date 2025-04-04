#!/usr/bin/env python
"""
Main Trainer Module
==================

This module provides the main Trainer class that integrates all training components.
"""

import logging
import os
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .checkpointing import CheckpointManager
from .callbacks import CallbackList
from .generation import TextGenerator
from .progress import ProgressTracker
from craft.data.tokenizers.base import BaseTokenizer
from ..config.schemas import TrainingConfig

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
        resume_from_checkpoint: Optional[str] = None, # Keep resume flag
        compile_model: bool = False,  # Added compile_model argument
        **kwargs, # Allow for extra args, though we might remove later
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Trainer...")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config # Store the Pydantic config object
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer # Store tokenizer
        self.resume_from_checkpoint = resume_from_checkpoint # Store resume flag
        self.compile_model = compile_model # Store compile_model flag

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
        # self.checkpoint_dir = self.config.checkpoint_dir # REMOVED - Doesn't exist on TrainingConfig

        # Initialize scaler for mixed precision *before* CheckpointManager
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # Initialize components
        self.callbacks = CallbackList(callbacks or [])
        self.callbacks.set_trainer(self)
        
        # CheckpointManager now implicitly uses os.getcwd() for its directory
        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config.model_dump() if self.config else {}, # Pass config as dict if needed
            callbacks=self.callbacks,
            device=self.device,
            tokenizer=self.tokenizer # Pass tokenizer
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.metrics = {}
        self.loaded_tb_log_dir = None
        
        # Move model to device
        self.model.to(self.device)
        
        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            self.logger.info(f"Attempting to resume from checkpoint: {self.resume_from_checkpoint}")
            self._resume_from_checkpoint()
        
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
        try:
            state = self.checkpoint_manager.load_checkpoint(self.resume_from_checkpoint)
            if state:
                self.epoch = state.get('epoch', 0)
                self.global_step = state.get('global_step', 0)
                self.best_val_metric = state.get('best_val_metric', float('inf'))
                self.metrics = state.get('metrics', {})
                # Store the loaded TensorBoard path
                self.loaded_tb_log_dir = state.get('tensorboard_log_dir') 
                self.logger.info(f"Resumed trainer state from checkpoint at epoch {self.epoch}, step {self.global_step}")
                if self.loaded_tb_log_dir:
                    self.logger.info(f"Loaded TensorBoard log directory: {self.loaded_tb_log_dir}")
            else:
                 self.logger.error(f"Checkpoint load returned None state. Cannot resume state.")
                 # Decide how to handle: raise error, start fresh?
                 # Starting fresh for now
                 self.resume_from_checkpoint = None # Clear flag

        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}")
            # Optionally re-raise or handle more gracefully
            raise

    def train(self):
        """Main training loop."""
        self.logger.info("--- Starting Training Run ---")
        self.callbacks.on_train_begin(self) # Pass self
        self._stop_training = False # Flag to signal training should stop

        # Instantiate TrainingLoop *once* before the epoch loop
        training_loop = TrainingLoop(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            use_amp=self.use_amp,
            scaler=self.scaler,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            log_interval=self.log_interval,
            callbacks=self.callbacks,
            global_step=self.global_step, # Pass initial global step
            epoch=self.epoch, # Pass starting epoch
            max_steps=self.max_steps, # Pass max_steps
            # compile_model=self.compile_model # Removed, handled internally
        )
        training_loop.model_to_device() # Ensure model is on correct device before loop

        try:
            for epoch in range(self.epoch, self.num_epochs):
                self.epoch = epoch
                self.callbacks.on_epoch_begin(epoch=epoch)

                # Removed TrainingLoop instantiation from here

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
                train_metrics = training_loop.train_epoch(
                    # Pass dataloader directly
                    dataloader=self.train_dataloader,
                    current_epoch=epoch,
                    progress=progress,
                    global_step=self.global_step,
                    loaded_global_step=self.global_step if self.resume_from_checkpoint and epoch == (self.global_step // steps_in_epoch) else None
                )

                # Update global step based on the final step reached in train_epoch
                final_step_in_epoch = train_metrics.get('final_global_step', self.global_step) # Default to current if key missing
                self.global_step = final_step_in_epoch

                # Check if max_steps reached after the epoch completed
                if self.max_steps is not None and self.global_step >= self.max_steps:
                     self.logger.info(f"Reached max_steps ({self.max_steps}) after epoch {epoch+1}. Stopping training.")
                     break # Exit the epoch loop
                
                # Evaluation - Use self.eval_interval derived from config
                if self.val_dataloader is not None and self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                    evaluator = Evaluator(
                        model=self.model,
                        val_dataloader=self.val_dataloader,
                        device=self.device,
                        config=self.config, # Pass TrainingConfig here too if needed
                        use_amp=self.use_amp,
                        callbacks=self.callbacks
                    )
                    val_metrics = evaluator.evaluate()
                    
                    # Update best metric
                    current_val_loss = val_metrics.get('loss', float('inf')) # Safely get loss
                    if current_val_loss < self.best_val_metric:
                        self.best_val_metric = current_val_loss
                        if self.checkpoint_dir:
                            # Use the manager's dir
                            save_path = os.path.join(self.checkpoint_manager.checkpoint_dir, f"checkpoint_step_{self.global_step}_best.pt") # Indicate best
                            self.logger.info(f"New best validation metric: {self.best_val_metric:.4f}. Saving checkpoint to {save_path}")
                            self.checkpoint_manager.save_checkpoint(
                                path=save_path, 
                                current_epoch=self.epoch, 
                                global_step=self.global_step,
                                best_val_metric=self.best_val_metric,
                                metrics={**train_metrics, **val_metrics},
                                is_best=True 
                            )

                self.callbacks.on_epoch_end(epoch=epoch, logs=train_metrics)
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True) # Log traceback
            raise
        finally:
            self.callbacks.on_train_end()
            
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

