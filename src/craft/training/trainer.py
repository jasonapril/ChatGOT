#!/usr/bin/env python
"""
Main Trainer Module
==================

This module provides the main Trainer class that integrates all training components.
"""

import logging
import os
from typing import Optional, Dict, Any, List
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

class Trainer:
    """Main trainer class that coordinates all training components."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Any]] = None,
        checkpoint_dir: Optional[str] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        num_epochs: int = 1,
        resume_from_checkpoint: Optional[str] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.num_epochs = num_epochs
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Initialize components
        self.logger = logging.getLogger(self.__class__.__name__)
        self.callbacks = CallbackList(callbacks or [])
        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.metrics = {}
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            self._resume_from_checkpoint()

    def _resume_from_checkpoint(self):
        """Resume training from a checkpoint."""
        try:
            state = self.checkpoint_manager.load_checkpoint(self.resume_from_checkpoint)
            self.epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_val_metric = state['best_val_metric']
            self.metrics = state['metrics']
            self.logger.info(f"Resumed from checkpoint at epoch {self.epoch}, step {self.global_step}")
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}")
            raise

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.callbacks.on_train_begin()
        
        try:
            for epoch in range(self.epoch, self.num_epochs):
                self.epoch = epoch
                self.callbacks.on_epoch_begin(epoch=epoch)
                
                # Training loop
                training_loop = TrainingLoop(
                    model=self.model,
                    optimizer=self.optimizer,
                    train_dataloader=self.train_dataloader,
                    device=self.device,
                    config=self.config,
                    scheduler=self.scheduler,
                    use_amp=self.use_amp,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    max_grad_norm=self.max_grad_norm,
                    log_interval=self.log_interval,
                    callbacks=self.callbacks.callbacks
                )
                
                # Create progress tracker
                progress = ProgressTracker(
                    total_steps=len(self.train_dataloader),
                    log_interval=self.log_interval,
                    desc=f"Epoch {epoch + 1}/{self.num_epochs}"
                )
                
                # Training epoch
                train_metrics = training_loop.train_epoch(
                    current_epoch=epoch,
                    progress=progress,
                    global_step=self.global_step
                )
                
                # Update global step
                self.global_step += len(self.train_dataloader)
                
                # Evaluation
                if self.val_dataloader is not None and (epoch + 1) % self.eval_interval == 0:
                    evaluator = Evaluator(
                        model=self.model,
                        val_dataloader=self.val_dataloader,
                        device=self.device,
                        config=self.config,
                        use_amp=self.use_amp,
                        callbacks=self.callbacks
                    )
                    val_metrics = evaluator.evaluate()
                    
                    # Update best metric
                    if val_metrics['loss'] < self.best_val_metric:
                        self.best_val_metric = val_metrics['loss']
                        if self.checkpoint_dir:
                            self.checkpoint_manager.save_checkpoint(
                                epoch=self.epoch,
                                global_step=self.global_step,
                                best_val_metric=self.best_val_metric,
                                metrics={**train_metrics, **val_metrics}
                            )
                
                # Save checkpoint
                if self.checkpoint_dir and (epoch + 1) % self.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        epoch=self.epoch,
                        global_step=self.global_step,
                        best_val_metric=self.best_val_metric,
                        metrics=train_metrics
                    )
                
                self.callbacks.on_epoch_end(epoch=epoch)
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
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

