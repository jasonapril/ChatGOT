"""
Base training abstractions for Craft.

This module provides base classes and utilities for model training,
including trainers for different model types and training strategies.
"""
from abc import ABC, abstractmethod
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.base import Model


class Trainer(ABC):
    """
    Abstract base class for model trainers in Craft.
    
    This class defines the common interface for all model trainers,
    providing methods for training, evaluation, and checkpointing.
    """
    
    def __init__(
        self,
        model: Model,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            optimizer: The optimizer to use
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Configuration dictionary
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else {}
        
        # Move model to device
        self.model.to(self.device)
        
        # Default training parameters
        self.epochs = self.config.get('epochs', 10)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Set up optimizer if not provided
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Set up scheduler if not provided
        if self.scheduler is None and self.optimizer is not None:
            self.scheduler = self._create_scheduler()
        
        # Set up logging
        self.log_interval = self.config.get('log_interval', 10)
        
        # Set up checkpointing
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up mixed precision training
        self.use_amp = self.config.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.metrics = {'train_loss': [], 'val_loss': []}
        
        # Initialize best model tracking
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create an optimizer for the model.
        
        Returns:
            The optimizer
        """
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adamw').lower()
        lr = optimizer_config.get('learning_rate', 5e-5)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
                eps=optimizer_config.get('eps', 1e-8),
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(optimizer_config.get('beta1', 0.9), optimizer_config.get('beta2', 0.999)),
                eps=optimizer_config.get('eps', 1e-8),
                weight_decay=weight_decay,
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create a learning rate scheduler.
        
        Returns:
            The scheduler or None
        """
        scheduler_config = self.config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps,
                eta_min=self.optimizer.param_groups[0]['lr'] * scheduler_config.get('min_lr_ratio', 0.1),
            )
        elif scheduler_name == 'linear':
            from transformers import get_linear_schedule_with_warmup
            
            warmup_ratio = scheduler_config.get('warmup_ratio', 0.1)
            total_steps = len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps
            warmup_steps = int(total_steps * warmup_ratio)
            
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_name == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get('max_lr', 1e-3),
                total_steps=len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps,
                pct_start=scheduler_config.get('pct_start', 0.3),
                div_factor=scheduler_config.get('div_factor', 25),
                final_div_factor=scheduler_config.get('final_div_factor', 1000),
            )
        elif scheduler_name == 'constant':
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary of metrics
        """
        pass
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary of metrics for each epoch
        """
        logging.info(f"Starting training for {self.epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            self.metrics['train_loss'].append(train_metrics['loss'])
            
            # Evaluate if validation data is available
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                self.metrics['val_loss'].append(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'))
                    logging.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 1) == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f'model_epoch_{epoch + 1}.pt'))
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            epoch_info = f"Epoch {epoch + 1}/{self.epochs} - "
            epoch_info += f"train_loss: {train_metrics['loss']:.4f} - "
            
            if self.val_dataloader is not None:
                epoch_info += f"val_loss: {val_metrics['loss']:.4f} - "
            
            epoch_info += f"time: {epoch_time:.2f}s"
            logging.info(epoch_info)
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f}s")
        
        return self.metrics
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the model and training state.
        
        Args:
            path: Path to save the checkpoint
        """
        from ..utils.checkpoint import save_checkpoint as save_checkpoint_util
        
        additional_data = {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'steps': self.steps,
            'config': self.config
        }
        
        save_checkpoint_util(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            loss=self.best_val_loss,
            additional_data=additional_data
        )
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint of the model and training state.
        
        Args:
            path: Path to the checkpoint
        """
        from ..utils.checkpoint import load_checkpoint as load_checkpoint_util
        
        checkpoint = load_checkpoint_util(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('loss', float('inf'))
        self.train_metrics = checkpoint.get('train_metrics', {'loss': []})
        self.val_metrics = checkpoint.get('val_metrics', {'loss': []})
        self.steps = checkpoint.get('steps', 0)
        
        # Restore configuration if available
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])


class LanguageModelTrainer(Trainer):
    """
    Trainer for language models in Craft.
    
    This trainer specializes in training language models with
    autoregressive loss functions.
    """
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for i, batch in enumerate(self.train_dataloader):
            # Get batch
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, loss = outputs
                else:
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        targets.view(-1)
                    )
                
                # Scale by accumulation factor
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with mixed precision
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if gradient accumulation step is reached
            if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(self.train_dataloader):
                # Clip gradients
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Apply optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_tokens += inputs.numel()
            
            # Log progress
            if (i + 1) % self.log_interval == 0:
                logging.info(
                    f"Batch {i + 1}/{len(self.train_dataloader)} - "
                    f"loss: {loss.item() * self.gradient_accumulation_steps:.4f}"
                )
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_dataloader)
        
        return {'loss': avg_loss, 'tokens': total_tokens}
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Get batch
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, loss = outputs
                else:
                    logits = outputs
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        targets.view(-1)
                    )
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += inputs.numel()
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / len(self.val_dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {'loss': avg_loss, 'perplexity': perplexity, 'tokens': total_tokens}


def create_trainer_from_config(
    model: Model,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Dict[str, Any] = None,
) -> Trainer:
    """
    Create a trainer from a configuration dictionary.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        config: Configuration dictionary
        
    Returns:
        Instantiated trainer
    """
    if config is None:
        config = {}
    
    model_type = getattr(model, 'model_type', 'language')
    
    if model_type == 'language':
        return LanguageModelTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
        )
    elif model_type == 'vision':
        raise NotImplementedError("Vision model trainers not yet implemented")
    elif model_type == 'multi-modal':
        raise NotImplementedError("Multi-modal model trainers not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}") 