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
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import gc # Import garbage collector
import contextlib

# Import SafeGradScaler
from .amp import SafeGradScaler
# Import gradient checkpointing utility
from .utils import enable_gradient_checkpointing
# Import callback base
from .callbacks import TrainerCallback

from ..models.base import Model
from src.utils import save_checkpoint, load_checkpoint, ensure_directory


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
        callbacks: Optional[List[TrainerCallback]] = None
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
            callbacks: List of TrainerCallback instances
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else {}
        
        # Set up logger for this instance
        self.logger = logging.getLogger(self.__class__.__name__)
        # Example: Set level from config or default to INFO
        log_level_name = self.config.get('log_level', 'INFO').upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        self.logger.setLevel(log_level)
        # Ensure handlers are set up (could be done globally or here)
        # If no handlers are configured, messages might not appear.
        if not logging.getLogger().hasHandlers():
             # Basic handler if none exist (consider a more robust setup)
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             logging.getLogger().addHandler(handler)
             logging.getLogger().setLevel(log_level) # Set root logger level too, maybe
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing if configured
        self.use_gradient_checkpointing = self.config.get('use_gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            enable_gradient_checkpointing(self.model)
        
        # Default training parameters
        self.epochs = self.config.get('epochs', 10)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Initialize step and epoch counters
        self.global_step = 0
        self.current_epoch = 0
        
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
        ensure_directory(self.checkpoint_dir)
        
        # Set up mixed precision training using SafeGradScaler
        self.use_amp = self.config.get('use_amp', False)
        # Pass max_nan_before_fallback from config, default to 3
        max_nan = self.config.get('amp_max_nan_fallback', 3)
        self.scaler = SafeGradScaler(enabled=self.use_amp, max_nan_before_fallback=max_nan) if self.use_amp else None
        
        # Metrics tracking
        self.metrics = {'train_loss': [], 'val_loss': []}
        
        # Initialize best model tracking
        self.best_val_loss = float('inf')

        # Initialize callbacks
        self.callbacks = callbacks if callbacks is not None else []
        for callback in self.callbacks:
            callback.set_trainer(self)
    
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
        self.logger.info(f"Starting training for {self.epochs} epochs on device {self.device}")
        self._callback_on_train_begin() # Call hook

        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self._callback_on_epoch_begin(epoch) # Call hook

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
                    best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                    self.save_checkpoint(best_model_path)
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}. Saved model to {best_model_path}")
            
            # Save checkpoint periodically
            save_interval = self.config.get('save_interval', 1)
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Log epoch progress
            epoch_time = time.time() - epoch_start_time
            epoch_info = f"Epoch {epoch + 1}/{self.epochs} - "
            epoch_info += f"train_loss: {train_metrics['loss']:.4f} - "
            
            if self.val_dataloader is not None:
                epoch_info += f"val_loss: {val_metrics['loss']:.4f} - "
            
            epoch_info += f"time: {epoch_time:.2f}s"
            self.logger.info(epoch_info)
            # Add metrics to logs for callback
            epoch_logs = {**train_metrics, **val_metrics} if self.val_dataloader else train_metrics
            self._callback_on_epoch_end(epoch, logs=epoch_logs) # Call hook
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        self._callback_on_train_end() # Call hook

        return self.metrics
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the model and training state.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save scaler state if using AMP
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_checkpoint(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint of the model and training state.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = load_checkpoint(path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_loss', float('inf'))
        
        # Note: Gradient checkpointing hooks should ideally persist after loading state_dict.
        # If issues arise, consider re-applying here:
        # if self.use_gradient_checkpointing:
        #     enable_gradient_checkpointing(self.model) # Re-apply after loading state
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load scaler state if using AMP and saved in checkpoint
        if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # --- Callback Hook Methods --- #
    def _callback_on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def _callback_on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)

    def _callback_on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=logs)

    def _callback_on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    def _callback_on_step_begin(self, step, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_begin(step, logs=logs)

    def _callback_on_step_end(self, step, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_end(step, logs=logs)

    # --- End Callback Hook Methods --- #


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
        epoch_loss = 0.0
        num_valid_steps_in_epoch = 0 # Track steps that didn't have NaN/Inf loss
        total_tokens = 0
        steps_in_epoch = len(self.train_dataloader)
        
        # Use tqdm for progress bar
        progress_bar = tqdm(enumerate(self.train_dataloader), total=steps_in_epoch, desc=f"Epoch {self.current_epoch + 1}")
        
        # Reset optimizer gradients at the start
        self.optimizer.zero_grad(set_to_none=True)
        
        for i, batch in progress_bar:
            # Get batch
            inputs, targets = batch
            # Pass step number (global_step) to callback
            step_logs = {}
            self._callback_on_step_begin(self.global_step, logs=step_logs)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Determine if this is an accumulation step
            is_accumulation_step = (i + 1) % self.gradient_accumulation_steps != 0
            is_last_batch_step = (i + 1) == steps_in_epoch
            should_step = not is_accumulation_step or is_last_batch_step
            
            # <<< OOM Handling Block Start >>>
            try:
                # --- Core Training Step ---
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    
                    # Handle different output formats (model might return loss directly)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, loss_unscaled = outputs
                    else:
                        logits = outputs # Assume model returns logits
                        loss_unscaled = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                    
                    # Check for NaN/Inf in unscaled loss *before* scaling/accumulation division
                    loss_val = loss_unscaled.item()
                    is_loss_invalid = torch.isnan(loss_unscaled).any() or torch.isinf(loss_unscaled).any()
                    
                    if is_loss_invalid:
                        self.logger.warning(f"Step {self.global_step}, Batch {i+1}/{steps_in_epoch}: NaN/Inf loss detected: {loss_val}. Skipping.")
                        if should_step: self.optimizer.zero_grad(set_to_none=True) # Clear potentially bad grads accumulated in this cycle
                        continue # Move to the next batch
                    
                    # Normalize loss for gradient accumulation
                    loss = loss_unscaled / self.gradient_accumulation_steps
                
                # --- Backward Pass --- (Scaled)
                # Context manager delays DDP sync until the final grad accumulation step
                ddp_sync_context = self.model.no_sync() if is_accumulation_step and isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else contextlib.nullcontext()

                with ddp_sync_context:
                    # self.scaler handles mixed precision scaling
                    self.scaler.scale(loss).backward()
                
                # --- Metrics Accumulation (only if valid step) ---
                epoch_loss += loss_val # Use the unscaled item for epoch average
                num_valid_steps_in_epoch += 1
                total_tokens += inputs.numel()

                # --- Optimizer Step & Gradient Clipping ---
                if should_step:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step (self.scaler handles grad check and skipping step if required)
                    self.scaler.step(self.optimizer)

                    # Update the scaler for next iteration
                    self.scaler.update()

                    # Zero gradients *after* stepping for the next accumulation cycle/batch
                    self.optimizer.zero_grad(set_to_none=True)

                    # Step the scheduler (if one exists)
                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1

                    # --- Logging --- (Optional: update progress bar)
                    # Add step metrics to logs for callback
                    step_logs['loss'] = loss_val
                    step_logs['lr'] = lr = self.optimizer.param_groups[0]['lr']

                    if self.global_step % self.log_interval == 0:
                        progress_bar.set_postfix({'loss': f'{loss_val:.4f}', 'lr': f'{lr:.2e}'})
                        # Example detailed log message
                        self.logger.debug(f"Step: {self.global_step}, Batch: {i+1}/{steps_in_epoch}, Loss: {loss_val:.4f}, LR: {lr:.2e}")

                    # Call step end callback *after* optimizer/scheduler steps
                    self._callback_on_step_end(self.global_step, logs=step_logs)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    self.logger.error(f"CUDA Out of Memory error encountered at step {self.global_step}, batch index {i}. Attempting graceful shutdown.")
                    # Clean up memory
                    del inputs, targets, outputs, loss, loss_unscaled, logits # Delete tensors from this iteration
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.logger.info("Cleared variables and GPU cache after OOM.")

                    # Attempt to save OOM checkpoint
                    try:
                        oom_checkpoint_path = os.path.join(self.checkpoint_dir, "oom_checkpoint.pt")
                        self.save_checkpoint(oom_checkpoint_path)
                        self.logger.info(f"Successfully saved OOM checkpoint to {oom_checkpoint_path}")
                    except Exception as save_e:
                        self.logger.error(f"Failed to save OOM checkpoint: {save_e}", exc_info=True)

                    # Re-raise the error to stop training
                    raise e
                else:
                    # Re-raise other runtime errors
                    self.logger.error(f"Caught unexpected RuntimeError at step {self.global_step}, batch index {i}: {e}", exc_info=True)
                    raise e
            # <<< OOM Handling Block End >>>

        # Calculate average loss for the epoch based on valid steps
        avg_loss = epoch_loss / num_valid_steps_in_epoch if num_valid_steps_in_epoch > 0 else 0.0

        return {'loss': avg_loss, 'tokens': total_tokens}
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        progress_bar = tqdm(self.val_dataloader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    # Handle different output formats
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, loss = outputs
                    else:
                        logits = outputs
                        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    total_loss += loss.item()
                    total_tokens += inputs.numel()

        avg_loss = total_loss / len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0.0
        # Calculate perplexity if possible (you might need vocab size or specific loss calculation)
        # perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float('inf')

        return {'loss': avg_loss} #, 'perplexity': perplexity}


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