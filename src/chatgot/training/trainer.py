"""Training module for ChatGoT."""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from chatgot.core.config import config_to_dict, get_full_path, save_config
from chatgot.data.dataset import prepare_dataloaders_from_config
from chatgot.models.model_io import find_latest_checkpoint, load_model, save_model
from chatgot.models.transformer import create_transformer_model
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


def setup_optimizer(
    model: nn.Module,
    cfg: DictConfig,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        cfg: Configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get optimizer settings
    optimizer_cfg = cfg.training.optimizer
    name = optimizer_cfg.name.lower()
    lr = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    
    # Set up optimizer
    if name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
            eps=optimizer_cfg.eps,
        )
    elif name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
            eps=optimizer_cfg.eps,
        )
    elif name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_cfg.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    
    # Set up scheduler
    scheduler = None
    scheduler_cfg = cfg.training.scheduler
    scheduler_name = scheduler_cfg.name.lower()
    
    if scheduler_name == "cosine":
        # Calculate warmup steps
        total_steps = cfg.training.epochs * (
            cfg.data.get("train_size", 1000) // cfg.training.batch_size + 1
        )
        warmup_steps = int(total_steps * scheduler_cfg.warmup_ratio)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * scheduler_cfg.min_lr_ratio,
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.step_size,
            gamma=scheduler_cfg.gamma,
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.factor,
            patience=scheduler_cfg.patience,
            min_lr=lr * scheduler_cfg.min_lr_ratio,
        )
    elif scheduler_name != "constant":
        logger.warning(f"Unsupported scheduler: {scheduler_name}, using constant learning rate")
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    clip_grad_norm: Optional[float] = None,
    use_amp: bool = False,
    log_interval: int = 10,
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        train_dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        epoch: Current epoch number
        clip_grad_norm: Gradient clipping norm
        use_amp: Whether to use automatic mixed precision
        log_interval: How often to log progress (in batches)
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    # Set up AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Use tqdm for progress bar
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (src, targets) in enumerate(pbar):
        # Move tensors to device
        src = src.to(device)
        targets = targets.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        if use_amp:
            # Use autocast for mixed precision
            with torch.cuda.amp.autocast():
                # Forward pass
                output = model(src)
                # Calculate loss
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            # Scale gradients and update
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            output = model(src)
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            # Backward pass
            loss.backward()
            # Clip gradients
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # Update weights
            optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
        
        # Log to MLflow
        if batch_idx % log_interval == 0:
            mlflow.log_metric("train_batch_loss", loss.item())
    
    # Calculate average loss
    avg_loss = total_loss / len(train_dataloader)
    elapsed = time.time() - start_time
    
    # Log metrics
    metrics = {
        "train_loss": avg_loss,
        "train_ppl": torch.exp(torch.tensor(avg_loss)).item(),
        "train_time": elapsed,
    }
    
    # Print summary
    logger.info(f"Epoch {epoch} | Train loss: {avg_loss:.4f} | Train PPL: {metrics['train_ppl']:.2f}")
    
    return metrics


def evaluate(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        val_dataloader: Validation dataloader
        criterion: Loss criterion
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for src, targets in tqdm(val_dataloader, desc="Evaluating"):
            # Move tensors to device
            src = src.to(device)
            targets = targets.to(device)
            
            # Forward pass
            output = model(src)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            # Update total loss
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(val_dataloader)
    
    # Log metrics
    metrics = {
        "val_loss": avg_loss,
        "val_ppl": torch.exp(torch.tensor(avg_loss)).item(),
    }
    
    # Print summary
    logger.info(f"Validation loss: {avg_loss:.4f} | Validation PPL: {metrics['val_ppl']:.2f}")
    
    return metrics


def train_model(
    cfg: DictConfig,
    resume: bool = False,
) -> int:
    """
    Train a model based on configuration.
    
    Args:
        cfg: Configuration
        resume: Whether to resume training from checkpoint
        
    Returns:
        0 for success, non-zero for failure
    """
    try:
        # Get training parameters
        train_cfg = cfg.training
        
        # Set device
        device = cfg.system.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Set random seed for reproducibility
        if cfg.system.seed is not None:
            torch.manual_seed(cfg.system.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.system.seed)
            logger.info(f"Set random seed to {cfg.system.seed}")
        
        # Prepare directories
        checkpoint_dir = get_full_path(cfg.paths.checkpoints_dir, cfg)
        tensorboard_dir = get_full_path(cfg.paths.tensorboard_dir, cfg)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri(cfg.experiment_tracking.uri)
        mlflow.set_experiment(cfg.experiment_tracking.experiment_name)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log config to MLflow
            mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
            
            # Get dataloaders
            logger.info("Preparing dataloaders")
            train_dataloader, val_dataloader = prepare_dataloaders_from_config(cfg)
            
            # Initialize model, optimizer, and criterion
            model = None
            optimizer = None
            scheduler = None
            start_epoch = 0
            best_val_loss = float('inf')
            
            # Resume from checkpoint if requested
            if resume:
                logger.info("Resuming from checkpoint")
                checkpoint_path = find_latest_checkpoint(checkpoint_dir)
                
                if checkpoint_path is not None:
                    checkpoint = load_model(checkpoint_path, cfg, device)
                    model = checkpoint["model"]
                    
                    if "optimizer_state_dict" in checkpoint:
                        optimizer_state = checkpoint["optimizer_state_dict"]
                    
                    if "epoch" in checkpoint:
                        start_epoch = checkpoint["epoch"] + 1
                        logger.info(f"Resuming from epoch {start_epoch}")
                    
                    if "loss" in checkpoint and checkpoint["loss"] < best_val_loss:
                        best_val_loss = checkpoint["loss"]
                        logger.info(f"Previous best validation loss: {best_val_loss:.4f}")
            
            # Create new model if not resuming or no checkpoint found
            if model is None:
                logger.info("Creating new model")
                model = create_transformer_model(
                    vocab_size=cfg.models.vocab_size,
                    d_model=cfg.models.d_model,
                    n_head=cfg.models.n_head,
                    d_hid=cfg.models.d_hid,
                    n_layers=cfg.models.n_layers,
                    dropout=cfg.models.dropout,
                    max_seq_length=cfg.training.sequence_length,
                    layer_norm_eps=cfg.models.layer_norm_eps,
                    memory_efficient=True,
                    use_activation_checkpointing=cfg.training.get("use_activation_checkpointing", False),
                    pos_encoding_type=cfg.models.positional_encoding.type,
                )
                model = model.to(device)
            
            # Set up optimizer and scheduler
            if optimizer is None:
                optimizer, scheduler = setup_optimizer(model, cfg)
                
                # Load optimizer state if resuming
                if resume and "optimizer_state_dict" in locals():
                    optimizer.load_state_dict(optimizer_state)
                    logger.info("Loaded optimizer state from checkpoint")
            
            # Set up loss function
            criterion = nn.CrossEntropyLoss()
            
            # Configure mixed precision
            use_amp = cfg.training.mixed_precision.enabled and torch.cuda.is_available()
            if use_amp:
                logger.info(f"Using automatic mixed precision ({cfg.training.mixed_precision.dtype})")
            
            # Set up gradient clipping
            clip_grad_norm = cfg.training.clip_grad_norm if cfg.training.clip_grad_norm > 0 else None
            
            # Training loop
            logger.info(f"Starting training for {train_cfg.epochs} epochs")
            
            # Calculate save and evaluation intervals
            save_interval = max(1, int(train_cfg.save_every))
            eval_steps = max(1, int(len(train_dataloader) * train_cfg.eval_every))
            
            logger.info(f"Saving checkpoints every {save_interval} epochs")
            logger.info(f"Evaluating every {eval_steps} steps")
            
            for epoch in range(start_epoch, train_cfg.epochs):
                mlflow.log_metric("epoch", epoch)
                
                # Train one epoch
                train_metrics = train_epoch(
                    model=model,
                    train_dataloader=train_dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    clip_grad_norm=clip_grad_norm,
                    use_amp=use_amp,
                )
                
                # Log training metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(key, value)
                
                # Evaluate on validation set
                val_metrics = evaluate(
                    model=model,
                    val_dataloader=val_dataloader,
                    criterion=criterion,
                    device=device,
                )
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    mlflow.log_metric(key, value)
                
                # Update learning rate scheduler
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics["val_loss"])
                    else:
                        scheduler.step()
                
                # Save model checkpoint
                save_if_best = val_metrics["val_loss"] < best_val_loss
                if save_if_best:
                    best_val_loss = val_metrics["val_loss"]
                    mlflow.log_metric("best_val_loss", best_val_loss)
                
                # Check if we should save checkpoint
                if (epoch + 1) % save_interval == 0 or save_if_best or epoch == train_cfg.epochs - 1:
                    # Create checkpoint path
                    checkpoint_name = f"model_epoch_{epoch:03d}.pt"
                    if save_if_best:
                        checkpoint_name = "model_best.pt"
                    
                    checkpoint_path = checkpoint_dir / checkpoint_name
                    
                    # Save model checkpoint
                    save_model(
                        model=model,
                        save_path=checkpoint_path,
                        config=cfg,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=val_metrics["val_loss"],
                    )
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")
            
            # Final evaluation
            final_metrics = evaluate(
                model=model,
                val_dataloader=val_dataloader,
                criterion=criterion,
                device=device,
            )
            
            logger.info("Training completed")
            logger.info(f"Final validation loss: {final_metrics['val_loss']:.4f}")
            logger.info(f"Final validation perplexity: {final_metrics['val_ppl']:.2f}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            
            # Log final metrics
            for key, value in final_metrics.items():
                mlflow.log_metric(f"final_{key}", value)
            
            return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1 