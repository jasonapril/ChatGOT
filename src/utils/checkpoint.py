"""
Checkpoint utilities for Craft.

This module provides functions for saving, loading, and managing model checkpoints.
"""
import os
import glob
import logging
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    loss: float = 0.0,
    additional_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        path: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer to save (optional)
        scheduler: Learning rate scheduler to save (optional)
        epoch: Current epoch number
        loss: Current loss value
        additional_data: Additional data to save in the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        path: Path to the checkpoint file
        model: Model to load the state into
        optimizer: Optimizer to load the state into (optional)
        scheduler: Learning rate scheduler to load the state into (optional)
        device: Device to load the model on (optional)
        
    Returns:
        Checkpoint data dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Checkpoint loaded from {path}")
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match checkpoint files
        
    Returns:
        Path to the latest checkpoint file or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Get all checkpoint files matching the pattern
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoints:
        logging.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    # Sort by modification time (most recent first)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    logging.info(f"Latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint


def count_checkpoints(checkpoint_dir: str, pattern: str = "*.pt") -> int:
    """
    Count the number of checkpoint files in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match checkpoint files
        
    Returns:
        Number of checkpoint files
    """
    if not os.path.exists(checkpoint_dir):
        return 0
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    return len(checkpoints)


def clean_old_checkpoints(
    checkpoint_dir: str,
    keep: int = 5,
    pattern: str = "model_epoch_*.pt"
) -> None:
    """
    Remove old checkpoints, keeping only the specified number of most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep: Number of recent checkpoints to keep
        pattern: File pattern to match checkpoint files
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all regular checkpoints (not including 'best_model.pt')
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if len(checkpoints) <= keep:
        return
    
    # Sort by modification time (oldest first)
    checkpoints.sort(key=os.path.getmtime)
    
    # Remove oldest checkpoints
    for checkpoint in checkpoints[:-keep]:
        os.remove(checkpoint)
        logging.info(f"Removed old checkpoint: {checkpoint}") 