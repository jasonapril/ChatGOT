"""
Checkpoint utility functions for saving and loading model states.

This module provides functions for handling model checkpoints.
"""
import logging
import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(checkpoint: Dict[str, Any], path: str) -> None:
    """
    Save a checkpoint to a file.
    
    Args:
        checkpoint: Dictionary containing the checkpoint data
        path: Path to save the checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load a checkpoint from a file.
    
    Args:
        path: Path to the checkpoint file
        device: Device to load the tensors to
        
    Returns:
        Dictionary containing the checkpoint data
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    logging.info(f"Checkpoint loaded from {path}")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path to the latest checkpoint or None if no checkpoint is found
    """
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Get list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    # Return None if no checkpoint files found
    if not checkpoint_files:
        logging.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Sort checkpoint files by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # Return path to latest checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    logging.info(f"Latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint


def count_checkpoints(checkpoint_dir: str) -> int:
    """
    Count the number of checkpoint files in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Number of checkpoint files
    """
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return 0
    
    # Count checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    return len(checkpoint_files)


def clean_old_checkpoints(checkpoint_dir: str, keep: int = 5) -> None:
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        keep: Number of most recent checkpoints to keep
    """
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Get list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    # Return if not enough files to clean
    if len(checkpoint_files) <= keep:
        return
    
    # Sort checkpoint files by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # Remove old checkpoint files
    for checkpoint_file in checkpoint_files[keep:]:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            os.remove(checkpoint_path)
            logging.info(f"Removed old checkpoint: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to remove checkpoint {checkpoint_path}: {str(e)}") 