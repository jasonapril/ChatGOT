"""
Common utility functions for the project.
"""
import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(device_str: Optional[str] = None) -> torch.device:
    """
    Set up the device for training.
    
    Args:
        device_str: Device string (e.g., 'cuda:0', 'cpu', or None for auto-detect)
        
    Returns:
        PyTorch device
    """
    if device_str is None or device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device_str)
    
    if device.type == "cuda":
        # Log device info
        device_name = torch.cuda.get_device_name(device.index or 0)
        device_capability = torch.cuda.get_device_capability(device.index or 0)
        print(f"Using GPU: {device_name} with CUDA capability {device_capability[0]}.{device_capability[1]}")
        
        # Optimize for the device
        if device_capability[0] >= 7:  # Volta or newer
            print("Using tensor cores for mixed precision training")
    else:
        print("Using CPU for training (this will be slow)")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Evaluation metrics
        save_path: Path to save the checkpoint
        is_best: Whether this is the best model so far
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: str = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load the model to
        checkpoint_path: Path to the checkpoint
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, checkpoint


def get_latest_checkpoint(checkpoint_dir: str, pattern: str = "model_epoch_*.pt") -> Optional[str]:
    """
    Get the path to the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint files
        
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create an output directory for an experiment.
    
    Args:
        base_dir: Base directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created directory
    """
    # Create a timestamped directory
    timestamp = torch.tensor(0).get_device()  # Get current timestamp
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def save_args(args: Dict[str, Any], save_path: str) -> None:
    """
    Save arguments to a JSON file.
    
    Args:
        args: Dictionary of arguments
        save_path: Path to save the JSON file
    """
    # Convert any non-serializable types to strings
    serializable_args = {}
    for k, v in args.items():
        if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None))):
            serializable_args[k] = v
        else:
            serializable_args[k] = str(v)
    
    # Save to file
    with open(save_path, 'w') as f:
        json.dump(serializable_args, f, indent=2) 