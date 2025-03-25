"""Utilities for loading and saving models."""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig

from chatgot.core.config import get_full_path
from chatgot.models.transformer import TransformerModel, create_transformer_model
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


def save_model(
    model: TransformerModel,
    save_path: Union[str, Path],
    config: Optional[DictConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    loss: Optional[float] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save the model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save the model to
        config: Configuration used for training
        optimizer: Optimizer state to save
        epoch: Current epoch
        step: Current step
        loss: Current loss value
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "max_seq_length": model.max_seq_length,
        },
    }
    
    # Add optional components
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
        
    if step is not None:
        checkpoint["step"] = step
        
    if loss is not None:
        checkpoint["loss"] = loss
        
    if metadata is not None:
        checkpoint["metadata"] = metadata
        
    # Save the checkpoint
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Save config as well if provided
    if config is not None:
        config_path = save_path.with_suffix(".yaml")
        with open(config_path, "w") as f:
            from omegaconf import OmegaConf
            f.write(OmegaConf.to_yaml(config))
        logger.info(f"Config saved to {config_path}")


def load_model(
    load_path: Union[str, Path],
    config: Optional[DictConfig] = None,
    device: Optional[str] = None,
    strict: bool = True,
) -> Dict:
    """
    Load a model from a checkpoint.
    
    Args:
        load_path: Path to load the model from
        config: Configuration to use for creating the model
        device: Device to load the model to
        strict: Whether to strictly enforce that the keys in state_dict match
            the keys returned by this module's state_dict function
            
    Returns:
        Dictionary containing the loaded model and optionally other saved components
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    logger.info(f"Loading model from {load_path}")
    checkpoint = torch.load(load_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint.get("model_config", {})
    
    # If config is provided, use it to override saved configurations
    if config is not None and hasattr(config, "models"):
        # Extract model parameters from config
        model_params = dict(config.models)
        # Update with saved parameters for compatibility
        model_params.update(model_config)
    else:
        model_params = model_config
    
    # Create model instance
    if not model_params:
        raise ValueError("No model configuration found in checkpoint or provided config")
    
    # Get required parameters
    vocab_size = model_params.get("vocab_size")
    if vocab_size is None:
        raise ValueError("vocab_size not found in model configuration")
    
    # Create a new model instance
    model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=model_params.get("d_model", 768),
        n_head=model_params.get("n_head", 12),
        d_hid=model_params.get("d_hid", 3072),
        n_layers=model_params.get("n_layers", 12),
        dropout=model_params.get("dropout", 0.1),
        max_seq_length=model_params.get("max_seq_length", 1024),
        layer_norm_eps=model_params.get("layer_norm_eps", 1e-5),
        memory_efficient=model_params.get("memory_efficient", True),
        use_activation_checkpointing=model_params.get("use_activation_checkpointing", False),
        pos_encoding_type=model_params.get("pos_encoding_type", "learned")
    )
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model = model.to(device)
    
    # Create result dictionary
    result = {
        "model": model,
        "model_config": model_config,
    }
    
    # Add optional components if they exist
    if "optimizer_state_dict" in checkpoint:
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
        
    if "epoch" in checkpoint:
        result["epoch"] = checkpoint["epoch"]
        
    if "step" in checkpoint:
        result["step"] = checkpoint["step"]
        
    if "loss" in checkpoint:
        result["loss"] = checkpoint["loss"]
        
    if "metadata" in checkpoint:
        result["metadata"] = checkpoint["metadata"]
    
    logger.info(f"Model loaded successfully from {load_path}")
    return result


def find_latest_checkpoint(
    checkpoint_dir: Union[str, Path], 
    config: Optional[DictConfig] = None,
    pattern: str = "*.pt"
) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        config: Configuration to use for resolving paths
        pattern: Glob pattern to match checkpoint files
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints are found
    """
    checkpoint_dir = get_full_path(checkpoint_dir, config)
    
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint 