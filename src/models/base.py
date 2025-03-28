"""
Base model classes and abstractions for Craft.

This module provides abstract base classes for all model types in Craft,
enabling a consistent interface regardless of modality.
"""
from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.utils.checkpoint import save_checkpoint, load_checkpoint


class ModelConfig:
    """
    Configuration class for models.
    
    This class stores and validates model configuration parameters.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize model configuration with provided parameters.
        
        Args:
            **kwargs: Configuration parameters
        """
        self.__dict__.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            ModelConfig instance
        """
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"ModelConfig({', '.join(items)})"


class Model(nn.Module, ABC):
    """
    Abstract base class for all models in Craft.
    
    This class defines the common interface for all models, regardless
    of the specific model type (language, vision, etc.).
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the base model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or ModelConfig()
        self.model_type = "base"
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the model.
        
        Args:
            *args: Arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
        pass
    
    def _log_model_size(self):
        """Log the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"Model initialized with {n_params:,} parameters")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, 
             epoch: Optional[int] = None, step: Optional[int] = None, **kwargs):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
            optimizer: Optional optimizer state to save
            epoch: Optional current epoch
            step: Optional current step
            **kwargs: Additional information to save
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_config(),
            "model_type": self.model_type,
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        # Add training metadata if provided
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if step is not None:
            checkpoint["step"] = step
            
        # Add any additional information
        for key, value in kwargs.items():
            checkpoint[key] = value
        
        # Save checkpoint
        save_checkpoint(checkpoint, path)
        logging.info(f"Model saved to {path}")
    
    def load(self, path: str, device: Optional[torch.device] = None, 
             optimizer: Optional[torch.optim.Optimizer] = None, strict: bool = True) -> Dict[str, Any]:
        """
        Load the model from a file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce that the keys in state_dict match
            
        Returns:
            Dictionary with additional checkpoint information
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = load_checkpoint(path, device)
        
        # Load model state
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if missing_keys:
            logging.warning(f"Missing keys when loading model state: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading model state: {unexpected_keys}")
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Move model to device
        self.to(device)
        logging.info(f"Model loaded from {path} to {device}")
        
        # Return the full checkpoint for additional information
        return checkpoint
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary with the model configuration
        """
        if self.config is not None:
            return {
                "model_type": self.model_type,
                **self.config.to_dict()
            }
        return {"model_type": self.model_type}


class GenerativeModel(Model):
    """
    Abstract base class for generative models in Craft.
    
    This class extends the base Model with generation capabilities.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the generative model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_type = "generative"
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generate outputs from the model.
        
        Args:
            *args: Arguments for generation
            **kwargs: Keyword arguments for generation
            
        Returns:
            Generated outputs
        """
        pass


class LanguageModel(GenerativeModel):
    """
    Abstract base class for language models in Craft.
    
    This class extends GenerativeModel with language-specific functionality.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the language model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_type = "language"
    
    def calculate_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate perplexity from logits and targets.
        
        Args:
            logits: Model output logits
            targets: Target token indices
            
        Returns:
            Perplexity
        """
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        return torch.exp(loss)


class VisionModel(Model):
    """
    Abstract base class for vision models in Craft.
    
    This class extends the base Model with vision-specific functionality.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the vision model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_type = "vision"


class MultiModalModel(Model):
    """
    Abstract base class for multi-modal models in Craft.
    
    This class extends the base Model with multi-modal capabilities.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the multi-modal model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.model_type = "multi-modal"


def create_model_from_config(config: Dict[str, Any]) -> Model:
    """
    Create a model from a configuration dictionary.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_config = ModelConfig.from_dict(config)
    model_type = config.get("model_type", "language")
    
    # Import the appropriate module based on model type
    if model_type == "language":
        if config.get("architecture") == "gpt":
            from .gpt_decoder import create_gpt_model
            return create_gpt_model(config=model_config)
        else:
            from .transformer import create_transformer_model
            return create_transformer_model(config=model_config)
    elif model_type == "vision":
        raise NotImplementedError("Vision models not yet implemented")
    elif model_type == "multi-modal":
        raise NotImplementedError("Multi-modal models not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}") 