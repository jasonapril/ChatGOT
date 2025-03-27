"""
Base model classes and abstractions for Craft.

This module provides abstract base classes for all model types in Craft,
enabling a consistent interface regardless of modality.
"""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class Model(ABC):
    """
    Abstract base class for all models in Craft.
    
    This class defines the common interface for all models, regardless
    of the specific model type (language, vision, etc.).
    """
    
    def __init__(self):
        """Initialize the base model."""
        super().__init__()
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
    
    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
        logging.info(f"Model saved to {path}")
    
    def load(self, path: str, device: Optional[torch.device] = None):
        """
        Load the model from a file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        logging.info(f"Model loaded from {path} to {device}")
    
    def get_config(self) -> Dict:
        """
        Get the model configuration.
        
        Returns:
            Dictionary with the model configuration
        """
        return {"model_type": self.model_type}


class GenerativeModel(Model):
    """
    Abstract base class for generative models in Craft.
    
    This class extends the base Model with generation capabilities.
    """
    
    def __init__(self):
        """Initialize the generative model."""
        super().__init__()
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
    
    def __init__(self):
        """Initialize the language model."""
        super().__init__()
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
    
    def __init__(self):
        """Initialize the vision model."""
        super().__init__()
        self.model_type = "vision"


class MultiModalModel(Model):
    """
    Abstract base class for multi-modal models in Craft.
    
    This class extends the base Model with multi-modal capabilities.
    """
    
    def __init__(self):
        """Initialize the multi-modal model."""
        super().__init__()
        self.model_type = "multi-modal"


def create_model_from_config(config: Dict) -> Model:
    """
    Create a model from a configuration dictionary.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_type = config.get("model_type", "language")
    
    # Import the appropriate module based on model type
    if model_type == "language":
        from .transformer import create_transformer_model
        if config.get("architecture") == "gpt":
            from .gpt_decoder import create_gpt_model
            return create_gpt_model(**config)
        else:
            return create_transformer_model(**config)
    elif model_type == "vision":
        raise NotImplementedError("Vision models not yet implemented")
    elif model_type == "multi-modal":
        raise NotImplementedError("Multi-modal models not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}") 