"""
Base model classes and abstractions for Craft.

This module provides abstract base classes for all model types in Craft,
enabling a consistent interface regardless of modality.
"""
from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.nn as nn
# Remove Pydantic imports if no longer needed directly
# from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationError
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

# Import the config class from the new location
from craft.config.schemas import BaseModelConfig # Import from central location

# Setup logger for this module
logger = logging.getLogger(__name__)

# Import from the new location
# Remove this import to break circular dependency. Checkpointing should be handled externally.
# from craft.training.checkpoint_utils import save_checkpoint, load_checkpoint


# Pydantic ModelConfig Base
# class BaseModelConfig(BaseModel):
#     """
#     Base Pydantic configuration class for all models.
#     Provides automatic validation and type hints.
#     Uses model_config for Pydantic V2 settings.
#     """
#     model_config = ConfigDict(extra='allow') # Keep allow for flexibility
#     model_type: str = Field("base", description="The type of the model (e.g., language, vision).")
#
#     # Common config fields can be added here if needed
#     # e.g., vocab_size: Optional[int] = None
#     # e.g., embedding_dim: Optional[int] = None
#
# class GenerativeModelConfig(BaseModelConfig):
#     """Config for Generative Models"""
#     model_type: str = Field("generative", description="Model type set to generative.")
#     max_seq_length: int = Field(1024, description="Maximum sequence length the model can handle")
#
# class LanguageModelConfig(GenerativeModelConfig):
#     """Config for Language Models"""
#     model_type: str = Field("language", description="Model type set to language.")
#     architecture: Optional[str] = Field(None, description="Name for the specific model architecture (e.g., transformer, rnn).")
#     vocab_size: int = Field(..., description="Size of the vocabulary (required).")
#     d_model: int = Field(768, description="Model dimension.")
#     n_head: int = Field(12, description="Number of attention heads.")
#     d_hid: Optional[int] = Field(None, description="Hidden dimension in feed-forward layers.")
#     n_layers: int = Field(12, description="Number of transformer layers.")
#     dropout: float = Field(0.1, description="Dropout probability.")
#     bias: bool = Field(True, description="Whether to use bias in linear layers.")
#     layer_norm_eps: float = Field(1e-5, description="Epsilon for layer normalization.")
#     activation: str = Field('gelu', description="Activation function.")
#     norm_first: bool = Field(True, description="Apply layer norm before attention/FFN.")
#
#     # Pydantic V2 validator for d_hid
#     @field_validator('d_hid', mode='before')
#     @classmethod
#     def set_d_hid_default(cls, v, info):
#         """Set default d_hid = d_model * 4 if not provided."""
#         # info.data should contain the raw input data being validated
#         if v is None and 'd_model' in info.data:
#             d_model = info.data.get('d_model')
#             # Use default d_model if not in data but defined in class
#             if d_model is None and 'd_model' in cls.model_fields:
#                  d_model = cls.model_fields['d_model'].default
#             
#             if isinstance(d_model, int):
#                 return d_model * 4
#         return v # Return original value if not calculated
#
# class VisionModelConfig(BaseModelConfig):
#     # ... (Additional fields specific to VisionModelConfig)
#     pass
#
# class MultiModalModelConfig(BaseModelConfig):
#     # ... (Additional fields specific to MultiModalModelConfig)
#     pass


# --- Base Model Class --- #
# *IMPORTANT*: Remove inheritance from Pydantic models. 
# Model IS an nn.Module, it HAS a config object.
class Model(nn.Module, ABC):
    """
    Abstract base class for all models.
    Inherits ONLY from nn.Module and ABC.
    Handles config assignment and basic save/load.
    """
    # Remove config related class attributes if any were added previously
    # No model_config = ConfigDict(...) here!

    def __init__(self, config: BaseModelConfig):
        """Initialize the base model with a validated Pydantic config object."""
        super().__init__() # Initialize nn.Module

        if not isinstance(config, BaseModelConfig):
             logging.error("Model received an invalid config object that is not a BaseModelConfig subclass.")
             # Raise error or use a default? Raising is safer.
             raise TypeError(f"Expected config to be a BaseModelConfig instance, got {type(config)}")
        
        self.config = config # Store the validated config object
        # Get architecture from the stored config object, replacing model_type
        self.architecture = self.config.architecture 
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass for the model.
        
        Args:
            *args: Arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
        pass
    
    def _log_model_size(self) -> None:
        """Log the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"Model initialized with {n_params:,} parameters")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration using Pydantic's model_dump.
        
        Returns:
            Dictionary with the model configuration
        """
        return self.config.model_dump()


# GenerativeModel inherits from the updated Model
class GenerativeModel(Model):
    """
    Base class for generative models.
    Checks config type and provides generate method.
    """
    def __init__(self, config: BaseModelConfig): # Accept BaseModelConfig initially
        super().__init__(config)
        
        # It's safe to assume self.config is GenerativeModelConfig or subclass here due to factory
        self.max_seq_length = getattr(self.config, 'max_seq_length', 1024)

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generates sequences based on input_ids.

        Args:
            input_ids: Tensor of starting token IDs (batch_size, seq_len).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Softmax temperature (0 for greedy).
            top_k: Keep only top_k tokens for sampling.
            top_p: Keep smallest set of tokens with cumulative probability >= top_p.
            repetition_penalty: Penalty applied to repeated tokens (1.0 = no penalty).
            eos_token_id: ID of the end-of-sequence token to stop generation.
            verbose: Log progress.

        Returns:
            Tensor containing the input_ids plus the generated tokens.
        """
        pass # Implementation moved to utils or handled by subclasses


# LanguageModel inherits from the updated GenerativeModel
class LanguageModel(GenerativeModel):
    """
    Base class for language models.
    Checks config type.
    """
    def __init__(self, config: BaseModelConfig): # Expects BaseModelConfig specifically
        super().__init__(config)
    
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
    """
    
    def __init__(self, config: BaseModelConfig):
        if not isinstance(config, BaseModelConfig):
            raise ValueError("VisionModel requires a BaseModelConfig instance.")
        super().__init__(config)


class MultiModalModel(Model):
    """
    Abstract base class for multi-modal models in Craft.
    """
    
    def __init__(self, config: BaseModelConfig):
        if not isinstance(config, BaseModelConfig):
            raise ValueError("MultiModalModel requires a BaseModelConfig instance.")
        super().__init__(config)


# --- Remove Factory Function --- 
# The create_model_from_config function has been moved to src/models/factory.py
# def create_model_from_config(config_dict: Dict[str, Any]) -> Model:
#    ... 