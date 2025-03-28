"""
Model implementations and utilities.

This module contains the model implementations for Craft, including:
- Abstract base model classes
- Transformer model implementations
- Factory functions for creating models
"""

from .base import (
    Model,
    ModelConfig,
    GenerativeModel,
    LanguageModel,
    VisionModel,
    MultiModalModel,
    create_model_from_config
)

from .transformer import (
    TransformerModel,
    create_transformer_model
)

from .gpt_decoder import (
    GPTDecoder,
    create_gpt_model
)

__all__ = [
    # Base classes
    'Model',
    'ModelConfig',
    'GenerativeModel',
    'LanguageModel',
    'VisionModel',
    'MultiModalModel',
    
    # Model implementations
    'TransformerModel',
    'GPTDecoder',
    
    # Factory functions
    'create_model_from_config',
    'create_transformer_model',
    'create_gpt_model',
] 