"""
Models module initialization.
Exposes base classes, factory functions, and triggers registration
by importing modules that use the @register_model decorator.
"""
import logging

# Import base classes and configs first
from .base import Model, GenerativeModel, LanguageModel, VisionModel, MultiModalModel
from .configs import (
    BaseModelConfig,
    GenerativeModelConfig,
    LanguageModelConfig,
    VisionModelConfig,
    MultiModalModelConfig,
    SimpleRNNConfig, # Ensure specific configs needed are available
    # Add other specific config imports if they are directly used or exported here
)

# Import the centralized registries and the registration decorator
from .registry import _MODEL_REGISTRY, MODEL_CONFIG_REGISTRY, register_model

# Import the factory function that USES the registries
from .factory import create_model_from_config # Can alias if needed: as create_model

# Import modules containing model definitions that use the @register_model decorator.
# This act of importing triggers the decorators and populates the registries.
from . import simple_rnn # Import the module
from . import transformer # Import the module
# Add imports for other model modules here (e.g., .cnn, .gan, etc.)

# Setup logger
logger = logging.getLogger(__name__)
logger.debug(f"Models module initialized. Models registered: {list(_MODEL_REGISTRY.keys())}")
logger.debug(f"Model configs registered: {list(MODEL_CONFIG_REGISTRY.keys())}")

# Define __all__ for explicit exports from this package level
__all__ = [
    # Base Classes & Configs
    "Model",
    "BaseModelConfig",
    "GenerativeModel", "GenerativeModelConfig",
    "LanguageModel", "LanguageModelConfig",
    "VisionModel", "VisionModelConfig",
    "MultiModalModel", "MultiModalModelConfig",

    # Factory & Registration Function (Expose if needed externally)
    "create_model_from_config",
    "register_model",

    # Registries (Expose if needed externally, use with caution)
    # "_MODEL_REGISTRY",
    # "MODEL_CONFIG_REGISTRY",

    # Specific Models & Configs (Exported from their modules, but can re-export here)
    "SimpleRNN", "SimpleRNNConfig",
    "TransformerModel", # Should be exported from transformer.py now
    # Add other re-exports as needed
] 