"""
Models package.

Imports implementations and the factory function for easy access.
Ensures model classes are registered with the factory upon import.
"""

# Import base classes and configs
from .base import Model, GenerativeModel, LanguageModel, VisionModel, MultiModalModel
from .base import BaseModelConfig, GenerativeModelConfig, LanguageModelConfig, VisionModelConfig, MultiModalModelConfig

# Import the factory function and registration decorator
from .factory import create_model_from_config, register_model

# Import specific model implementations to ensure they get registered
# Add other model implementations here as they are created
try:
    # from .gpt_decoder import GPTDecoder # Removed GPTDecoder
    pass # Placeholder if no other models are explicitly imported yet
except ImportError as e:
    print(f"Could not import GPTDecoder: {e}") # Use logging if available
try:
    from .transformer import TransformerModel
except ImportError as e:
    print(f"Could not import TransformerModel: {e}") # Use logging if available

__all__ = [
    # Base Classes & Configs
    "Model",
    "BaseModelConfig",
    "GenerativeModel",
    "GenerativeModelConfig",
    "LanguageModel",
    "LanguageModelConfig",
    "VisionModel",
    "VisionModelConfig",
    "MultiModalModel",
    "MultiModalModelConfig",
    # Factory & Registration
    "create_model_from_config",
    "register_model",
    # Implementations
    "TransformerModel",
] 