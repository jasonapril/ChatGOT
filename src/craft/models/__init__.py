"""
Models module initialization.
Exposes base classes, factory functions, and registers specific models.
"""
from typing import Dict, Type, Tuple

# Import base classes first (least likely to cause cycles)
from .base import Model, GenerativeModel, LanguageModel, VisionModel, MultiModalModel
from .base import BaseModelConfig, GenerativeModelConfig, LanguageModelConfig, VisionModelConfig, MultiModalModelConfig

# --- Registries (Defined here to break cycles) --- #

# Model Config Registry: Maps model_type string to Pydantic config class
MODEL_CONFIG_REGISTRY: Dict[str, Type[BaseModelConfig]] = {
    "language": LanguageModelConfig,
    "vision": VisionModelConfig,
    "multimodal": MultiModalModelConfig,
    "generative": GenerativeModelConfig, # Add generative base config
    "base": BaseModelConfig,           # Add base config
}

# Model Implementation Registry: Maps (model_type, architecture_name) tuple to Model class
_MODEL_REGISTRY: Dict[Tuple[str, str | None], Type[Model]] = {}

def register_model(model_type: str, model_cls: Type[Model], architecture_name: str | None = None):
    """Decorator or function to register a model class."""
    key = (model_type, architecture_name)
    if key in _MODEL_REGISTRY:
        # Use logging here if available, otherwise print
        print(f"Warning: Overwriting registration for key {key}") 
    _MODEL_REGISTRY[key] = model_cls
    # Return the class for decorator usage (though we moved away from it)
    return model_cls

def get_model_class(model_type: str, architecture_name: str | None = None) -> Type[Model] | None:
     """Looks up a model class in the registry."""
     key = (model_type, architecture_name)
     model_cls = _MODEL_REGISTRY.get(key)
     if not model_cls and architecture_name is not None:
         # Fallback to type default if specific architecture not found
         key = (model_type, None)
         model_cls = _MODEL_REGISTRY.get(key)
     return model_cls

# --- Import Factory AFTER defining registries --- #
from .factory import create_model_from_config

# Import specific model implementations *after* base and registries
from .transformer import TransformerModel
# from .rnn import RNNModel # Example

# --- Explicit Model Registration --- #
# Register models here using the locally defined function
register_model("language", TransformerModel, architecture_name="transformer")
# register_model("language", RNNModel, architecture_name="rnn") # Example

# --- Define __all__ --- #
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
    "MODEL_CONFIG_REGISTRY",
    "get_model_class",
    # Implementations
    "TransformerModel",
] 