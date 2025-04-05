"""
Central registry for models and their associated Pydantic configuration classes.
"""
import logging
from typing import Dict, Type, Optional
import inspect

# Import base classes used for type hinting
from .base import Model
from .configs import BaseModelConfig

logger = logging.getLogger(__name__)

# Registry mapping _target_ string (or alias) to Model Class
_MODEL_REGISTRY: Dict[str, Type[Model]] = {}

# Registry mapping _target_ string (or alias) to Pydantic Config Class
MODEL_CONFIG_REGISTRY: Dict[str, Type[BaseModelConfig]] = {}

def register_model(name: str, config_cls: Optional[Type[BaseModelConfig]] = None):
    """
    Decorator to register a model class and optionally its Pydantic config class.

    Args:
        name: The registration name (string key, e.g., 'transformer', 'my_model',
              or full import path if preferred).
        config_cls: The Pydantic BaseModelConfig subclass associated with this model.
    """
    def decorator(cls: Type[Model]):
        if not issubclass(cls, Model):
            raise TypeError(f"Class {cls.__name__} is not a subclass of craft.models.base.Model")

        if name in _MODEL_REGISTRY:
            logger.warning(f"Overwriting registration for model name '{name}'. ")
        _MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model '{name}' -> {cls.__name__}")

        if config_cls:
            if not issubclass(config_cls, BaseModelConfig):
                 raise TypeError(f"Config class {config_cls.__name__} for model '{name}' is not a subclass of BaseModelConfig")
            if name in MODEL_CONFIG_REGISTRY:
                 logger.warning(f"Overwriting config registration for model name '{name}'.")
            MODEL_CONFIG_REGISTRY[name] = config_cls
            logger.debug(f"Registered config for '{name}' -> {config_cls.__name__}")
        elif name in MODEL_CONFIG_REGISTRY:
             # If registering a model without a config, but a config was previously registered,
             # it might indicate an issue. Log a warning.
             logger.warning(f"Model '{name}' registered without a config, but a config was previously registered ({MODEL_CONFIG_REGISTRY[name].__name__}).")

        # Add the registration name to the class itself for potential lookup
        # cls._registration_name = name
        return cls
    return decorator

# --- Helper Functions (Optional) ---

def get_model_class(name: str) -> Optional[Type[Model]]:
    """Retrieves a model class from the registry by name."""
    return _MODEL_REGISTRY.get(name)

def get_config_class(name: str) -> Optional[Type[BaseModelConfig]]:
    """Retrieves a Pydantic config class from the registry by name."""
    return MODEL_CONFIG_REGISTRY.get(name) 