"""
Factory function for creating model instances based on configuration.
"""
import logging
from typing import Any, Dict, Type

# Import base classes and configs
from .base import Model, BaseModelConfig, LanguageModelConfig, VisionModelConfig, MultiModalModelConfig

# --- Model Registration ---
# Simple registry dictionary: maps (model_type, architecture_name) -> model_class
_MODEL_REGISTRY: Dict[tuple[str, str | None], Type[Model]] = {}

def register_model(model_type: str, architecture_name: str | None = None):
    """
    Decorator to register a model class with the factory.

    Args:
        model_type (str): The general type of the model (e.g., 'language', 'vision').
        architecture_name (str | None): A specific architecture name for this type 
                                         (e.g., 'gpt_decoder', 'transformer'). If None,
                                         registers as the default for the model_type.
    """
    def decorator(cls: Type[Model]):
        if not issubclass(cls, Model):
            raise ValueError(f"Class {cls.__name__} is not a subclass of Model.")
        
        key = (model_type, architecture_name)
        if key in _MODEL_REGISTRY:
            logging.warning(f"Model key {key} is already registered. Overwriting {cls.__name__} -> {_MODEL_REGISTRY[key].__name__}")
        
        _MODEL_REGISTRY[key] = cls
        logging.debug(f"Registered model: {key} -> {cls.__name__}")
        return cls
    return decorator

# --- Factory Function ---

def create_model_from_config(config_dict: Dict[str, Any]) -> Model:
    """
    Factory function to create a model instance from a configuration dictionary.
    Uses Pydantic for config validation and instantiation, and a registry for model lookup.

    Args:
        config_dict: Dictionary containing the model configuration. 
                     Must include 'model_type'. May include 'architecture'.

    Returns:
        An instance of the specified model class.
    """
    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError("Configuration must include a 'model_type' field.")

    # Architecture can be specified to select among models of the same type
    # Look for architecture first at the top level, then inside nested 'config' if present
    architecture = config_dict.get("architecture")
    if not architecture and isinstance(config_dict.get("config"), dict):
        architecture = config_dict["config"].get("architecture")
    # e.g., "gpt_decoder", "transformer"

    # Determine the appropriate Pydantic config class based *only* on model_type
    config_cls: Type[BaseModelConfig]
    if model_type == "language":
        config_cls = LanguageModelConfig
    elif model_type == "vision":
        config_cls = VisionModelConfig
    elif model_type == "multimodal":
        config_cls = MultiModalModelConfig
    elif model_type == "base" or model_type == "generative":
        raise ValueError(f"Cannot directly instantiate abstract model type '{model_type}'. Provide a concrete type.")
    else:
        # If type is unknown, maybe try BaseModelConfig? Or raise error?
        # Let's raise for now, expecting known types.
        raise ValueError(f"Unknown model type for config determination: {model_type}")

    # Validate and create the specific Pydantic config object
    try:
        # We primarily validate the structure here. The architecture variable extracted
        # earlier (from the raw dict) is used for registry lookup.
        model_config = config_cls(**config_dict)
        
        # Optional: Log a warning if the architecture extracted from the dict differs
        # from what might be in the validated Pydantic object (if it exists there).
        if architecture and hasattr(model_config, 'architecture') and model_config.architecture != architecture:
             logging.warning(f"Architecture mismatch: Dict specified '{architecture}', "
                             f"but Pydantic config resolved to '{model_config.architecture}'. "
                             f"Using '{architecture}' for model registry lookup.")
        # DO NOT reset the 'architecture' variable here: 
        # architecture = model_config.architecture 

    except Exception as e: # Catch Pydantic validation errors
        logging.error(f"Configuration validation failed for model type '{model_type}': {e}")
        raise # **EXIT POINT FOR VALIDATION ERRORS**

    # --- Model Class Lookup using Registry ---
    # Uses the 'architecture' variable determined from the input dictionary
    model_cls: Type[Model] | None = None
    
    # Try lookup with specific architecture first
    if architecture:
        key = (model_type, architecture)
        model_cls = _MODEL_REGISTRY.get(key)
        if not model_cls:
             logging.warning(f"No model registered for key {key}. Falling back to default for type '{model_type}'.")

    # Fallback to default for the model type (architecture=None)
    if not model_cls:
         key = (model_type, None)
         model_cls = _MODEL_REGISTRY.get(key)

    if not model_cls:
        # Import implementations here dynamically or ensure they are imported elsewhere
        # to populate the registry *before* the factory is called.
        # For now, we rely on models being registered via decorators at import time.
        logging.error(f"Model registry: {_MODEL_REGISTRY}") # Log registry content for debugging
        raise ValueError(f"No model implementation found in registry for type='{model_type}', architecture='{architecture}'. Ensure the model class is registered.")
    # --- End Lookup ---

    # Instantiate the model with the validated config
    try:
        model = model_cls(config=model_config)
        model._log_model_size() # Log size after instantiation
        return model
    except Exception as e:
         logging.error(f"Failed to instantiate model class {model_cls.__name__} with config {model_config}: {e}")
         raise 