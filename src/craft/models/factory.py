"""
Factory function for creating model instances based on configuration.
"""
import logging
from typing import Any, Dict, Type, Optional
import torch.nn as nn
from pydantic import ValidationError

# Import OmegaConf
from omegaconf import DictConfig, OmegaConf

# Import Hydra utils AFTER OmegaConf
import hydra.utils

# Get logger for this module
logger = logging.getLogger(__name__)

# Import base classes and configs (needed for type hints, but registry is moved)
from .base import Model, BaseModelConfig, LanguageModelConfig, VisionModelConfig, MultiModalModelConfig
# Import specific model implementations to register them (registration moved to __init__.py)
# from .transformer import TransformerModel # Example: Ensure this is imported
# from .rnn import RNNModel              # Example: Import other models

# --- Import Registries from __init__ --- #
from . import MODEL_CONFIG_REGISTRY, _MODEL_REGISTRY # Assuming _MODEL_REGISTRY is also moved or handled

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

def create_model_from_config(model_cfg: DictConfig, 
                               vocab_size: Optional[int] = None) -> nn.Module:
    """Instantiates a model based on the provided OmegaConf configuration.

    Args:
        model_cfg: The OmegaConf DictConfig object for the model.
                   Expected to have at least '_target_' and 'model_type'.
                   It might also contain a nested 'config' DictConfig for Pydantic validation.
        vocab_size: Explicit vocabulary size to override any value in the config.
                    Useful when vocab size is determined from data.

    Returns:
        An instantiated PyTorch model (nn.Module).

    Raises:
        ValueError: If configuration is invalid (e.g., missing target, unknown type).
        ValidationError: If the nested Pydantic config fails validation.
        ImportError: If the target class cannot be imported.
        Exception: For other instantiation errors.
    """
    logger.info(f"Attempting to create model with config: {model_cfg}")

    # Basic validation
    if "_target_" not in model_cfg:
        raise ValueError("Model configuration must include '_target_'.")
    if "model_type" not in model_cfg:
        logger.warning("'model_type' not found in top-level model config. Model type validation skipped.")
        model_type = None
    else:
        model_type = model_cfg.model_type

    # --- Pydantic Config Validation (if applicable) ---
    pydantic_config_instance = None
    if "config" in model_cfg and isinstance(model_cfg.config, DictConfig):
        logger.info(f"Found nested 'config' section. Attempting Pydantic validation for model_type: {model_type}")
        nested_config_cfg = model_cfg.config
        
        # Determine the expected Pydantic config class based on model_type
        config_cls = MODEL_CONFIG_REGISTRY.get(model_type)
        
        if not config_cls:
            logger.warning(f"No Pydantic config class registered for model_type '{model_type}'. Skipping Pydantic validation.")
        else:
            logger.info(f"Validating nested config against: {config_cls.__name__}")
            try:
                # Convert OmegaConf nested config to a plain dictionary for Pydantic
                # Resolve interpolations first
                resolved_nested_config = OmegaConf.to_container(nested_config_cfg, resolve=True)
                if not isinstance(resolved_nested_config, dict):
                    raise TypeError(f"Resolved nested config is not a dict: {type(resolved_nested_config)}")
                config_dict = resolved_nested_config

                # **Inject/Override vocab_size before validation if provided**
                if vocab_size is not None:
                    if 'vocab_size' in config_dict and config_dict['vocab_size'] != vocab_size:
                        logger.warning(f"Overriding vocab_size from config ({config_dict['vocab_size']}) with provided value ({vocab_size})")
                    config_dict['vocab_size'] = vocab_size
                elif 'vocab_size' not in config_dict:
                     # If not provided and not in config, Pydantic will raise error if required
                     logger.warning(f"vocab_size not provided explicitly and not found in nested config for {config_cls.__name__}. Validation might fail if required.")

                # Instantiate the Pydantic model for validation
                pydantic_config_instance = config_cls(**config_dict)
                logger.info(f"Pydantic validation successful for {config_cls.__name__}.")

            except ValidationError as e:
                logger.error(f"Configuration validation failed for model type '{model_type}': {e}")
                raise e # Re-raise the validation error
            except Exception as e:
                logger.error(f"Unexpected error during Pydantic validation: {e}")
                raise ValueError(f"Error validating nested model config: {e}") from e

    # --- Model Instantiation --- # 
    logger.info(f"Instantiating model: {model_cfg._target_}")
    try:
        # If Pydantic validation happened, pass the validated instance
        # Modify the instantiation call if the target model expects the *Pydantic object*
        # Check if the target expects a 'config' argument
        # This requires inspecting the target class signature, which is complex.
        # Simpler Approach: Assume the target model takes the validated pydantic_config_instance
        # if it exists, otherwise instantiate directly with model_cfg.

        # Revised approach: Always instantiate using Hydra, but ensure the 
        # nested 'config' within model_cfg IS the validated Pydantic object if validation occurred.
        if pydantic_config_instance is not None:
             # Replace the original DictConfig 'config' with the validated Pydantic object
             # This assumes the model's __init__ expects the Pydantic object at model_cfg.config
             # OmegaConf.update(model_cfg, "config", pydantic_config_instance, merge=True)
             # ^^^ This might not work if model_cfg is read-only or doesn't trigger setter.
             
             # Safer approach: If the target model's __init__ expects the Pydantic config
             # object DIRECTLY as an argument named 'config', we need to handle that.
             # Let's assume the common pattern where the model takes the Pydantic config:
             # model = TargetClass(config=pydantic_config_instance)
             # We can achieve this by passing the instance via hydra.utils.instantiate
             # if we structure the config carefully, but it's often simpler 
             # to handle it explicitly if validation occurred.
             
             # Let's stick to the assumption that the model __init__ takes the pydantic config.
             # We modify the args passed to instantiate.
             # Remove the nested 'config' DictConfig and pass the Pydantic object separately?
             # Or assume the target expects the Pydantic object.
             
             # Standard Hydra instantiation - assumes target reads model_cfg.config
             # We need to ensure model_cfg.config *is* the pydantic object. 
             # This is tricky with OmegaConf's structure.

             # Simplest Fix: Modify the target model's __init__ to accept the Pydantic config.
             # Assume `TransformerModel.__init__(self, config: LanguageModelConfig)`
             target_class = hydra.utils.get_class(model_cfg._target_)
             model = target_class(config=pydantic_config_instance) 
        else:
            # If no Pydantic validation, instantiate directly using Hydra
            model = hydra.utils.instantiate(model_cfg)

        if not isinstance(model, nn.Module):
            raise TypeError(f"Instantiated object is not a PyTorch Module (nn.Module), got: {type(model)}")
            
        logger.info(f"Successfully created model: {model.__class__.__name__}")
        return model
    
    except ImportError as e:
        logger.error(f"Could not import target class '{model_cfg._target_}': {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to instantiate model '{model_cfg._target_}': {e}", exc_info=True)
        # Provide more context if possible
        if pydantic_config_instance:
            logger.error(f"Instantiation attempted with validated Pydantic config: {pydantic_config_instance}")
        raise Exception(f"Error instantiating model '{model_cfg._target_}': {e}") from e 