"""
Factory function for creating model instances based on configuration.
"""
import logging
from typing import Any, Dict, Type, Optional, Union
import torch.nn as nn
from pydantic import ValidationError

# Import OmegaConf
from omegaconf import DictConfig, OmegaConf

# Import Hydra utils AFTER OmegaConf
import hydra.utils

# Get logger for this module
logger = logging.getLogger(__name__)

# Import base model class only
from .base import Model
# Import config base class from the correct location
from .configs import BaseModelConfig
# Import specific model implementations to register them (registration moved to __init__.py)
# from .transformer import TransformerModel # Example: Ensure this is imported
# from .rnn import RNNModel              # Example: Import other models

# --- Import Registries and Decorator from registry.py --- #
from .registry import _MODEL_REGISTRY, MODEL_CONFIG_REGISTRY, register_model

# --- Model Registration ---
# Simple registry dictionary: maps (model_type, architecture_name) -> model_class
# _MODEL_REGISTRY: Dict[tuple[str, str | None], Type[Model]] = {}
# def register_model(model_type: str, architecture_name: str | None = None):
#     """
#     Decorator to register a model class with the factory.
#
#     Args:
#         model_type (str): The general type of the model (e.g., 'language', 'vision').
#         architecture_name (str | None): A specific architecture name for this type 
#                                          (e.g., 'gpt_decoder', 'transformer'). If None,
#                                          registers as the default for the model_type.
#     """
#     def decorator(cls: Type[Model]):
#         if not issubclass(cls, Model):
#             raise ValueError(f"Class {cls.__name__} is not a subclass of Model.")
#         
#         key = (model_type, architecture_name)
#         if key in _MODEL_REGISTRY:
#             logging.warning(f"Model key {key} is already registered. Overwriting {cls.__name__} -> {_MODEL_REGISTRY[key].__name__}")
#         
#         _MODEL_REGISTRY[key] = cls
#         logging.debug(f"Registered model: {key} -> {cls.__name__}")
#         return cls
#     return decorator
# Registry definitions and decorator are now in registry.py

# --- Factory Function ---

def create_model_from_config(model_cfg: Union[DictConfig, dict],
                                vocab_size: Optional[int] = None) -> nn.Module:
    """
    Creates a PyTorch model based on the provided configuration.
    Handles registry keys and full import paths for _target_.
    Performs Pydantic validation if a config class is registered.
    Passes the validated Pydantic config object (if applicable) or kwargs to the model's __init__.
    """
    logger.info(f"Attempting to create model with config: {model_cfg}")

    # --- Prepare Config Dictionary --- #
    if isinstance(model_cfg, DictConfig):
        model_dict = OmegaConf.to_container(model_cfg, resolve=True)
        if not isinstance(model_dict, dict):
             raise TypeError(f"Resolved model_cfg is not a dict: {type(model_dict)}")
    elif isinstance(model_cfg, dict):
        model_dict = model_cfg.copy()
    else:
         raise TypeError(f"model_cfg must be a DictConfig or dict, got {type(model_cfg)}")

    # --- Get Target Path/Key --- #
    if "_target_" in model_dict:
        target_key_or_path = model_dict.pop("_target_")
    elif "target" in model_dict:
        target_key_or_path = model_dict.pop("target")
    else:
        raise ValueError("Model configuration must include '_target_' or 'target'.")

    # --- Determine Target Class --- #
    target_class: Optional[Type[nn.Module]] = None
    if target_key_or_path in _MODEL_REGISTRY:
        target_class = _MODEL_REGISTRY[target_key_or_path]
        logger.info(f"Found registered model target: '{target_key_or_path}' -> {target_class.__name__}")
    else:
        logger.info(f"Target '{target_key_or_path}' not found in registry, attempting import...")
        try:
            target_class = hydra.utils.get_class(target_key_or_path)
            # Relax check: Allow non-nn.Module targets, responsibility lies with the caller/config
            if not callable(target_class):
                 raise TypeError(f"Located target '{target_key_or_path}' is not callable.")
            logger.info(f"Successfully located target class: {target_class.__name__}")
        except ImportError as e:
             logger.error(f"Could not import target class '{target_key_or_path}': {e}")
             raise Exception(f"Error locating target class '{target_key_or_path}'. Ensure it's registered or a valid import path.") from e
        except Exception as e:
            logger.error(f"Unexpected error locating target class '{target_key_or_path}': {e}")
            raise Exception(f"Error locating target '{target_key_or_path}'.") from e

    if target_class is None:
        raise ValueError(f"Could not determine target class for '{target_key_or_path}'.")

    # --- Prepare Pydantic Config Object (if applicable) --- #
    pydantic_config_instance: Optional[BaseModelConfig] = None
    # Use target_key_or_path to lookup config class, covers both registry and imported targets if registered
    config_cls: Optional[Type[BaseModelConfig]] = MODEL_CONFIG_REGISTRY.get(target_key_or_path)

    # Extract nested config dict (if present), remove from main dict
    # Nested dict takes priority for Pydantic validation
    nested_config_dict = model_dict.pop("config", None) # Use None if missing
    if nested_config_dict is not None and not isinstance(nested_config_dict, dict):
        logger.warning(f"'config' key found but its value is not a dictionary ({type(nested_config_dict)}). Ignoring nested config.")
        nested_config_dict = None # Treat as non-existent if not a dict

    config_source_dict = nested_config_dict if nested_config_dict is not None else model_dict

    # Inject/Override vocab_size into the source dictionary for validation
    if vocab_size is not None:
        if 'vocab_size' in config_source_dict and config_source_dict['vocab_size'] != vocab_size:
            logger.warning(f"Overriding config vocab_size ({config_source_dict['vocab_size']}) with provided value ({vocab_size})")
        config_source_dict['vocab_size'] = vocab_size

    if config_cls:
        logger.info(f"Validating config for '{target_key_or_path}' using {config_cls.__name__} with data: {config_source_dict}")
        try:
            pydantic_config_instance = config_cls(**config_source_dict)
            logger.info(f"Pydantic validation successful for {config_cls.__name__}.")
        except ValidationError as e:
            logger.error(f"Pydantic config validation failed for {config_cls.__name__} with data {config_source_dict}: {e}")
            raise Exception(f"Error validating config for model '{target_key_or_path}': {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Pydantic instantiation for {config_cls.__name__}: {e}")
            raise Exception(f"Error preparing config for model '{target_key_or_path}': {e}") from e
    else:
        # Indented block for the else statement
        logger.warning(f"No specific Pydantic config class registered for '{target_key_or_path}'. Passing remaining args. Model init must accept these args or handle config dict manually.")

    # --- Prepare Final Init Arguments --- #
    init_args = {}
    if pydantic_config_instance:
        # If we created a specific Pydantic config, pass it as 'config'
        init_args['config'] = pydantic_config_instance
        # Pass only the validated config object, assuming model uses it solely.
        # If models need other top-level args, they should be in the Pydantic model.
        # Clear model_dict if nested config was used for validation?
        if nested_config_dict is not None:
            model_dict = {} # Clear top-level if nested config was validated & used
        else:
             # If top-level dict was validated, remove validated keys? Safer to pass all.
             pass # Pass remaining model_dict args
        if model_dict: # Log if any top-level args remain after taking nested config
             logger.warning(f"Passing potentially unused top-level arguments ({list(model_dict.keys())}) along with validated config object to {target_class.__name__}.")
             init_args.update(model_dict)
    else:
        # If no Pydantic instance was created, pass the arguments directly.
        # Combine potentially modified nested dict with remaining top-level args.
        # Prioritize top-level arguments in case of conflicts?
        final_kwargs = {}
        if nested_config_dict is not None:
            final_kwargs.update(nested_config_dict)
        final_kwargs.update(model_dict) # Top-level overrides nested
        init_args = final_kwargs
        if not init_args:
             logger.warning(f"Instantiating {target_class.__name__} with no arguments after processing config.")

    # --- Model Instantiation --- #
    logger.info(f"Instantiating model: {target_class.__name__} with final args: {list(init_args.keys())}")
    logger.debug(f"Final init_args for {target_class.__name__}: {init_args}")
    try:
        # Instantiate the class with the prepared arguments
        model = target_class(**init_args)

        # Check if the instantiated object is an nn.Module AFTER instantiation
        if not isinstance(model, nn.Module):
             logger.warning(f"Instantiated object {type(model)} is not an nn.Module subclass.")

        logger.info(f"Successfully created model: {model.__class__.__name__}")
        return model

    except Exception as e:
        logger.error(f"Failed to instantiate model '{target_class.__name__}' with args {list(init_args.keys())}: {e}", exc_info=True)
        logger.debug(f"Instantiation failed with args: {init_args}")
        raise Exception(f"Error instantiating model '{target_class.__name__}': {e}") from e 