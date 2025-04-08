# src/craft/utils/model_io.py
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, Type, cast

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
import torch.nn as nn

# Craft specific imports
from ..models.base import Model
from ..data.tokenizers.base import Tokenizer
from ..core.factories import create_model, create_tokenizer
from ..config.schemas import BaseModelConfig

logger = logging.getLogger(__name__)

ARTIFACT_CONFIG_KEY = "model_config"
ARTIFACT_STATE_DICT_KEY = "model_state_dict"
TOKENIZER_CONFIG_KEY = "tokenizer_config"

def save_model_for_inference(
    model: Model,
    tokenizer: Tokenizer,
    path: Union[str, Path],
    save_format: str = "directory",
    **kwargs: Any
) -> None:
    """
    Save a model and its tokenizer for inference or transfer.

    Supports saving as a directory (recommended, HF compatible) or single .pt file (legacy).

    Args:
        model: The model instance.
        tokenizer: The tokenizer instance.
        path: Path to the output directory or .pt file.
        save_format: 'directory' or 'pt_legacy'.
        **kwargs: Additional metadata to include in the saved dictionary (pt_legacy only).
    """
    path = Path(path)

    if save_format == "directory":
        logger.info(f"Saving model and tokenizer to directory: {path}")
        path.mkdir(parents=True, exist_ok=True)

        # Save Model
        try:
             config_dict = model.config.model_dump(mode='json')
             config_path = path / "config.json"
             with open(config_path, 'w', encoding='utf-8') as f:
                 json.dump(config_dict, f, indent=4)
             logger.info(f"Model config saved to {config_path}")

             weights_path = path / "pytorch_model.bin"
             torch.save(model.state_dict(), weights_path)
             logger.info(f"Model weights saved to {weights_path}")
        except Exception as e:
             logger.error(f"Failed to save model components to {path}: {e}", exc_info=True)
             raise RuntimeError(f"Failed to save model to {path}") from e

        # Save Tokenizer
        try:
             tokenizer.save(str(path))
             logger.info(f"Tokenizer saved successfully to directory {path}")
        except Exception as e:
             logger.error(f"Failed to save tokenizer to {path}: {e}", exc_info=True)
             raise RuntimeError(f"Failed to save tokenizer to {path}") from e

    elif save_format == "pt_legacy":
         logger.warning(f"Saving in legacy .pt format to {path}. 'directory' format is recommended.")
         save_dir = path.parent
         if save_dir:
              save_dir.mkdir(parents=True, exist_ok=True)

         try:
              model_config_dict = model.config.model_dump(mode='json')
         except Exception as e:
              logger.error(f"Failed to serialize model config: {e}", exc_info=True)
              model_config_dict = {}

         # Try to get tokenizer config as a dictionary if possible
         tokenizer_config_dict: Optional[Dict] = None
         try:
             tokenizer_config_attr = getattr(tokenizer, "config", None) # Can be dict, Pydantic model, or None
             if isinstance(tokenizer_config_attr, dict):
                 tokenizer_config_dict = tokenizer_config_attr
             elif tokenizer_config_attr is not None and hasattr(tokenizer_config_attr, "to_dict") and callable(getattr(tokenizer_config_attr, "to_dict")):
                  tokenizer_config_dict = tokenizer_config_attr.to_dict()
             elif tokenizer_config_attr is not None and hasattr(tokenizer_config_attr, "model_dump") and callable(getattr(tokenizer_config_attr, "model_dump")):
                 tokenizer_config_dict = tokenizer_config_attr.model_dump(mode='json')
             else:
                 if tokenizer_config_attr is not None:
                      logger.warning(f"Could not automatically serialize tokenizer config (type: {type(tokenizer_config_attr)}). Not saving in .pt file.")
         except Exception as e:
             logger.warning(f"Failed to serialize tokenizer config: {e}", exc_info=True)

         checkpoint = {
             ARTIFACT_STATE_DICT_KEY: model.state_dict(),
             ARTIFACT_CONFIG_KEY: model_config_dict,
             TOKENIZER_CONFIG_KEY: tokenizer_config_dict, # Store if available
             **kwargs
         }

         try:
             torch.save(checkpoint, path)
             logger.info(f"Legacy model artifact saved to {path}")
             logger.warning("Tokenizer files must be saved separately (e.g., in ./tokenizer/) when using 'pt_legacy' format.")
         except Exception as e:
             logger.error(f"Failed to save legacy artifact to {path}: {e}", exc_info=True)
             raise RuntimeError(f"Failed to save legacy artifact to {path}") from e
    else:
         raise ValueError(f"Unsupported save_format: {save_format}. Choose 'directory' or 'pt_legacy'.")

def load_model_for_inference(
    path: Union[str, Path],
    device: Optional[Union[torch.device, str]] = None,
    strict_load: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Model, Tokenizer]:
    """
    Load a model and its tokenizer for inference or fine-tuning.

    Supports loading from a directory (containing config.json, model weights, 
    tokenizer files) or a legacy .pt artifact (requires tokenizer files 
    in a sibling 'tokenizer/' directory).

    Args:
        path: Path to the directory or legacy .pt artifact file.
        device: Device to load the model onto ('cpu', 'cuda', etc.). Auto-detects if None.
        strict_load: Whether to strictly enforce state_dict key matching for the model.
        config_overrides: Optional dictionary to override loaded model config values.

    Returns:
        A tuple containing the instantiated (Model, Tokenizer).

    Raises:
        FileNotFoundError: If the path or required files do not exist.
        KeyError: If required keys are missing in a legacy .pt artifact.
        Exception: For underlying instantiation or state loading errors.
    """
    input_path = Path(path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Model/tokenizer path not found: {input_path}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model: Optional[Model] = None
    tokenizer: Optional[Tokenizer] = None

    if input_path.is_dir():
        logger.info(f"Loading model and tokenizer from directory: {input_path}")
        config_path = input_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config file 'config.json' not found in {input_path}")

        # Load Model
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            if config_overrides:
                logger.info(f"Applying model config overrides: {config_overrides}")
                config_dict.update(config_overrides)

            logger.info(f"Instantiating model from {config_path}...")
            model_config = OmegaConf.create(config_dict)
            created_model = create_model(model_config, tokenizer=None)
            model = cast(Model, created_model)

            weights_path = None
            if (input_path / "model.safetensors").is_file():
                 weights_path = input_path / "model.safetensors"
                 logger.warning("Safetensors loading not yet implemented, falling back to .bin")
                 weights_path = None
            
            if weights_path is None:
                if (input_path / "pytorch_model.bin").is_file():
                    weights_path = input_path / "pytorch_model.bin"
                elif (input_path / "model.bin").is_file():
                     weights_path = input_path / "model.bin"
                else:
                    raise FileNotFoundError(f"Model weights file (.safetensors, pytorch_model.bin, model.bin) not found in {input_path}")

            logger.info(f"Loading model weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict_load)
            if missing_keys: logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys: {unexpected_keys}")
            logger.info(f"Model {type(model).__name__} loaded successfully from directory.")

        except Exception as e:
            logger.error(f"Failed to load model from directory {input_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed loading model from {input_path}") from e

        # Load Tokenizer
        try:
             logger.info(f"Attempting to load tokenizer from directory {input_path}...")
             created_tokenizer = create_tokenizer(input_path)
             tokenizer = cast(Tokenizer, created_tokenizer)
             if tokenizer is None:
                 raise RuntimeError(f"create_tokenizer returned None for path {input_path}")
             logger.info(f"Tokenizer {type(tokenizer).__name__} loaded successfully from directory.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from directory {input_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed loading tokenizer from {input_path}") from e

    elif input_path.is_file() and input_path.suffix in [".pt", ".pth"]:
        logger.warning(f"Loading model from legacy .pt file: {input_path}. 'directory' format is preferred.")
        # Load Model from .pt
        try:
            logger.warning(f"Loading legacy artifact {input_path} with weights_only=False. Ensure source is trusted.")
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

            if ARTIFACT_CONFIG_KEY not in checkpoint:
                raise KeyError(f"Legacy artifact {input_path} missing '{ARTIFACT_CONFIG_KEY}'.")
            if ARTIFACT_STATE_DICT_KEY not in checkpoint:
                raise KeyError(f"Legacy artifact {input_path} missing '{ARTIFACT_STATE_DICT_KEY}'.")

            loaded_config_dict = checkpoint[ARTIFACT_CONFIG_KEY]
            state_dict = checkpoint[ARTIFACT_STATE_DICT_KEY]

            if not isinstance(loaded_config_dict, dict):
                raise TypeError("Loaded model_config is not a dictionary.")

            if config_overrides:
                logger.info(f"Applying model config overrides: {config_overrides}")
                loaded_config_dict.update(config_overrides)

            logger.info(f"Instantiating model from config in {input_path}...")
            model_config = OmegaConf.create(loaded_config_dict)
            created_model = create_model(model_config, tokenizer=None)
            model = cast(Model, created_model)

            logger.info(f"Loading state dict from {input_path}...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict_load)
            if missing_keys: logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys: {unexpected_keys}")
            logger.info(f"Model {type(model).__name__} loaded successfully from .pt file.")

        except Exception as e:
            logger.error(f"Failed to load model from legacy artifact {input_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed loading model from {input_path}") from e

        # Load Tokenizer from Sibling Directory or Checkpoint Config
        tokenizer_dir = input_path.parent / "tokenizer"
        tokenizer_source: Union[Path, Dict[str, Any], None] = None
        tokenizer_config_dict = checkpoint.get(TOKENIZER_CONFIG_KEY)

        if isinstance(tokenizer_config_dict, dict):
             logger.info(f"Using tokenizer config found within {input_path}.")
             tokenizer_source = tokenizer_config_dict
        elif tokenizer_dir.is_dir():
            logger.info(f"Attempting to load tokenizer from expected directory for legacy format: {tokenizer_dir}")
            tokenizer_source = tokenizer_dir # Pass Path object
        else:
             raise FileNotFoundError(f"Expected tokenizer directory {tokenizer_dir} not found for legacy artifact, and no config in artifact.")

        try:
            logger.info(f"Loading tokenizer with source type: {type(tokenizer_source)}")
            if tokenizer_source is not None:
                created_tokenizer = create_tokenizer(tokenizer_source)
                tokenizer = cast(Tokenizer, created_tokenizer)
                if tokenizer is None:
                    raise RuntimeError(f"create_tokenizer returned None for legacy source {tokenizer_source}")
                logger.info(f"Tokenizer {type(tokenizer).__name__} loaded successfully.")
            else:
                # Should not happen based on logic above, but satisfy mypy
                raise ValueError("Tokenizer source could not be determined for legacy artifact.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for legacy artifact {input_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed loading tokenizer for {input_path}") from e
    else:
        raise ValueError(f"Unsupported path type or suffix for loading: {input_path}. Expected directory or .pt/.pth file.")

    # Final checks
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load model or tokenizer from path {input_path}. Check logs for details.")

    # Move model to the target device and return
    model.to(device)
    logger.info(f"Model and tokenizer loaded from {input_path}. Model moved to {device}.")
    return model, tokenizer 