#!/usr/bin/env python
"""
Initialization Utilities for Trainer Components
==============================================

This module provides helper functions to instantiate various components
(model, dataloaders, optimizer, etc.) required by the Trainer,
based on Hydra configuration objects. This helps keep the Trainer's
__init__ method cleaner and focused on orchestration.
"""

import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, Type, cast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_class
from pathlib import Path
import inspect # For checking function signatures
import time
import hashlib
import torch.amp # Add import for torch.amp
from torch.utils.data import Dataset

# Ensure relative imports work correctly within the package
try:
    from .checkpointing import CheckpointManager
    from .evaluation import Evaluator
    from .callbacks import CallbackList, Callback
    from ..data.tokenizers.base import Tokenizer
    from ..utils.common import setup_device
except ImportError:
    # Handle cases where the script might be run directly or structure changes
    # This might occur during testing or if the file is moved.
    # For robustness, you might add alternative import paths or error handling.
    print("Warning: Relative imports failed in initialization.py. Ensure structure is correct.")
    # As a fallback, you might try absolute imports if your project structure supports it
    # from craft.training.checkpointing import CheckpointManager
    # from craft.training.evaluation import Evaluator
    # ... etc.
    # If imports fail critically, reraise or handle appropriately
    raise

logger = logging.getLogger(__name__)

# --- Device Setup ---

def initialize_device(device_pref: str) -> torch.device:
    """Sets up the computation device based on preference and availability."""
    logger.info(f"Setting up device from preference: {device_pref}")
    device = setup_device(device_pref) # Use utility function
    logger.info(f"Device set to: {device}")
    return device

# --- Tokenizer ---

def initialize_tokenizer(data_cfg_node: Optional[DictConfig]) -> Optional[Tokenizer]:
    """Instantiates the tokenizer if configured."""
    if data_cfg_node and data_cfg_node.get("tokenizer"):
        logger.info("Instantiating tokenizer...")
        try:
            tokenizer = cast(Optional[Tokenizer], instantiate(data_cfg_node.tokenizer))
            if tokenizer:
                 logger.info(f"Instantiated tokenizer: {type(tokenizer).__name__}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to instantiate tokenizer: {e}", exc_info=True)
            raise
    else:
        logger.info("No tokenizer configuration found in data config.")
        return None

# --- Model ---

def initialize_model(
    model_cfg_node: DictConfig,
    device: torch.device,
    tokenizer: Optional[Tokenizer] = None
) -> nn.Module:
    """Instantiates the model, potentially injecting vocab size from tokenizer."""
    logger.info("Instantiating model...")
    if not model_cfg_node:
        raise ValueError("Model configuration ('model:') is missing.")

    model_target = model_cfg_node.get('_target_')
    model_pydantic_config_node = model_cfg_node.get('config')
    if not model_target or not model_pydantic_config_node:
        raise ValueError("Model config must contain '_target_' and a nested 'config:' block.")

    # Instantiate the Pydantic config first by converting to dict
    model_pydantic_dict_any = OmegaConf.to_container(model_pydantic_config_node, resolve=True)
    # Ensure it's a Dict[str, Any] for **kwargs
    if not isinstance(model_pydantic_dict_any, dict):
        raise TypeError(f"Resolved model config ('model.config') is not a dictionary, got {type(model_pydantic_dict_any)}.")
    model_pydantic_dict: Dict[str, Any] = cast(Dict[str, Any], model_pydantic_dict_any)

    # Check for vocab size injection possibility
    needs_vocab_injection = (
        tokenizer and
        model_pydantic_dict.get('vocab_size') is None and
        hasattr(tokenizer, 'get_vocab_size')
    )
    if needs_vocab_injection:
        try:
            tokenizer_vocab_size = tokenizer.get_vocab_size() # type: ignore
            if tokenizer_vocab_size:
                model_pydantic_dict['vocab_size'] = tokenizer_vocab_size
                logger.info(f"Injecting vocab_size={tokenizer_vocab_size} from tokenizer into model config dict.")
            else:
                 logger.warning("Tokenizer returned invalid vocab_size. Model config might be incomplete.")
        except Exception as e_vocab:
             logger.warning(f"Could not get vocab_size from tokenizer: {e_vocab}")

    # Instantiate the actual model using the config dict
    try:
        ModelClass = get_class(model_target)
        # Check signature: does it take 'config=' or '**kwargs'?
        sig = inspect.signature(ModelClass.__init__)
        if 'config' in sig.parameters:
             logger.debug(f"Instantiating model {ModelClass.__name__} using config= argument.")
             model = ModelClass(config=model_pydantic_dict)
        else:
             logger.debug(f"Instantiating model {ModelClass.__name__} using **kwargs.")
             # General type ignore for unsound __init__ access
             model = ModelClass(**model_pydantic_dict) # type: ignore

        logger.info(f"Instantiated model: {type(model).__name__}")
    except TypeError as te:
         # This might catch cases where neither config= nor **kwargs works as expected
         logger.error(f"TypeError during model instantiation ({ModelClass.__name__}): {te}. Check __init__ signature and config.", exc_info=True)
         raise
    except Exception as model_init_e:
        logger.error(f"Failed to instantiate model '{model_target}' with config: {model_init_e}", exc_info=True)
        raise

    model.to(device)
    logger.info(f"Moved model to device: {device}")
    # Cast the return value to nn.Module
    return cast(nn.Module, model)

# --- Dataloaders ---

def _instantiate_single_dataloader(
    split_cfg_node: Optional[DictConfig], # Allow None for optional splits like 'val'
    data_cfg_node: DictConfig, # For top-level params like batch_size, num_workers
    split: str,
    device: torch.device,
    tokenizer: Optional[Tokenizer] = None
) -> Optional[DataLoader]:
    """Helper to instantiate a dataloader for a single split ('train' or 'val')."""
    logger.debug(f"Attempting to instantiate dataloader for split: {split}")
    if not split_cfg_node:
        logger.debug(f"No configuration found for dataset split: {split}")
        return None

    try:
        dl_kwargs = {"tokenizer": tokenizer} if tokenizer else {}
        dataset_cfg = split_cfg_node.get('dataset')
        dataloader_cfg = split_cfg_node.get('dataloader')

        # --- Instantiate Dataset ---
        if dataset_cfg and dataset_cfg.get('_target_'):
            # Inject tokenizer if dataset's __init__ accepts it
            try:
                 DatasetClass = get_class(dataset_cfg._target_)
                 sig = inspect.signature(DatasetClass.__init__)
                 if 'tokenizer' in sig.parameters:
                      init_args = {"tokenizer": tokenizer}
                 else:
                      init_args = {}

            except Exception as e_dataset:
                 logger.error(f"Failed to instantiate dataset for split '{split}': {e_dataset}", exc_info=True)
                 return None # Cannot proceed without dataset
        else:
            logger.error(f"Dataset configuration missing or invalid for split '{split}'. Needs 'dataset._target_'.")
            return None

        logger.debug(f"Instantiated dataset for split '{split}': {type(dataset_instance).__name__}")


        # --- Instantiate DataLoader ---
        if dataloader_cfg and dataloader_cfg.get('_target_'):
            # Use explicit dataloader config
            dataloader_params_any = OmegaConf.to_container(dataloader_cfg, resolve=True)
            # Ensure it's a Dict[str, Any]
            if not isinstance(dataloader_params_any, dict):
                raise TypeError(f"Resolved dataloader config for split '{split}' is not a dict.")
            dataloader_params: Dict[str, Any] = cast(Dict[str, Any], dataloader_params_any)

            _target_ = dataloader_params.pop('_target_', None) # Remove target before passing as kwargs

            # Apply defaults from top-level config if not present in split-specific config
            dataloader_params.setdefault('batch_size', data_cfg_node.get('batch_size', 1))
            dataloader_params.setdefault('num_workers', data_cfg_node.get('num_workers', 0))
            dataloader_params.setdefault('pin_memory', (device.type == 'cuda')) # Default based on device

            # Shuffle logic: default True for train, False otherwise
            if split == 'train':
                dataloader_params.setdefault('shuffle', True)
            else:
                dataloader_params.setdefault('shuffle', False)

            # Instantiate the dataloader class
            if not _target_:
                 raise ValueError(f"Missing '_target_' in dataloader config for split '{split}'.")
            DataLoaderClass = get_class(_target_)
            # General type ignore for unsound __init__ access
            dataloader_instance = DataLoaderClass(dataset=dataset_instance, **dataloader_params) # type: ignore
            logger.debug(f"Instantiated explicit dataloader for split '{split}': {type(dataloader_instance).__name__}")

        else:
            # No explicit dataloader config, wrap dataset in default torch.utils.data.DataLoader
            batch_size = data_cfg_node.get('batch_size', 1)
            num_workers = data_cfg_node.get('num_workers', 0)
            shuffle = (split == 'train')
            pin_memory = (device.type == 'cuda')
            logger.info(f"Wrapping dataset for split '{split}' in default DataLoader (batch={batch_size}, workers={num_workers}, shuffle={shuffle}, pin_memory={pin_memory}).")
            dataloader_instance = DataLoader(
                dataset=dataset_instance, batch_size=batch_size,
                num_workers=num_workers, shuffle=shuffle,
                pin_memory=pin_memory
            )

        # Cast the return value
        return cast(Optional[DataLoader], dataloader_instance)

    except Exception as e:
        logger.error(f"Failed to instantiate dataloader components for split '{split}': {e}", exc_info=True)
        return None


def initialize_dataloaders(
    data_cfg_node: DictConfig,
    device: torch.device,
    tokenizer: Optional[Tokenizer] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Instantiates train and validation dataloaders."""
    logger.info("Instantiating dataloaders...")
    if not data_cfg_node or not data_cfg_node.get("datasets"):
        raise ValueError("Configuration section 'data.datasets:' is missing or empty.")

    train_split_cfg = data_cfg_node.datasets.get('train')
    train_dataloader = _instantiate_single_dataloader(train_split_cfg, data_cfg_node, 'train', device, tokenizer)
    if not train_dataloader:
        # Error should have been logged in _instantiate_single_dataloader
        raise ValueError("Training dataloader instantiation failed. Check logs for details.")
    logger.info(f"Instantiated training dataloader: {type(train_dataloader).__name__}")

    val_split_cfg = data_cfg_node.datasets.get('val')
    val_dataloader = _instantiate_single_dataloader(val_split_cfg, data_cfg_node, 'val', device, tokenizer)
    if val_dataloader:
        logger.info(f"Instantiated validation dataloader: {type(val_dataloader).__name__}")
    else:
        logger.info("No validation dataloader configured or instantiation failed.")

    return train_dataloader, val_dataloader


# --- Optimizer ---

def initialize_optimizer(
    optimizer_cfg_node: DictConfig,
    model: nn.Module
) -> torch.optim.Optimizer:
    """Instantiates the optimizer from its configuration."""
    logger.info(f"Instantiating optimizer ({optimizer_cfg_node.get('_target_', 'N/A')})...")
    try:
        optimizer = instantiate(optimizer_cfg_node, params=model.parameters(), _convert_="partial")
        logger.info(f"Instantiated optimizer: {type(optimizer).__name__}")
        # Cast the return value
        return cast(torch.optim.Optimizer, optimizer)
    except Exception as e:
        logger.error(f"Failed to instantiate optimizer: {e}", exc_info=True)
        raise

# --- Scheduler ---

def initialize_scheduler(
    scheduler_cfg_node: Optional[DictConfig],
    optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Instantiates the learning rate scheduler from its configuration."""
    if scheduler_cfg_node:
        logger.info(f"Instantiating scheduler ({scheduler_cfg_node.get('_target_', 'N/A')})...")
        try:
            scheduler = instantiate(scheduler_cfg_node, optimizer=optimizer, _convert_="partial")
            if scheduler:
                logger.info(f"Instantiated scheduler: {type(scheduler).__name__}")
            # Cast the return value
            return cast(Optional[torch.optim.lr_scheduler._LRScheduler], scheduler)
        except Exception as e:
            logger.error(f"Failed to instantiate scheduler: {e}", exc_info=True)
            # Decide whether to raise or just return None
            logger.warning("Proceeding without scheduler due to instantiation error.")
            return None
    else:
        logger.info("No scheduler configuration found.")
        return None

# --- AMP Scaler ---

def initialize_amp_scaler(
    use_amp: bool,
    device: torch.device
) -> torch.amp.GradScaler:
    """Instantiates the AMP GradScaler."""
    # Ensure we pass device_type='cuda' or 'cpu'
    device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu' # Default to cpu if xla etc.
    if use_amp and device_type != 'cuda':
        logger.warning(f"AMP (use_amp=True) requested but device is {device_type}. AMP is primarily for CUDA. Scaler will be created but likely inactive.")

    # Use the new torch.amp API - device_type is positional
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    logger.info(f"AMP GradScaler initialized (Requested Enabled: {use_amp}, Actual Enabled: {scaler.is_enabled()}, Device Type: {device_type})")
    return scaler

# --- Callbacks ---

def initialize_callbacks(
    callbacks_cfg_node: Optional[DictConfig],
    # trainer: "Trainer" # Pass trainer if callbacks need it at init
) -> List[Callback]:
    """Instantiates a list of callbacks from configuration."""
    logger.info("Instantiating callbacks...")
    callback_list: List[Callback] = []
    if not callbacks_cfg_node:
        logger.info("No callbacks configuration found ('callbacks:').")
        return callback_list

    if isinstance(callbacks_cfg_node, DictConfig):
        for cb_name, cb_conf in callbacks_cfg_node.items():
            # Check if it's a valid config node with a _target_
            if isinstance(cb_conf, DictConfig) and cb_conf.get('_target_'):
                logger.info(f"Instantiating callback: {cb_name} ({cb_conf._target_})")
                try:
                    # Instantiate directly. Assume callbacks get trainer via set_trainer() later if needed.
                    callback_instance = instantiate(cb_conf)
                    if isinstance(callback_instance, Callback):
                        callback_list.append(callback_instance)
                    else:
                        logger.warning(f"Instantiated object for callback '{cb_name}' is not a subclass of Callback ({type(callback_instance)}). Skipping.")
                except Exception as e:
                    logger.error(f"Failed to instantiate callback '{cb_name}': {e}", exc_info=True)
                    # Decide if failure is critical - perhaps continue without the callback?
            else:
                # Log if the entry isn't a valid instantiable config
                if cb_conf is not None: # Avoid logging for explicitly null entries if any
                    logger.warning(f"Skipping invalid callback configuration entry under 'callbacks': {cb_name} = {cb_conf}")
    else:
        logger.warning(f"Expected 'callbacks' config to be a dictionary (DictConfig), got {type(callbacks_cfg_node)}. No callbacks loaded.")

    logger.info(f"Instantiated {len(callback_list)} callbacks.")
    return callback_list


# --- Checkpoint Manager ---

def initialize_checkpoint_manager(
    checkpoint_cfg_node: Optional[DictConfig],
    full_app_config: DictConfig, # Pass the root config for saving
    experiment_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    callbacks: Optional[CallbackList] = None, # Pass CallbackList obj
    tokenizer: Optional[Tokenizer] = None,
) -> Optional[CheckpointManager]:
    """Creates and configures the CheckpointManager from configuration."""
    if not checkpoint_cfg_node:
        logger.info("No checkpointing configuration found ('checkpointing:'). CheckpointManager will not be instantiated.")
        return None

    # Extract checkpointing parameters
    try:
        # Resolve the config node to a primitive dict
        chkpt_params_any = OmegaConf.to_container(checkpoint_cfg_node, resolve=True)
        if not isinstance(chkpt_params_any, dict):
             raise TypeError("Resolved checkpointing config is not a dictionary.")
        chkpt_params: Dict[str, Any] = cast(Dict[str, Any], chkpt_params_any)

        # Determine checkpoint directory path
        checkpoint_dir_str = chkpt_params.get('checkpoint_dir')
        checkpoint_dir: Path

        if checkpoint_dir_str:
            checkpoint_dir = Path(checkpoint_dir_str).resolve()
            logger.info(f"Using checkpoint directory specified in config: {checkpoint_dir}")
        else:
            # Fallback to Hydra output directory
            try:
                hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
                exp_name_for_path = experiment_name or "default_run"
                checkpoint_dir = hydra_output_dir / "checkpoints" / exp_name_for_path
                # General ignore for persistent str-bytes-safe error
                logger.info(f"Checkpoint directory not specified, deriving from Hydra output: {str(checkpoint_dir)}") # type: ignore[str-bytes-safe]
            except Exception as e:
                # General ignore for persistent str-bytes-safe error
                logger.error(f"Could not determine checkpoint directory from Hydra config: {e}. Defaulting to ./checkpoints/<exp_name>.") # type: ignore[str-bytes-safe]
                exp_name_for_path = experiment_name or "default_run"
                # General ignore for persistent str-bytes-safe error
                checkpoint_dir = Path(f"./checkpoints/{exp_name_for_path}").resolve() # type: ignore

        # Ensure directory exists
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_e:
             # General ignore for persistent str-bytes-safe error
             logger.error(f"Failed to create checkpoint directory {str(checkpoint_dir)}: {mkdir_e}") # type: ignore[str-bytes-safe]
             raise # Stop if directory creation fails

        # Prepare arguments for CheckpointManager constructor
        manager_args = {
            "checkpoint_dir": str(checkpoint_dir), # Pass as string
            "keep_last_n": chkpt_params.get('keep_last', 3),
            "keep_best_n": chkpt_params.get('keep_best', 1),
            "save_best_only": chkpt_params.get('save_best_only', False),
            "checkpoint_prefix": chkpt_params.get('checkpoint_prefix', "checkpoint"),
            "experiment_name": experiment_name,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "tokenizer": tokenizer,
            "callbacks": callbacks, # Pass CallbackList instance
            # Pass the *original* full app config for saving state, if CheckpointManager needs it
            # Or potentially pass just the experiment_config part?
            # CheckpointManager uses TrainingState now, which includes config.
            # Let's pass the resolved checkpointing params for now, as CM doesn't need the full app config?
            # Revised: TrainingState expects full config, so pass full_app_config converted
            "config": OmegaConf.to_container(full_app_config, resolve=True)
        }

        # Remove None values if CheckpointManager doesn't handle them gracefully in __init__
        # Example: manager_args = {k: v for k, v in manager_args.items() if v is not None}

        logger.info("Instantiating CheckpointManager...")
        manager = CheckpointManager(**manager_args)
        logger.info("CheckpointManager instantiated successfully.")
        return manager

    except Exception as e:
        logger.error(f"Failed to instantiate CheckpointManager: {e}", exc_info=True)
        # Decide whether to raise or return None
        logger.warning("Proceeding without CheckpointManager due to instantiation error.")
        return None


def initialize_evaluator(
    eval_cfg_node: Optional[DictConfig],
    model: nn.Module,
    val_dataloader: Optional[DataLoader],
    device: torch.device,
    use_amp: bool,
    callbacks: CallbackList # Pass CallbackList obj
) -> Optional[Evaluator]:
    """Instantiates the evaluator if configured and validation dataloader exists."""
    if not val_dataloader:
        logger.info("No validation dataloader provided, Evaluator will not be instantiated.")
        return None

    logger.info("Validation dataloader found, instantiating Evaluator...")
    try:
        eval_params_any = OmegaConf.to_container(eval_cfg_node, resolve=True)
        # Ensure it's a Dict[str, Any]
        if not isinstance(eval_params_any, dict):
            raise TypeError("Resolved evaluator config is not a dictionary.")
        eval_params: Dict[str, Any] = cast(Dict[str, Any], eval_params_any)

        logger.info(f"Instantiating evaluator ({eval_params.get('_target_', 'N/A')})...")
        evaluator_instance = instantiate(
            eval_params,
            model=model,
            dataloader=val_dataloader,
            device=device,
            use_amp=use_amp,
            # Pass the inner list of callbacks if Evaluator expects List[Callback]
            # Or adjust Evaluator's type hint if it can accept CallbackList directly
            callbacks=callbacks.callbacks, # Assuming Evaluator wants List[Callback]
            _convert_="partial"
        )
        logger.info(f"Instantiated evaluator: {type(evaluator_instance).__name__}")
        # Cast the return value
        return cast(Optional[Evaluator], evaluator_instance)

    except Exception as e:
        logger.error(f"Failed to instantiate evaluator: {e}", exc_info=True)
        # Decide whether to raise or just return None and log warning
        logger.warning("Proceeding without Evaluator due to instantiation error.")
        return None

# --- Compile Model Helper ---
# Note: Actual compilation should happen in Trainer *after* component init and potential resume

def compile_model_if_enabled(
    model: nn.Module,
    compile_flag: bool,
    compile_options_cfg: Optional[DictConfig]
) -> nn.Module:
    """Compiles the model using torch.compile if the flag is set."""
    # Handle cases where compile_flag might be None
    if not compile_flag:
        logger.info("torch.compile disabled by flag.")
        return model

    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
        return model

    # Convert OmegaConf options to dict, ensure it resolves
    compile_options: Dict[str, Any] = {}
    if compile_options_cfg:
        try:
            compile_options = OmegaConf.to_container(compile_options_cfg, resolve=True) # type: ignore
            if not isinstance(compile_options, dict):
                 logger.warning("torch_compile_options did not resolve to a dictionary. Using defaults.")
                 compile_options = {}
        except Exception as e:
            logger.warning(f"Failed to resolve torch_compile_options: {e}. Using defaults.")
            compile_options = {}

    try:
        start_time = time.time()
        logger.info(f"Compiling model with options: {compile_options}")
        # Add type ignore for potential mypy issue with torch.compile return type
        compiled_model = torch.compile(model, **compile_options) # type: ignore
        compile_time = time.time() - start_time
        logger.info(f"Model compiled successfully in {compile_time:.2f} seconds.")
        # Return the compiled model
        return compiled_model # type: ignore
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}. Falling back to original model.")
        return model

    except Exception as e:
        logger.warning(f"Failed to compile model: {e}. Proceeding without compilation.", exc_info=True)
        return model # Return original model 