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
from typing import Optional, Dict, Any, List, Union, Tuple, Type
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
            tokenizer = instantiate(data_cfg_node.tokenizer)
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
    model_pydantic_dict = OmegaConf.to_container(model_pydantic_config_node, resolve=True)
    if not isinstance(model_pydantic_dict, dict):
        raise TypeError(f"Resolved model config ('model.config') is not a dictionary, got {type(model_pydantic_dict)}.")

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
             model = ModelClass(**model_pydantic_dict)

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
    return model

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
                      dataset_instance = instantiate(dataset_cfg, tokenizer=tokenizer) # Pass tokenizer explicitly
                      logger.debug(f"Instantiated dataset {DatasetClass.__name__} with tokenizer.")
                 else:
                      dataset_instance = instantiate(dataset_cfg)
                      logger.debug(f"Instantiated dataset {DatasetClass.__name__} (tokenizer not in signature).")

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
            dataloader_params = OmegaConf.to_container(dataloader_cfg, resolve=True)
            if not isinstance(dataloader_params, dict):
                raise TypeError(f"Resolved dataloader config for split '{split}' is not a dict.")
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
            dataloader_instance = DataLoaderClass(dataset=dataset_instance, **dataloader_params)
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

        return dataloader_instance

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
    """Instantiates the optimizer."""
    logger.info("Instantiating optimizer...")
    if not optimizer_cfg_node:
        raise ValueError("Optimizer configuration ('optimizer:') is missing.")
    try:
        # Ensure params argument is passed correctly
        optimizer = instantiate(optimizer_cfg_node, params=model.parameters())
        logger.info(f"Instantiated optimizer: {type(optimizer).__name__}")
        return optimizer
    except Exception as e:
        logger.error(f"Failed to instantiate optimizer: {e}", exc_info=True)
        raise

# --- Scheduler ---

def initialize_scheduler(
    scheduler_cfg_node: Optional[DictConfig],
    optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Instantiates the learning rate scheduler, if configured."""
    if scheduler_cfg_node:
        logger.info("Instantiating scheduler...")
        try:
            scheduler = instantiate(scheduler_cfg_node, optimizer=optimizer)
            logger.info(f"Instantiated scheduler: {type(scheduler).__name__}")
            return scheduler
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
    """Instantiates the CheckpointManager if configured."""
    if not checkpoint_cfg_node:
        logger.warning("No checkpointing configuration found ('checkpointing:'). Checkpoints will not be saved.")
        return None

    logger.info("Instantiating CheckpointManager...")
    try:
        # Create a mutable copy to allow modification (e.g., resolving paths)
        mutable_ckpt_cfg = OmegaConf.create(OmegaConf.to_container(checkpoint_cfg_node, resolve=True) or {})
        if not isinstance(mutable_ckpt_cfg, DictConfig):
            logger.error(f"Resolved checkpoint config is not a DictConfig: {type(mutable_ckpt_cfg)}. Cannot instantiate CheckpointManager.")
            return None

        # Resolve checkpoint_dir relative to Hydra run directory if not absolute
        ckpt_dir_path = mutable_ckpt_cfg.get('checkpoint_dir')
        hydra_run_dir = None
        try:
             hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
             hydra_run_dir = Path(hydra_cfg.runtime.output_dir)
        except Exception:
             logger.warning("Could not get Hydra run directory. Relative checkpoint paths may not resolve correctly.")

        if ckpt_dir_path and not os.path.isabs(ckpt_dir_path) and hydra_run_dir:
            resolved_path = hydra_run_dir / ckpt_dir_path
            mutable_ckpt_cfg.checkpoint_dir = str(resolved_path)
            logger.info(f"Resolved relative checkpoint_dir to: {resolved_path}")
        elif not mutable_ckpt_cfg.get('checkpoint_dir') and hydra_run_dir:
            # Default if missing
            default_ckpt_dir = hydra_run_dir / "checkpoints"
            mutable_ckpt_cfg.checkpoint_dir = str(default_ckpt_dir)
            logger.warning(f"'checkpoint_dir' not found, defaulting to Hydra run output: {default_ckpt_dir}")
        elif not mutable_ckpt_cfg.get('checkpoint_dir') and not hydra_run_dir:
             logger.error("Checkpoint directory is not specified and Hydra run directory could not be determined. Cannot initialize CheckpointManager.")
             return None

        # Convert full app config to primitive dict for CheckpointManager to save
        config_to_save = OmegaConf.to_container(full_app_config, resolve=True)
        if not isinstance(config_to_save, (dict, list)): # Allow list for some root configs? Dict is safer.
             logger.error(f"Failed to convert full app config to dict/list for saving. Got type: {type(config_to_save)}. Saving None for config.")
             config_to_save = None

        manager = instantiate(
            mutable_ckpt_cfg, # Use potentially updated mutable config
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            callbacks=callbacks, # Pass CallbackList object
            tokenizer=tokenizer,
            experiment_name=experiment_name,
        )
        # Ensure the directory exists after instantiation
        if hasattr(manager, 'checkpoint_dir') and manager.checkpoint_dir:
            try:
                Path(manager.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured CheckpointManager directory exists: {manager.checkpoint_dir}")
            except Exception as dir_err:
                 logger.error(f"Failed to create checkpoint directory {manager.checkpoint_dir}: {dir_err}")
        logger.info(f"Instantiated CheckpointManager.")
        return manager

    except Exception as e:
         logger.error(f"Failed to instantiate CheckpointManager: {e}", exc_info=True)
         # Decide: raise or return None? Raising is probably safer.
         raise

# --- Evaluator ---

def initialize_evaluator(
    eval_cfg_node: Optional[DictConfig],
    model: nn.Module,
    val_dataloader: Optional[DataLoader],
    device: torch.device,
    use_amp: bool,
    callbacks: CallbackList # Pass CallbackList obj
) -> Optional[Evaluator]:
    """Instantiates the Evaluator if a validation dataloader exists."""
    if not val_dataloader:
        logger.info("No validation dataloader provided, Evaluator will not be instantiated.")
        return None

    logger.info("Validation dataloader found, instantiating Evaluator...")
    try:
        # Prepare config dict for Evaluator (can be empty)
        eval_config_dict = OmegaConf.to_container(eval_cfg_node or {}, resolve=True)
        if not isinstance(eval_config_dict, dict):
             logger.warning(f"Evaluation config resolved to non-dict type ({type(eval_config_dict)}). Passing empty dict to Evaluator.")
             eval_config_dict = {}

        evaluator = Evaluator(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            config=eval_config_dict, # Pass resolved primitive dict
            use_amp=use_amp,
            callbacks=callbacks # Pass CallbackList object
        )
        logger.info(f"Instantiated Evaluator: {type(evaluator).__name__}")
        return evaluator
    except Exception as e:
         logger.error(f"Failed to instantiate Evaluator: {e}", exc_info=True)
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
    """
    Compiles the model using torch.compile if enabled and available.
    Intended to be called *after* model initialization and potential checkpoint loading.
    """
    if not compile_flag:
        logger.info("Model compilation is disabled.")
        return model
    try:
        # Check PyTorch version compatibility if necessary
        # Assume >= 2.0 for now
        logger.info("Attempting model compilation with torch.compile...")
        compile_options_dict = OmegaConf.to_container(compile_options_cfg or {}, resolve=True)
        if not isinstance(compile_options_dict, dict):
             logger.warning(f"Compile options resolved to non-dict type ({type(compile_options_dict)}). Using default compile options.")
             compile_options_dict = {}

        compiled_model = torch.compile(model, **compile_options_dict) # type: ignore
        logger.info("Model compilation successful.")
        return compiled_model
    except ImportError:
        logger.warning("torch.compile requires PyTorch 2.0 or later. Skipping compilation.")
        return model # Return original model
    except Exception as e:
        logger.warning(f"Failed to compile model: {e}. Proceeding without compilation.", exc_info=True)
        return model # Return original model 