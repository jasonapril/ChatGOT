# src/craft/core/factories.py
import logging
import json # Added for tokenizer config loading
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler

# Assuming these imports are valid based on project structure
from ..data.tokenizers.base import Tokenizer
from ..config.schemas import TrainingConfig, ExperimentConfig, CheckpointingConfig # Added CheckpointingConfig
from ..training.evaluation import Evaluator
from ..training.checkpointing import CheckpointManager
from ..training.callbacks.base import Callback, CallbackList

logger = logging.getLogger(__name__)

def create_tokenizer(cfg: Optional[Union[DictConfig, dict, str, Path]]) -> Optional[Tokenizer]:
    """Instantiates the tokenizer from its configuration or a path."""
    resolved_cfg: Optional[Union[DictConfig, dict]] = None

    if isinstance(cfg, (str, Path)):
        path = Path(cfg).resolve()
        logger.info(f"Attempting to load tokenizer config from directory: {path}")
        tokenizer_config_file = path / "tokenizer_config.json"
        if tokenizer_config_file.is_file():
            try:
                with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                resolved_cfg = OmegaConf.create(config_dict)
                logger.info(f"Loaded tokenizer config from {tokenizer_config_file}")
            except Exception as e:
                logger.warning(f"Failed to load or parse {tokenizer_config_file}: {e}. Will try instantiating tokenizer with path.")
                # Fallback: Maybe the tokenizer class handles path directly?
                # For now, we stick to config-based instantiation primarily.
                # If create_tokenizer is expected to handle paths directly later,
                # this logic needs refinement. We'll set resolved_cfg to None
                # which might lead to failure if no _target_ is present later.
                resolved_cfg = None
        else:
            logger.warning(f"'{tokenizer_config_file}' not found. Cannot load config from path.")
            resolved_cfg = None # Cannot proceed with config-based approach
    elif isinstance(cfg, (DictConfig, dict)):
        resolved_cfg = cfg
    elif cfg is None:
        logger.info("No tokenizer configuration or path provided.")
        return None
    else:
        logger.error(f"Invalid configuration type for tokenizer: {type(cfg)}")
        return None

    if resolved_cfg is None:
         logger.error("Could not resolve a valid tokenizer configuration.")
         return None # Cannot proceed without a config dict/DictConfig

    # Use .get() instead of OmegaConf.select
    target = resolved_cfg.get('_target_')
    if not target:
        logger.warning(f"Tokenizer config {type(resolved_cfg)} is missing '_target_'. Cannot instantiate.")
        return None

    try:
        logger.info(f"Instantiating tokenizer ({target})...")
        tokenizer: Optional[Tokenizer] = hydra.utils.instantiate(resolved_cfg)
        if tokenizer:
            logger.info(f"Tokenizer {type(tokenizer).__name__} instantiated successfully.")
        else:
            logger.warning("Hydra instantiation for tokenizer returned None.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to instantiate tokenizer from config: {e}", exc_info=True)
        raise

# Change return type hint
def create_dataloaders(cfg: Union[DictConfig, dict], tokenizer: Optional[Tokenizer] = None) -> Dict[str, Optional[DataLoader]]:
    """
    Instantiates dataloaders (train, val, test) using Hydra, potentially via a factory.
    Returns a dict like {'train': DataLoader, 'val': Optional[DataLoader], 'test': Optional[DataLoader]}.
    """
    if not cfg:
        raise ValueError("Data configuration is required to create dataloaders.")
    # Use .get() instead of OmegaConf.select
    target = cfg.get('_target_')
    if not target:
         raise ValueError("Data config is present but missing '_target_'. Cannot instantiate dataloaders automatically.")

    try:
        logger.info(f"Instantiating dataloaders via target: {target}...")
        dataloaders_dict = hydra.utils.instantiate(
            cfg,
            tokenizer=tokenizer,
            _convert_="partial"
        )

        if not isinstance(dataloaders_dict, dict):
            raise TypeError(f"Instantiating data config did not return a dictionary of dataloaders, got {type(dataloaders_dict)}")

        # Ensure keys exist, even if value is None
        train_loader = dataloaders_dict.get('train')
        val_loader = dataloaders_dict.get('val')
        test_loader = dataloaders_dict.get('test')

        if not train_loader:
            raise ValueError("Dataloader instantiation did not produce a 'train' dataloader.")
        if not isinstance(train_loader, DataLoader):
             raise TypeError(f"Instantiated 'train' object is not a DataLoader, got {type(train_loader)}")
        logger.info("Train dataloader instantiated.")

        if val_loader:
            if not isinstance(val_loader, DataLoader):
                 raise TypeError(f"Instantiated 'val' object is not a DataLoader, got {type(val_loader)}")
            logger.info("Validation dataloader instantiated.")
        else:
            logger.info("Validation dataloader not found or configured.")

        if test_loader:
             if not isinstance(test_loader, DataLoader):
                 raise TypeError(f"Instantiated 'test' object is not a DataLoader, got {type(test_loader)}")
             logger.info("Test dataloader instantiated.")

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    except Exception as e:
        logger.error(f"Failed to instantiate dataloaders from config: {e}", exc_info=True)
        raise ValueError("Dataloader instantiation failed.") from e

def create_model(cfg: Union[DictConfig, dict], tokenizer: Optional[Tokenizer] = None) -> nn.Module:
    """Instantiates the model from its configuration, injecting vocab size if needed."""
    if not cfg:
        raise ValueError("Model configuration is required.")
    # Use .get() instead of OmegaConf.select
    target = cfg.get('_target_')
    if not target:
         raise ValueError("Model config is present but missing '_target_'. Cannot instantiate model.")

    try:
        target_cls_str = str(target)
        requires_vocab_size = "LanguageModel" in target_cls_str or "TransformerModel" in target_cls_str # Example check
        if requires_vocab_size and tokenizer and not cfg.get('vocab_size'):
            vocab_size = tokenizer.get_vocab_size()
            logger.info(f"Injecting vocab_size={vocab_size} from tokenizer into model config.")
            cfg_copy = OmegaConf.to_container(cfg, resolve=True) # type: ignore
            if isinstance(cfg_copy, dict):
                cfg_copy['vocab_size'] = vocab_size
                cfg = OmegaConf.create(cfg_copy) # type: ignore
            else:
                 logger.warning("Could not inject vocab_size: config did not resolve to a dict.")

        logger.info(f"Instantiating model ({target})...")
        model: nn.Module = hydra.utils.instantiate(cfg, _convert_="partial")

        if not isinstance(model, nn.Module):
             raise TypeError(f"Instantiated model is not an nn.Module, got {type(model)}")

        logger.info(f"Model {type(model).__name__} instantiated successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to instantiate model from config: {e}", exc_info=True)
        raise ValueError("Model instantiation failed.") from e

def create_optimizer(cfg: Union[DictConfig, dict], model: nn.Module) -> Optimizer:
    """Instantiates the optimizer from its configuration."""
    if not cfg:
        raise ValueError("Optimizer configuration is required.")
    # Use .get() instead of OmegaConf.select
    target = cfg.get('_target_')
    if not target:
         raise ValueError("Optimizer config is present but missing '_target_'. Cannot instantiate optimizer.")

    try:
        logger.info(f"Instantiating optimizer ({target})...")
        optimizer: Optimizer = hydra.utils.instantiate(cfg, params=model.parameters(), _convert_="partial")

        if not isinstance(optimizer, Optimizer):
             raise TypeError(f"Instantiated optimizer is not a torch.optim.Optimizer, got {type(optimizer)}")

        logger.info(f"Optimizer {type(optimizer).__name__} instantiated successfully.")
        return optimizer
    except Exception as e:
        logger.error(f"Failed to instantiate optimizer from config: {e}", exc_info=True)
        raise ValueError("Optimizer instantiation failed.") from e

def create_scheduler(cfg: Optional[Union[DictConfig, dict]], optimizer: Optimizer) -> Optional[_LRScheduler]:
    """Instantiates the learning rate scheduler from its configuration."""
    if not cfg:
        logger.info("No scheduler configuration provided.")
        return None
    # Use .get() instead of OmegaConf.select
    target = cfg.get('_target_')
    if not target:
        logger.warning("Scheduler config is present but missing '_target_'. Cannot instantiate.")
        return None
    try:
        logger.info(f"Instantiating scheduler ({target})...")
        scheduler: Optional[_LRScheduler] = hydra.utils.instantiate(cfg, optimizer=optimizer, _convert_="partial")

        if scheduler and not isinstance(scheduler, _LRScheduler):
             raise TypeError(f"Instantiated scheduler is not a torch.optim.lr_scheduler._LRScheduler, got {type(scheduler)}")

        if scheduler:
            logger.info(f"Scheduler {type(scheduler).__name__} instantiated successfully.")
        else:
             logger.info("Scheduler instantiation returned None (config might resolve to null).")
        return scheduler
    except Exception as e:
        logger.error(f"Failed to instantiate scheduler from config: {e}", exc_info=True)
        raise ValueError("Scheduler instantiation failed.") from e

def create_callbacks(cfg: Optional[Union[ListConfig, list, DictConfig, dict]]) -> List[Callback]:
    """Instantiates a list of callbacks from configuration."""
    callbacks_list: List[Callback] = []
    if not cfg:
        logger.info("No callbacks configuration provided.")
        return callbacks_list

    if isinstance(cfg, (DictConfig, dict)):
        logger.info("Instantiating callbacks from dictionary config...")
        for name, callback_cfg in cfg.items():
            if not callback_cfg:
                logger.warning(f"Callback config for '{name!r}' is null, skipping.")
                continue
            # Use .get() on callback_cfg
            target = callback_cfg.get('_target_') if isinstance(callback_cfg, (dict, DictConfig)) else None
            if target:
                try:
                    logger.info(f"  Instantiating callback '{name!r}' ({target})...")
                    cb_instance: Callback = hydra.utils.instantiate(callback_cfg)
                    if not isinstance(cb_instance, Callback):
                        logger.warning(f"Instantiated object for '{name!r}' is not a Callback subclass, skipping.")
                        continue
                    callbacks_list.append(cb_instance)
                    logger.info(f"  Callback {type(cb_instance).__name__} instantiated successfully.")
                except Exception as e:
                    logger.error(f"Failed to instantiate callback '{name!r}' from config: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping callback '{name!r}': Invalid configuration or missing '_target_'.")

    elif isinstance(cfg, (ListConfig, list)):
         logger.info("Instantiating callbacks from list config...")
         for i, callback_cfg in enumerate(cfg):
            if not callback_cfg:
                 logger.warning(f"Callback config at index {i} is null, skipping.")
                 continue
            # Use .get() on callback_cfg
            target = callback_cfg.get('_target_') if isinstance(callback_cfg, (dict, DictConfig)) else None
            if target:
                 try:
                     logger.info(f"  Instantiating callback at index {i} ({target})...")
                     cb_instance_list: Callback = hydra.utils.instantiate(callback_cfg)
                     if not isinstance(cb_instance_list, Callback):
                         logger.warning(f"Instantiated object at index {i} is not a Callback subclass, skipping.")
                         continue
                     callbacks_list.append(cb_instance_list)
                     logger.info(f"  Callback {type(cb_instance_list).__name__} instantiated successfully.")
                 except Exception as e:
                     logger.error(f"Failed to instantiate callback at index {i} from config: {e}", exc_info=True)
            else:
                 logger.warning(f"Skipping callback at index {i}: Invalid configuration or missing '_target_'.")
    else:
        logger.warning(f"Invalid callbacks configuration type: {type(cfg)}. Expected Dict, List, DictConfig, or ListConfig.")

    logger.info(f"Instantiated {len(callbacks_list)} callbacks.")
    return callbacks_list

# --- Add factories for Evaluator, CheckpointManager, GradScaler etc. following similar patterns ---

def create_grad_scaler(amp_enabled: bool, amp_dtype: str = "float16") -> Optional[GradScaler]:
    """Creates a GradScaler if AMP is enabled."""
    if not amp_enabled:
        logger.info("AMP is disabled, GradScaler not created.")
        return None

    scaler_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16 if amp_dtype == "bfloat16" else None
    if scaler_dtype is None:
         logger.warning(f"Unsupported amp_dtype '{amp_dtype}' for GradScaler. Defaulting to float16.")
         scaler_dtype = torch.float16 # Should this default? Maybe error?

    logger.info(f"Creating GradScaler with dtype {scaler_dtype}...")
    return GradScaler(enabled=True) # Note: dtype setting isn't direct in constructor

# Remove config param, adjust Evaluator call
def create_evaluator(
    cfg: Union[DictConfig, dict],
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    tokenizer: Optional[Tokenizer] = None,
) -> Evaluator:
    """Creates the Evaluator instance."""
    if not cfg:
        raise ValueError("Evaluation configuration ('eval') is required to create an Evaluator.")
    # Use .get() instead of OmegaConf.select
    target = cfg.get('_target_')
    if not target:
         raise ValueError("Evaluation config ('eval') is present but missing '_target_'. Cannot instantiate Evaluator automatically.")

    logger.info(f"Instantiating Evaluator ({target})...")
    try:
        evaluator: Evaluator = hydra.utils.instantiate(
            cfg,
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            tokenizer=tokenizer,
            _convert_="partial"
        )
        if not isinstance(evaluator, Evaluator):
            raise TypeError(f"Instantiated object is not an Evaluator, got {type(evaluator)}")
        logger.info(f"Evaluator {type(evaluator).__name__} instantiated successfully.")
        return evaluator
    except Exception as e:
        logger.error(f"Failed to instantiate Evaluator from config: {e}", exc_info=True)
        raise ValueError("Evaluator instantiation failed.") from e

def create_checkpoint_manager(
    experiment_config: ExperimentConfig,
    model: nn.Module,
    optimizer: Optimizer,
    callbacks: Optional[List[Callback]] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> CheckpointManager:
    """Creates and configures the CheckpointManager."""
    # Access the checkpointing config, may be None
    chkpt_cfg: Optional[CheckpointingConfig] = experiment_config.checkpointing

    # --- Determine Checkpoint Directory --- #
    checkpoint_dir_str: Optional[str] = None
    if chkpt_cfg and chkpt_cfg.checkpoint_dir:
        checkpoint_dir_str = chkpt_cfg.checkpoint_dir

    checkpoint_dir: Path
    if checkpoint_dir_str:
        checkpoint_dir = Path(checkpoint_dir_str).resolve()
        logger.info(f"Using checkpoint directory specified in checkpointing config: {checkpoint_dir}")
    else:
        # Fallback logic using Hydra output directory
        try:
            hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            exp_name_for_path = experiment_config.experiment_name or "default_run"
            checkpoint_dir = hydra_output_dir / "checkpoints" / exp_name_for_path
            logger.info(f"Checkpoint directory not specified, deriving from Hydra output: {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Could not determine checkpoint directory from Hydra config: {e}. Checkpointing might fail.")
            # Assign a default path and let CheckpointManager potentially handle errors
            checkpoint_dir = Path("./checkpoints_undetermined").resolve()
            logger.warning(f"Defaulting undetermined checkpoint directory to: {checkpoint_dir}")

    # --- Extract other parameters from config (use defaults if chkpt_cfg is None) --- #
    keep_last_n = chkpt_cfg.keep_last if chkpt_cfg and chkpt_cfg.keep_last is not None else 3
    keep_best_n = chkpt_cfg.keep_best if chkpt_cfg and chkpt_cfg.keep_best is not None else 1
    save_best_only = chkpt_cfg.save_best_only if chkpt_cfg else False
    checkpoint_prefix = chkpt_cfg.checkpoint_prefix if chkpt_cfg else "checkpoint"
    experiment_name = experiment_config.experiment_name or "default_experiment" # Use from experiment config

    try:
        # Ensure the directory exists before initializing CheckpointManager
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir), # Pass resolved dir as string
            # Pass other checkpointing parameters
            keep_last_n=keep_last_n,
            keep_best_n=keep_best_n,
            save_best_only=save_best_only,
            checkpoint_prefix=checkpoint_prefix,
            # Pass required components
            experiment_name=experiment_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            callbacks=callbacks,
            # Pass the raw config dictionary from the original TrainingState config if needed
            # This depends on whether CheckpointManager needs the *original* config dict
            # or the validated Pydantic model. Passing the validated one for now.
            config=experiment_config.model_dump() # Pass the validated experiment config
        )
        logger.info("CheckpointManager created successfully.")
        return manager
    except Exception as e:
        logger.error(f"Failed to create CheckpointManager: {e}", exc_info=True)
        raise

# TODO: Add factory for TextGenerator if needed elsewhere, or keep its instantiation simple? 