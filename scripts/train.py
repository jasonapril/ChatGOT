import argparse
import logging
import os
import sys

# Add project root to path to allow importing 'craft'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import necessary hydra/torch/pydantic/typing first
import hydra
from omegaconf import DictConfig, OmegaConf, errors as OmegaConfErrors
import torch
from pydantic import ValidationError
from typing import Optional, Dict, Any, List

# Import dataloader factory directly at the top level
from craft.data import prepare_dataloaders_from_config

# Import utility functions here
try:
    from craft.utils.common import set_seed, setup_device
    from craft.config.schemas import AppConfig # Import the main Pydantic schema
except ImportError as e:
    logging.error(f"Failed to import common utilities: {e}")
    sys.exit(1)

# Delayed imports for better error handling if dependencies are missing
# We can leave the rest here, but prepare_dataloaders_from_config is now imported above
DataLoadersDict = None
create_data_loaders_from_config = None # This name is likely unused now
Model = None
create_model_from_config = None
Trainer = None
create_optimizer = None
create_scheduler = None
BaseTokenizer = None
CharLevelTokenizer = None

def _delayed_imports():
    # Remove prepare_dataloaders_from_config from here
    global DataLoadersDict, create_data_loaders_from_config, Model, create_model_from_config, Trainer, create_optimizer, create_scheduler, BaseTokenizer, CharLevelTokenizer
    try:
        # from craft.data.base import prepare_dataloaders_from_config # Removed from here
        from craft.data import create_data_loaders_from_config, DataLoadersDict # Keep the others
        from craft.models import Model, create_model_from_config
        from craft.training.trainer import Trainer
        from craft.training.optimizers import create_optimizer
        from craft.training.schedulers import create_scheduler
        from craft.data.tokenizers.base import BaseTokenizer
        from craft.data.tokenizers.char_level import CharLevelTokenizer
    except ImportError as e:
        # Re-raise the import error to be handled by the main function's exception block,
        # which will have access to the properly configured logger.
        raise e

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training script execution using Hydra."""
    # Get logger *after* Hydra has configured logging
    logger = logging.getLogger(__name__)

    logger.info("Starting training script...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Hydra working directory: {os.getcwd()}") # Hydra changes cwd

    try:
        # --- Configuration Validation ---
        logger.info("Validating configuration...")
        try:
            # Resolve interpolations and convert OmegaConf to a plain dict
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            # Validate the dictionary using the Pydantic schema
            validated_cfg = AppConfig(**cfg_dict)
            logger.info("Configuration validation successful.")
            logger.debug(f"Validated Config:\n{validated_cfg.model_dump_json(indent=2)}")
        except ValidationError as e:
            logger.error(f"Configuration validation failed!\n{e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred during configuration processing: {e}")
            sys.exit(1)

        # --- Delayed Imports ---
        _delayed_imports()

        # --- Setup ---
        set_seed(validated_cfg.seed)
        device = setup_device(validated_cfg.device)
        logger.info(f"Using device: {device}")

        # --- Create DataLoaders ---
        logger.info("Creating dataloaders...")
        try:
            # Pass the main config object directly
            train_loader, val_loader, test_loader, tokenizer = prepare_dataloaders_from_config(cfg)
        
            if train_loader is None:
                logger.error("Failed to create train dataloader. Exiting.")
                sys.exit(1)
            logger.info("Dataloaders created.")
        except Exception as e:
            logger.error(f"Failed to create dataloaders: {e}")
            sys.exit(1)

        # --- Load Tokenizer (for saving with checkpoints) ---
        # tokenizer is now returned by prepare_dataloaders_from_config
        # Ensure it's captured and used below
        if tokenizer is None:
             logger.warning("prepare_dataloaders_from_config did not return a tokenizer.")
        else:
            logger.info(f"Tokenizer returned from data factory: {type(tokenizer).__name__}")

        # --- Determine vocab_size (example: infer from dataset or LOADED tokenizer) ---
        # If vocab_size is needed by the model and not set, try to infer it
        vocab_size_inferred = None
        if hasattr(validated_cfg.model, 'config') and validated_cfg.model.config.get('vocab_size') is None:
            # Prioritize getting vocab_size from the loaded tokenizer if available
            if tokenizer is not None and hasattr(tokenizer, 'vocab_size'):
                vocab_size_inferred = tokenizer.vocab_size
                logger.info(f"Using vocab_size from loaded tokenizer: {vocab_size_inferred}")
            # Fallback: Try inferring from dataset (might be relevant for other dataset types)
            elif hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'get_vocab_size'):
                try:
                    vocab_size_inferred = train_loader.dataset.get_vocab_size()
                    if vocab_size_inferred:
                        logger.info(f"Inferred vocab_size from training dataset: {vocab_size_inferred}")
                        # Update the nested config
                        validated_cfg.model.config.vocab_size = vocab_size_inferred
                    else:
                        logger.warning("train_loader.dataset.get_vocab_size() returned None or 0.")
                except Exception as e:
                    logger.warning(f"Could not infer vocab_size from dataset: {e}")
            else:
                logger.warning("Cannot infer vocab_size: No tokenizer loaded and train_loader.dataset missing or has no 'get_vocab_size' method.")

        # --- Save Tokenizer (if exists and saveable) ---
        # Removed tokenizer saving logic here

        # --- Create Model ---
        logger.info("Creating model...")
        # Explicitly pass inferred vocab_size if it was found
        model_creation_kwargs = {}
        if vocab_size_inferred is not None:
            # Check if the config expects vocab_size at the top level or nested
            # Assuming factory handles placing it correctly based on target type
            model_creation_kwargs['vocab_size'] = vocab_size_inferred
            logger.info(f"Set validated_cfg.model.vocab_size to {vocab_size_inferred}")
            # Ensure the nested config also reflects this, if it exists
            if hasattr(validated_cfg.model, 'config') and hasattr(validated_cfg.model.config, 'vocab_size'):
                 validated_cfg.model.config.vocab_size = vocab_size_inferred # Update nested config directly

        model = create_model_from_config(validated_cfg.model, **model_creation_kwargs)
        model.to(device)
        logger.info("Model created.")

        # --- Optimizer and Scheduler --- # 
        logger.info("Creating optimizer and scheduler...")
        # Use model_dump() - factory functions now handle both 'target' and '_target_'
        optimizer_config_dict = validated_cfg.optimizer.model_dump()
        optimizer = create_optimizer(model, optimizer_config_dict)
        try:
            # Use model_dump() - factory functions now handle both 'target' and '_target_'
            scheduler_dict = validated_cfg.scheduler.model_dump() if validated_cfg.scheduler else None
            scheduler = create_scheduler(optimizer, scheduler_dict)
            logger.info("Optimizer created. Scheduler created.")
        except Exception as e:
            logger.exception("Failed during scheduler creation.")
            return

        # --- Callbacks --- #
        logger.info("Instantiating callbacks...")
        callbacks = []
        if validated_cfg.callbacks:
            for cb_name, cb_conf in validated_cfg.callbacks.items():
                # Check if cb_conf is not None and is a DictConfig (or dict) containing _target_
                if cb_conf and isinstance(cb_conf, (DictConfig, dict)) and '_target_' in cb_conf:
                    logger.info(f"  Instantiating callback '{cb_name}'...")
                    try:
                        callback_instance = hydra.utils.instantiate(cb_conf)
                        callbacks.append(callback_instance)
                        logger.info(f"    Successfully instantiated callback: {type(callback_instance).__name__}")
                    except Exception as e:
                        logger.error(f"    Failed to instantiate callback '{cb_name}': {e}", exc_info=True)
                else:
                    logger.warning(f"Callback '{cb_name}' configuration is invalid, None, or missing '_target_'. Skipping. Config: {cb_conf}")
        logger.info(f"Instantiated {len(callbacks)} callbacks.")

        # --- Trainer Initialization ---
        logger.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler=scheduler,
            device=device,
            config=validated_cfg.training, # Pass the training sub-config
            callbacks=callbacks,
            tokenizer=tokenizer # Pass the loaded tokenizer
        )
        logger.info("Trainer initialized.")

        # --- Run Training ---
        logger.info("Starting training process...")
        trainer.train()
        logger.info("Training finished.")

    except ImportError as ie:
         logger.error(f"Import error: {ie}. Ensure all dependencies and the 'craft' package are installed.")
         sys.exit(1)
    except ValidationError as ve: # Catch validation errors specifically if missed earlier
        logger.error(f"Caught Pydantic Validation Error later in the process: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}") # Log traceback
        sys.exit(1)

if __name__ == "__main__":
    main() 