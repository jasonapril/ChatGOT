import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
import torch
from pydantic import ValidationError, BaseModel
from typing import Optional, List, Any
from hydra.core.hydra_config import HydraConfig

# Add project root to path if needed (adjust if running as a module)
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# --- Core Imports ---
from craft.config.schemas import TrainingConfig, AppConfig, CallbackConfigEntry # Import main config schema and CallbackConfigEntry
from craft.training.trainer import Trainer
from craft.utils.logging import setup_logging # Keep logging setup
from craft.utils.common import set_seed, setup_device # Import setup utilities

# --- Factory/Instantiation Imports (adjust paths if needed) ---
from craft.data.factory import prepare_dataloaders_from_config
from craft.models.factory import create_model_from_config
from craft.training.optimizers import create_optimizer # Correct path for optimizer
from craft.training.schedulers import create_scheduler # Correct path for scheduler
# Tokenizer might be handled within prepare_dataloaders

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training script orchestrated by Hydra.
    """
    # Use Hydra's built-in logging configuration based on conf/hydra/job_logging
    # setup_logging() might be redundant if Hydra handles it, but call if it does custom setup.
    # If setup_logging comes from your utils, call it. If it was just a placeholder, remove it.
    # Assuming setup_logging is necessary for now:
    setup_logging() 

    log.info("--- Starting Craft Training Script ---")
    log.info(f"Hydra CWD: {os.getcwd()}")
    log.info(f"Raw Configuration:\n{OmegaConf.to_yaml(cfg)}")

    try:
        # --- Configuration Validation (Optional but Recommended) ---
        log.info("Validating configuration...")
        try:
            # Resolve interpolations BEFORE validation
            resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            # Validate the entire config structure
            validated_cfg = AppConfig(**resolved_cfg_dict)
            log.info("Configuration validation successful.")
            # Use validated_cfg (Pydantic model) below where possible for type safety
        except ValidationError as e:
            log.error(f"Configuration validation failed!\n{e}")
            raise ValueError("Configuration validation failed.") from e
        except OmegaConf.MissingMandatoryValue as e:
             log.error(f"Configuration validation failed due to missing value: {e}")
             raise ValueError(f"Configuration validation failed: {e}") from e
        except Exception as e:
            log.error(f"An unexpected error occurred during configuration processing: {e}")
            raise ValueError("Configuration processing error.") from e

        # --- Setup ---
        set_seed(validated_cfg.seed)
        device = setup_device(validated_cfg.device)
        log.info(f"Using device: {device}")

        # --- Create Components --- #
        
        # DataLoaders & Tokenizer
        log.info("Preparing dataloaders...")
        # prepare_dataloaders expects the root config containing the 'data' section
        # Pass the whole validated config, it should find cfg.experiment.data
        train_loader, val_loader, test_loader, tokenizer = prepare_dataloaders_from_config(cfg)
        log.info(f"Dataloaders prepared. Train: {train_loader is not None}, Val: {val_loader is not None}, Test: {test_loader is not None}")
        if tokenizer:
            log.info(f"Tokenizer: {type(tokenizer).__name__}")

        # Model
        log.info("Creating model...")
        # Ensure vocab_size from tokenizer is passed to model config if needed
        vocab_size = tokenizer.get_vocab_size() if tokenizer else None
        log.info(f"Using vocab size: {vocab_size}")
        
        # Pass the raw DictConfig from Hydra to the factory, not the validated Pydantic object
        # The factory and hydra instantiate work better with DictConfig
        model_cfg_raw = cfg.experiment.model # Get the original DictConfig
        model = create_model_from_config(model_cfg_raw, vocab_size=vocab_size)
        model.to(device) # Ensure model is on the correct device
        log.info(f"Model created: {model.__class__.__name__}")

        # Optimizer
        log.info("Creating optimizer...")
        # create_optimizer needs optimizer config and model parameters
        optimizer = create_optimizer(model, validated_cfg.experiment.optimizer)
        log.info(f"Optimizer created: {optimizer.__class__.__name__}")

        # Scheduler (Optional)
        scheduler = None
        if validated_cfg.experiment.scheduler:
            log.info("Creating scheduler...")
            # create_scheduler needs optimizer and scheduler config
            scheduler = create_scheduler(optimizer, validated_cfg.experiment.scheduler)
            log.info(f"Scheduler created: {scheduler.__class__.__name__}")
        else:
            log.info("No scheduler configured.")

        # Callbacks (Optional)
        callbacks_list: List[Any] = []
        if validated_cfg.experiment.callbacks:
            log.info("Instantiating callbacks...")
            for name, cb_conf in validated_cfg.experiment.callbacks.items():
                 if cb_conf and isinstance(cb_conf, (DictConfig, dict, BaseModel)) and hasattr(cb_conf, '_target_'):
                     try:
                         # Use Hydra's instantiate for callbacks
                         callback_instance = hydra.utils.instantiate(cb_conf)
                         callbacks_list.append(callback_instance)
                         log.info(f"  Instantiated callback '{name}': {type(callback_instance).__name__}")
                     except Exception as e:
                         log.error(f"  Failed to instantiate callback '{name}': {e}", exc_info=True)
                 # Adjusted condition for Pydantic object
                 elif isinstance(cb_conf, CallbackConfigEntry):
                     try:
                         callback_instance = hydra.utils.instantiate(cb_conf)
                         callbacks_list.append(callback_instance)
                         log.info(f"  Instantiated callback '{name}': {type(callback_instance).__name__}")
                     except Exception as e:
                         log.error(f"  Failed to instantiate callback '{name}': {e}", exc_info=True)
                 else:
                     log.warning(f"  Callback '{name}' configuration invalid or missing '_target_'. Skipping.")
        log.info(f"Total callbacks instantiated: {len(callbacks_list)}")

        # --- Trainer Initialization ---
        log.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler=scheduler,
            device=device,
            config=validated_cfg.experiment.training, # Pass the nested TrainingConfig
            callbacks=callbacks_list,
            tokenizer=tokenizer
        )
        log.info("Trainer initialized.")

        # --- Run Training ---
        log.info("Starting training process...")
        trainer.train()
        log.info("--- Training Finished --- ")

    except ValueError as e:
        log.error(f"Caught configuration/setup error: {e}")
        # Don't raise here, allow Hydra to potentially handle exit
    except ImportError as e:
         log.error(f"Import error: {e}. Ensure dependencies are installed and paths are correct.")
         # Don't raise here
    except Exception as e:
        log.exception(f"An unexpected error occurred during setup or training: {e}")
        # Don't raise here

if __name__ == "__main__":
    main() 