import hydra
from omegaconf import DictConfig, OmegaConf, errors as omegaconf_errors
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
        # -- Early Log: Print Resolved Hydra Run Directory --
        try:
            resolved_run_dir = HydraConfig.get().run.dir
            # Use the root logger initially setup by Hydra or setup_logging if called
            logging.info(f"[HYDRA CHECK] Resolved Hydra run directory: {resolved_run_dir}")
            # Also print to be absolutely sure it appears
            print(f"[HYDRA CHECK VIA PRINT] Resolved Hydra run directory: {resolved_run_dir}")
        except Exception as e:
            logging.warning(f"[HYDRA CHECK] Could not get Hydra run directory early: {e}")

        # --- Configuration Validation (Optional but Recommended) ---
        log.info("Validating configuration...")
        # Attempt to resolve and convert the config to a dict for early validation
        # This catches interpolation errors and mandatory value checks if `throw_on_missing=True`
        resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        log.debug("Resolved config (dict):\n%s", resolved_cfg_dict) # Careful logging potentially sensitive data

        # If using Pydantic or similar for structured config validation:
        log.info("Validating config structure with Pydantic (AppConfig)...")
        validated_cfg = AppConfig(**resolved_cfg_dict)
        log.info("Configuration validation successful.")

        # Log validated config (consider using a filtered version)
        log.debug("Validated AppConfig:\n%s", validated_cfg)

        # Set seed for reproducibility
        set_seed(validated_cfg.seed)
        log.info(f"Random seed set to: {validated_cfg.seed}")

        # Determine device
        device = setup_device(validated_cfg.device)
        log.info(f"Using device: {device}")

        # Handle potential AMP override based on device
        effective_use_amp = validated_cfg.experiment.training.use_amp
        if device.type == 'cpu' and effective_use_amp:
            log.warning("AMP is enabled but running on CPU. Disabling AMP for CPU execution.")
            effective_use_amp = False
        # --- End Override ---

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
        # Iterate through the RAW DictConfig callbacks from the original cfg object
        if cfg.experiment and hasattr(cfg.experiment, "callbacks") and cfg.experiment.callbacks:
            log.info("Instantiating callbacks...")
            # Get Hydra run directory AFTER hydra setup is complete
            hydra_run_dir = None
            try:
                hydra_run_dir = HydraConfig.get().run.dir
                log.info(f"Resolved Hydra run directory: {hydra_run_dir}")
            except Exception as e:
                log.warning(f"Could not automatically resolve Hydra run directory: {e}. Callbacks needing it might use defaults.")

            for name, cb_conf_raw in cfg.experiment.callbacks.items(): # Use raw cfg
                # Check if it's a valid config (DictConfig or dict) and has a _target_
                 if cb_conf_raw and isinstance(cb_conf_raw, (DictConfig, dict)) and hasattr(cb_conf_raw, '_target_'):
                     try:
                         cb_conf_to_instantiate = cb_conf_raw
                         instantiate_args = {} # Start with default args from config

                         # --- Special handling for SampleGenerationCallback ---
                         # Check if this is the sample generation callback
                         target_class_str = cb_conf_raw.get('_target_')
                         if target_class_str == 'craft.training.callbacks.SampleGenerationCallback':
                             log.debug(f"Special handling for {target_class_str} instantiation.")
                             try:
                                 training_cfg = validated_cfg.experiment.training
                                 instantiate_args['start_prompt'] = training_cfg.sample_start_text
                                 instantiate_args['max_new_tokens'] = training_cfg.sample_max_new_tokens
                                 instantiate_args['temperature'] = training_cfg.sample_temperature
                                 log.debug(f"  Injecting params: start_prompt='{instantiate_args['start_prompt']}', max_new_tokens={instantiate_args['max_new_tokens']}, temperature={instantiate_args['temperature']}")
                             except AttributeError as ae:
                                  log.error(f"  Failed to get sampling parameters from training config for {name}: {ae}. Callback might not function correctly.")
                                  # Decide: continue without these args, or raise/skip? Let's continue for now.
                             except Exception as e_inner:
                                 log.error(f"  Unexpected error getting sampling parameters for {name}: {e_inner}")


                         # Pass the original config and any specifically extracted args
                         # Hydra will merge cb_conf_to_instantiate with instantiate_args,
                         # with instantiate_args taking precedence for duplicate keys.
                         callback_instance = hydra.utils.instantiate(cb_conf_to_instantiate, **instantiate_args)
                         callbacks_list.append(callback_instance)
                         log.info(f"  Instantiated callback '{name}': {type(callback_instance).__name__}")
                     except Exception as e:
                         log.error(f"  Failed to instantiate callback '{name}': {e}", exc_info=True)
                 else:
                     log.warning(f"  Callback '{name}' configuration invalid or missing '_target_'. Skipping: {cb_conf_raw}")
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
            # Pass the potentially overridden use_amp value via config
            # Note: Trainer reads use_amp from its config object
            # We need to update the config object passed to Trainer
            config=validated_cfg.experiment.training.model_copy(update={'use_amp': effective_use_amp}),
            callbacks=callbacks_list,
            tokenizer=tokenizer,
            resume_from_checkpoint=validated_cfg.resume_from, # Use the top-level 'resume_from' field
            experiment_name=validated_cfg.experiment_name # <-- Pass experiment name
        )
        log.info("Trainer initialized.")

        # --- Run Training ---
        log.info("Starting training process...")
        trainer.train()
        log.info("--- Training Finished --- ")

    except omegaconf_errors.MissingMandatoryValue as e: # Corrected exception type
        log.error(f"Caught configuration validation error (MissingMandatoryValue): {e}")
        # Don't raise here, allow Hydra to potentially handle exit
    except omegaconf_errors.InterpolationKeyError as e: # Catch interpolation errors explicitly
        log.error(f"Caught configuration validation error (InterpolationKeyError): {e}")
    except ValueError as e:
        log.error(f"Caught configuration/setup error: {e}")
    except ImportError as e:
         log.error(f"Import error: {e}. Ensure dependencies are installed and paths are correct.")
         # Don't raise here
    except Exception as e:
        log.exception(f"An unexpected error occurred during setup or training: {e}")
        # Don't raise here

if __name__ == "__main__":
    main() 