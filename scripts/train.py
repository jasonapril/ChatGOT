import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
import os
import torch
from typing import Optional

# Add project root to path to allow importing 'craft'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import utility functions here
try:
    from craft.utils.common import set_seed, setup_device
except ImportError as e:
    logging.error(f"Failed to import common utilities: {e}")
    sys.exit(1)

# Delayed imports for better error handling if dependencies are missing
DataLoadersDict = None
create_data_loaders_from_config = None
Model = None
create_model_from_config = None
Trainer = None
create_optimizer = None
create_scheduler = None

def _delayed_imports():
    global DataLoadersDict, create_data_loaders_from_config, Model, create_model_from_config, Trainer, create_optimizer, create_scheduler
    try:
        from craft.data import create_data_loaders_from_config, DataLoadersDict
        from craft.models import Model, create_model_from_config
        from craft.training.trainer import Trainer
        from craft.training.optimizers import create_optimizer
        from craft.training.schedulers import create_scheduler
    except ImportError as e:
        # Re-raise the import error to be handled by the main function's exception block,
        # which will have access to the properly configured logger.
        # logger.error(f"Failed to import necessary craft components: {e}")
        # logger.error("Please ensure the 'craft' package is installed and accessible.")
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
        # --- Delayed Imports ---
        _delayed_imports()

        # --- Setup ---
        set_seed(cfg.get('seed', 42))
        device = setup_device(cfg.get('device', 'auto'))
        logger.info(f"Using device: {device}")

        # --- Data Loading ---
        logger.info("Creating dataloaders...")
        # Assuming create_data_loaders_from_config handles train/val/test based on cfg.data
        # It should return a dict like {'train': train_loader, 'val': val_loader}
        dataloaders = create_data_loaders_from_config(cfg.data)
        train_loader = dataloaders.get('train')
        val_loader = dataloaders.get('val')
        if train_loader is None:
            raise ValueError("Training dataloader could not be created. Check data configuration.")
        logger.info("Dataloaders created.")

        # --- Determine vocab_size from data --- #
        vocab_size = None
        try:
            # Attempt to get vocab_size from the dataset object
            if hasattr(train_loader.dataset, 'vocab_size'):
                vocab_size = train_loader.dataset.vocab_size
                logger.info(f"Inferred vocab_size from training dataset: {vocab_size}")
            else:
                logger.warning("Training dataset or dataloader does not have a 'vocab_size' attribute.")
        except Exception as e:
            logger.warning(f"Could not automatically determine vocab_size from dataset: {e}")

        # If not found in data, use config value or default
        if vocab_size is None:
            vocab_size = cfg.model.get("vocab_size") # Get from config if present
            if vocab_size is None:
                logger.error("vocab_size could not be determined from data or config.")
                # Decide whether to raise error or use a hardcoded default
                raise ValueError("Missing vocab_size definition.") 
            else:
                logger.warning(f"Using vocab_size from configuration: {vocab_size}")
        
        # --- Model Creation ---
        logger.info("Creating model...")
        # Pass the inferred vocab_size directly to the factory function
        model = create_model_from_config(cfg.model, vocab_size=vocab_size)
        model.to(device) # Ensure model is on the correct device initially
        logger.info("Model created.")

        # --- Optimizer and Scheduler --- # 
        logger.info("Creating optimizer and scheduler...")
        optimizer = create_optimizer(model, cfg.optimizer)
        try:
            scheduler = create_scheduler(optimizer, cfg.scheduler)
            logger.info("Optimizer created. Scheduler created.")
        except Exception as e:
            logger.exception("Failed during scheduler creation.")
            return

        # --- Callback Instantiation ---
        callbacks = []
        if "callbacks" in cfg and cfg.callbacks:
            logger.info("Instantiating callbacks...")
            # Iterate over the configuration *values* (dictionaries), not the keys
            for cb_name, cb_conf in cfg.callbacks.items(): 
                if isinstance(cb_conf, DictConfig): # Ensure it's a config object
                    logger.info(f"  Instantiating callback '{cb_name}'...")
                    try:
                        cb_instance = hydra.utils.instantiate(cb_conf)
                        callbacks.append(cb_instance)
                        logger.info(f"    Successfully instantiated callback: {type(cb_instance).__name__}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate callback '{cb_name}' with config: {cb_conf}")
                        logger.exception(e)
                        logger.warning(f"Skipping callback '{cb_name}' due to instantiation error.")
                else:
                     logger.warning(f"Skipping non-dictionary item found in callbacks config: key='{cb_name}', value='{cb_conf}'")
            logger.info(f"Instantiated {len(callbacks)} callbacks.")
        else:
            logger.info("No callbacks configured.")

        # --- Trainer Initialization ---
        logger.info("Initializing Trainer...")
        # Pass relevant parts of the validated config to the Trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=OmegaConf.to_container(cfg, resolve=True), # Pass plain dict
            device=device,
            callbacks=callbacks,
            checkpoint_dir=os.getcwd(), # Hydra sets cwd to output dir
            use_amp=cfg.training.get('use_amp', False),
            gradient_accumulation_steps=cfg.training.get('gradient_accumulation_steps', 1),
            max_grad_norm=cfg.training.get('max_grad_norm', None),
            log_interval=cfg.training.get('log_interval', 10),
            eval_interval=cfg.training.get('eval_interval', 1000),
            save_interval=cfg.training.get('save_interval', 5000),
            num_epochs=cfg.training.get('num_epochs', 1),
            resume_from_checkpoint=cfg.get('resume_from_checkpoint', None)
        )
        logger.info("Trainer initialized.")

        # --- Run Training ---
        logger.info("Starting training process...")
        trainer.train()
        logger.info("Training finished.")

    except ImportError as ie:
         logger.error(f"Import error: {ie}. Ensure all dependencies and the 'craft' package are installed.")
         sys.exit(1)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}") # Log traceback
        sys.exit(1)

if __name__ == "__main__":
    main() 