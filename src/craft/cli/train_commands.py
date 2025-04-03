# src/cli/train_commands.py
import typer
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from typing import List, Optional
from pathlib import Path  # Added pathlib
import os
import datetime # Import datetime
from hydra.core.hydra_config import HydraConfig
import traceback

# Rich console for better output formatting
from rich.console import Console

# Import necessary components (adjust paths as needed based on project structure)
from ..data.base import create_data_loaders_from_config # Corrected import path
from ..models.factory import create_model_from_config # Corrected import name
from ..training.optimizers import create_optimizer
from ..training.schedulers import create_scheduler
from ..training.trainer import Trainer
from ..training.callbacks import CallbackList # Or potentially a factory
# from ..utils.logging import setup_logging # Setup is likely handled by Hydra/CLI run.py
from ..utils.common import set_seed # If seeding is desired

# Get the logger for this module
logger = logging.getLogger(__name__)
console = Console()

# Create a Typer app for training commands
train_app = typer.Typer()

# Calculate absolute config path relative to this file
# Assumes conf directory is at the project root, 3 levels up from this file's directory
# src/craft/cli/ -> src/craft/ -> src/ -> <root>
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
CONFIG_DIR_ABS = str(_PROJECT_ROOT / "conf")
CONFIG_NAME = "config" # Main config file name

# Typer command function - This is what Typer calls
@train_app.command("language", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
def train_language_model_entry(context: typer.Context):
    """
    Typer entry point for training a language model using Hydra's Compose API.
    Hydra overrides are passed as extra arguments.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Typer command 'language' invoked. Raw context.args for Hydra: {context.args}")
    
    # Ensure overrides are in the expected list format for compose
    overrides = list(context.args)

    try:
        # --- Initialize Hydra --- 
        # Use version_base=None for compatibility if needed, or specify e.g., "1.2"
        hydra.initialize_config_dir(config_dir=CONFIG_DIR_ABS, job_name="train_language_cli", version_base="1.2") # Using 1.2 for example
        
        # --- Compose Config --- 
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=overrides)
        logger.info("Hydra configuration composed successfully via API.")

        # --- Manually Create Timestamped Output Directory --- 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Construct path relative to project root
        run_output_dir_relative = Path("outputs") / "runs" / timestamp
        run_output_dir_abs = _PROJECT_ROOT / run_output_dir_relative
        
        try:
            run_output_dir_abs.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created run output directory: {run_output_dir_abs}")
        except Exception as e:
            logger.error(f"Failed to create run output directory {run_output_dir_abs}: {e}")
            raise

        # --- Setup Logging (using absolute path for file handler) --- 
        log_file_path = run_output_dir_abs / "train_runner.log"
        # Modify cfg.logging *before* passing to dictConfig if needed, or configure separately
        # For simplicity, we'll assume the default logging config exists and just log the path
        logger.info(f"Logging run to file: {log_file_path}")
        # Actual file handler setup might need adjustment based on logging config structure
        logging_conf = OmegaConf.to_container(cfg.logging, resolve=True)
        if isinstance(logging_conf, dict):
            # Find file handler and update filename if possible (crude example)
            if 'handlers' in logging_conf and 'file' in logging_conf['handlers']:
                logging_conf['handlers']['file']['filename'] = str(log_file_path)
            logging.config.dictConfig(logging_conf)
            logger.info("Logging configured via composed Hydra config (file path updated).")
        else:
            logger.warning("Hydra logging configuration is not a dictionary, using default logging.")
            logging.basicConfig(level=logging.INFO) # Fallback
        
        logger.info("Composed Hydra Config:\n" + OmegaConf.to_yaml(cfg))

        # --- Define Absolute Checkpoint Path --- 
        checkpoint_dir_abs = run_output_dir_abs / "checkpoints"
        # No need to create it here, CheckpointManager will handle it

        # --- Call Core Training Logic (pass absolute checkpoint path) --- 
        _run_language_training_core(cfg, checkpoint_dir_abs=str(checkpoint_dir_abs)) # Pass absolute path

    except Exception as e:
        logger.error(f"Hydra initialization, composition, or training failed:\n{traceback.format_exc()}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


# RENAMED & MODIFIED: Core training logic function (no @hydra.main decorator)
# Signature changed back to only accept cfg
def _run_language_training_core(cfg: DictConfig, checkpoint_dir_abs: str):
    """
    Core training loop logic.
    Receives the fully composed Hydra config and the absolute checkpoint directory.
    CWD remains the project root.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting core training logic (_run_language_training_core). Checkpoint Dir: {checkpoint_dir_abs}")
    
    try:
        # Determine device
        # ... (device selection logic remains the same) ...
        if cfg.get('force_cpu', False):
            device = torch.device("cpu")
            logger.info("CPU forced via configuration.")
        else:
            device_str = cfg.training.get('device', 'auto')
            if device_str == 'auto':
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device_str)
        logger.info(f"Using device: {device}")

        # --- Data Loading (Instantiate Train Dataset FIRST to get vocab_size) ---
        logger.info("Creating datasets...")
        train_dataset = None
        val_dataset = None
        if cfg.data.get('train') and cfg.data.train.get('dataset') and cfg.data.train.dataset.get('_target_'):
            logger.info("  Instantiating train dataset...")
            train_dataset = hydra.utils.instantiate(cfg.data.train.dataset)
            logger.info(f"  Train dataset size: {len(train_dataset)}")
        else:
            raise ValueError("Training dataset configuration ('data.train.dataset') is missing or invalid.")

        if cfg.data.get('val') and cfg.data.val.get('dataset') and cfg.data.val.dataset.get('_target_'):
            logger.info("  Instantiating validation dataset...")
            val_dataset = hydra.utils.instantiate(cfg.data.val.dataset)
            logger.info(f"  Validation dataset size: {len(val_dataset)}")

        # --- Model Creation (Inject vocab_size) ---
        logger.info("Creating model...")
        if not hasattr(train_dataset, 'vocab_size') or not train_dataset.vocab_size:
             raise ValueError("Loaded train_dataset does not have a valid 'vocab_size' attribute.")

        vocab_size = train_dataset.vocab_size
        logger.info(f"Got vocab_size={vocab_size} from train dataset.")

        # Convert the OmegaConf DictConfig to a plain dictionary for the factory
        model_config_dict = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=False)
        # Ensure model_type is correctly passed as a string
        if 'model_type' in model_config_dict and not isinstance(model_config_dict['model_type'], str):
            model_config_dict['model_type'] = str(model_config_dict['model_type'])
            
        # Inject vocab_size into the model config dict
        model_config_dict['vocab_size'] = vocab_size
        logger.info(f"Injected vocab_size={vocab_size} into model config dictionary.")

        model = create_model_from_config(model_config_dict)
        # Log memory before moving model to device
        if device.type == 'cuda':
            logger.info(f"GPU Memory BEFORE model.to(device): {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB reserved")
        
        model.to(device)
        
        # Log memory after moving model to device
        if device.type == 'cuda':
            logger.info(f"GPU Memory AFTER model.to(device): {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB reserved")
            
        logger.info(f"Model created: {type(model).__name__}")

        # --- Create DataLoaders (after datasets are instantiated) ---
        logger.info("Creating dataloaders...")
        # Use a helper or manual instantiation based on cfg.data (batch_size, num_workers, etc.)
        # Example using a helper if available:
        # dataloaders = create_dataloaders(train_dataset, val_dataset, cfg.data)
        # Or manual:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.get('batch_size', 16),
            shuffle=True, # Usually shuffle training data
            num_workers=cfg.data.get('num_workers', 0),
            pin_memory=True # Often good practice with GPUs
        )
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.data.get('batch_size', 16) * 2, # Often use larger batch for val
                shuffle=False,
                num_workers=cfg.data.get('num_workers', 0),
                pin_memory=True
            )
        logger.info("Dataloaders created.")

        # --- Optimizer & Scheduler ---
        logger.info("Creating optimizer and scheduler...")
        optimizer = create_optimizer(model, cfg.optimizer)
        scheduler = create_scheduler(optimizer, cfg.scheduler) if cfg.get('scheduler') else None
        logger.info(f"Optimizer: {type(optimizer).__name__}")
        if scheduler:
             logger.info(f"Scheduler: {type(scheduler).__name__}")

        # --- Callbacks --- 
        logger.info("Instantiating callbacks...")
        callbacks_list = []
        if "callbacks" in cfg and cfg.callbacks:
            for name, cb_conf in cfg.callbacks.items():
                logger.info(f"Instantiating callback: {name}")
                # REMOVED: Logic to inject hydra_cfg into TensorBoardLogger
                # instantiate_kwargs = {}
                # if cb_conf._target_ == "craft.training.callbacks.TensorBoardLogger":
                #     logger.info("Removing explicit log_dir for TensorBoardLogger, will use CWD.")
                #     if "log_dir" in cb_conf: 
                #         cb_conf_mutable = OmegaConf.to_container(cb_conf, resolve=False)
                #         del cb_conf_mutable['log_dir']
                #         cb_conf = OmegaConf.create(cb_conf_mutable)
                
                # Instantiate the callback using Hydra's utility
                # Ensure we pass a DictConfig or OmegaConf object if required by instantiate
                callback_cfg_obj = cb_conf if isinstance(cb_conf, DictConfig) else OmegaConf.create(cb_conf)
                # Pass NO extra kwargs now
                callback = hydra.utils.instantiate(callback_cfg_obj)
                callbacks_list.append(callback)
        else:
            logger.info("No callbacks configured.")

        # --- Trainer --- 
        logger.info("Creating Trainer...")
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=OmegaConf.to_container(cfg, resolve=True), # Pass plain dict
            device=device,
            callbacks=callbacks_list,
            checkpoint_dir=checkpoint_dir_abs, # Use the passed absolute path
            use_amp=cfg.training.get('use_amp', True),
            gradient_accumulation_steps=cfg.training.get('gradient_accumulation_steps', 1),
            max_grad_norm=cfg.training.get('max_grad_norm', 1.0),
            log_interval=cfg.training.get('log_interval', 10),
            eval_interval=cfg.training.get('eval_interval', 1000),
            save_interval=cfg.training.get('save_steps_interval', 1000),
            num_epochs=cfg.training.get('num_epochs', 1),
            resume_from_checkpoint=cfg.training.get('resume_from_checkpoint') # Pass the absolute path for resume, if provided in config
        )

        # --- Run Training --- 
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training finished successfully.")

    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        logger.error(traceback.format_exc()) # Log the full traceback for debugging
        typer.echo(f"Error during training: {e}", err=True)
        # We might not want to exit here, but let the main entry handle it
        # raise typer.Exit(code=1) 
        raise  # Re-raise the exception to be caught by the main entry point

    # Removed the old direct training logic
    # ... (old hydra init, data prep, model creation, trainer code) ... 