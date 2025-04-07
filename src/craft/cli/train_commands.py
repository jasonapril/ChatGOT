# src/cli/train_commands.py
import typer
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, errors as OmegaConfErrors
import torch
from typing import List, Optional
from pathlib import Path  # Added pathlib
import os
import datetime # Import datetime
from hydra.core.hydra_config import HydraConfig
import traceback
import sys

# Rich console for better output formatting
from rich.console import Console

# Add project root to path to allow importing 'craft'
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up 3 levels
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Corrected imports based on recent refactoring
try:
    from craft.config.schemas import AppConfig, TrainingConfig, OptimizerConfig, SchedulerConfig
    from craft.data.factory import prepare_dataloaders_from_config # Corrected import
    from craft.models.factory import create_model_from_config
    from craft.training.optimizers import create_optimizer
    from craft.training.schedulers import create_scheduler
    from craft.training.trainer import Trainer
    from craft.utils.logging import setup_logging # Keep setup_logging for now, maybe needed
    from craft.utils.common import set_seed, setup_device
except ImportError as e:
    print(f"[Train Commands] Error importing necessary Craft modules: {e}")
    sys.exit(1)

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
    Train a language model using Hydra configuration.

    Pass Hydra overrides directly after the command, e.g.:
    `craft train language experiment=my_exp optimizer.lr=0.0001`
    """
    logger = logging.getLogger(__name__) # Get logger within command context
    logger.debug(f"Typer command 'train language' invoked. Raw context.args for Hydra: {context.args}")

    # Ensure overrides are in the expected list format for compose
    overrides = list(context.args)

    # --- Initialize Hydra & Compose Config --- #
    cfg: Optional[DictConfig] = None
    try:
        # Use version_base=None for compatibility if needed, or specify e.g., "1.2"
        hydra.initialize_config_dir(config_dir=CONFIG_DIR_ABS, job_name="train_language_cli", version_base="1.2")
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=overrides)
        logger.info("Hydra configuration composed successfully via API.")
        # Log the final composed config (potentially large)
        logger.debug("Composed Hydra Config:\n%s", OmegaConf.to_yaml(cfg))

        # Set Hydra CWD manually if needed (optional, depends on where outputs go)
        # os.chdir(hydra.utils.get_original_cwd()) # Example: Change back to original dir

    except Exception as e:
        logger.error(f"Hydra initialization or composition failed:\n{traceback.format_exc()}")
        console.print(f"[bold red]Error composing configuration:[/bold red] {e}")
        raise typer.Exit(code=1)

    # --- Run Core Training Logic --- #
    try:
        # This function now mirrors the logic from the refactored scripts/train.py
        _run_training_with_hydra_cfg(cfg)
    except Exception as e:
        # Catch errors during the training process itself
        logger.error(f"Training process failed:\n{traceback.format_exc()}")
        console.print(f"[bold red]Error during training:[/bold red] {e}")
        raise typer.Exit(code=1)

# --- Core Training Logic (Using Composed Hydra Config) --- #
def _run_training_with_hydra_cfg(cfg: DictConfig):
    """
    Core training logic that takes the composed Hydra config.
    Mirrors the main logic from the refactored scripts/train.py.
    """
    validated_cfg: Optional[AppConfig] = None
    try:
        # --- Configuration Validation --- #
        logger.info("Validating configuration with Pydantic...")
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            validated_cfg = AppConfig(**cfg_dict)
            logger.info("Configuration validation successful.")
            # Log validated config details if needed (be mindful of verbosity/secrets)
            # logger.debug(f"Validated Config (Pydantic):\n{validated_cfg.model_dump_json(indent=2)}")
        except OmegaConfErrors.ConfigKeyError as e:
            logger.error(f"Config validation failed! Missing key or interpolation error: {e}")
            raise ValueError(f"Config validation failed: {e}") from e # Raise to be caught below
        except ValidationError as e:
            logger.error(f"Config validation failed! Schema mismatch:\n{e}")
            raise ValueError(f"Config validation failed: {e}") from e # Raise to be caught below
        except Exception as e:
            logger.error(f"An unexpected error occurred during configuration processing: {e}", exc_info=True)
            raise # Re-raise unexpected errors

        # --- Basic Setup --- #
        set_seed(validated_cfg.seed)
        device = setup_device(validated_cfg.device)
        logger.info(f"Using device: {device}")
        logger.info(f"Experiment Name: {validated_cfg.experiment_name}")
        if validated_cfg.resume_from:
            logger.info(f"Resume requested from: {validated_cfg.resume_from}")

        # --- Prepare Model Config Dict --- #
        # Trainer expects the model config as a dictionary
        model_config_obj = validated_cfg.experiment.model
        model_config_dict = model_config_obj.model_dump(exclude_none=True)
        logger.debug("Prepared model_config dictionary for Trainer.")

        # --- Create Trainer (Handles internal component instantiation) --- #
        logger.info("Creating Trainer...")
        trainer = Trainer(
            model_config=model_config_dict,
            config=validated_cfg.experiment.training, # Pass validated TrainingConfig Pydantic model
            device=device,
            experiment_config=cfg.experiment, # Pass experiment OmegaConf node for internal instantiation
            experiment_name=validated_cfg.experiment_name,
            resume_from_checkpoint=validated_cfg.resume_from
            # compile_model could be passed from validated_cfg if added there
        )
        logger.info("Trainer initialized successfully.")

        # --- Start Training --- #
        logger.info("Starting training process...")
        trainer.train()
        logger.info("--- Training Finished --- ")

    except ValueError as e:
         # Catch config/setup errors specifically
         logger.error(f"Configuration or setup error: {e}", exc_info=True)
         raise # Re-raise to be caught by the Typer entry point
    except ImportError as e:
        logger.error(f"Import error during setup or training: {e}. Ensure all dependencies are installed.")
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the training script execution: {e}", exc_info=True)
        raise # Re-raise to be caught by the Typer entry point

# --- Removed _run_language_training_core function --- #
# This logic is now replaced by _run_training_with_hydra_cfg
# def _run_language_training_core(cfg: DictConfig, checkpoint_dir_abs: str):
#     ...
#     (Old implementation deleted)
# ------------------------------------------------------- # 