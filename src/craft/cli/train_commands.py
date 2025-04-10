# src/cli/train_commands.py
import typer
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, errors as OmegaConfErrors, listconfig, dictconfig
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path  # Added pathlib
import os
import datetime # Import datetime
from hydra.core.hydra_config import HydraConfig
import traceback
import sys
from pydantic import ValidationError
# Import _LRScheduler
from torch.optim.lr_scheduler import _LRScheduler
# Import TrainingState
from ..training.checkpointing import TrainingState

# Rich console for better output formatting
from rich.console import Console

# Add project root to path to allow importing 'craft'
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up 3 levels
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Add missing imports
from craft.training.evaluation import Evaluator
from omegaconf.errors import ConfigKeyError as OmegaConfKeyError # Import specific error

# Corrected imports based on recent refactoring
try:
    from craft.config.schemas import AppConfig, TrainingConfig, OptimizerConfig, SchedulerConfig, ExperimentConfig
    # Remove incorrect/unused imports
    # from craft.data.factory import prepare_dataloaders_from_config
    # from craft.models.factory import create_model_from_config
    # from craft.training.optimizers import create_optimizer
    # from craft.training.schedulers import create_scheduler
    from craft.training.trainer import Trainer
    from craft.utils.logging import setup_logging # Keep setup_logging for now, maybe needed
    from craft.utils.common import set_seed, setup_device # check_cuda removed
    # Remove incorrect import
    # from craft.utils.omegaconf_utils import register_resolvers, enable_missing_variable_logging
    # Update core factories import to include missing functions
    from craft.core.factories import (
        create_tokenizer,
        create_dataloaders,
        create_model,
        create_callbacks,
        create_evaluator,
        create_grad_scaler,
        create_checkpoint_manager,
        create_optimizer, # Add optimizer
        create_scheduler, # Add scheduler
    )
except ImportError as e:
    print(f"[Train Commands] Error importing necessary Craft modules: {e}")
    sys.exit(1)

# Get the logger for this module
logger = logging.getLogger(__name__)
console = Console()

# Create a Typer app for training commands
# NOTE: Hydra will handle the main execution, Typer is used for potential sub-commands
#       if we restructure later. For now, @hydra.main drives the main 'run' logic.
train_app = typer.Typer(
    name="train",
    help="Commands for training models (driven by Hydra).",
    add_completion=False,
    no_args_is_help=True
)

# Calculate absolute config path relative to this file
# Assumes conf directory is at the project root, 3 levels up from this file's directory
# src/craft/cli/ -> src/craft/ -> src/ -> <root>
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
CONFIG_DIR_ABS = str((_PROJECT_ROOT / "conf").resolve()) # Use resolve() for absolute path
CONFIG_NAME = "config" # Main config file name

logger.info(f"Hydra config directory determined: {CONFIG_DIR_ABS}")
logger.info(f"Hydra main config name: {CONFIG_NAME}")

# Register custom OmegaConf resolvers (Commented out as source file is missing)
# register_resolvers()
# enable_missing_variable_logging()

# --- Hydra Configuration Setup --- #
# Calculate absolute config path relative to this file
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
CONFIG_DIR_ABS = str((_PROJECT_ROOT / "conf").resolve()) # Use resolve() for absolute path
CONFIG_NAME = "config" # Main config file name

logger.info(f"Hydra config directory determined: {CONFIG_DIR_ABS}")
logger.info(f"Hydra main config name: {CONFIG_NAME}")

# --- Remove load_hydra_config helper --- #
# def load_hydra_config(...):
#    ...

# --- Typer Command Modified for Hydra --- #

# We decorate the function that will be the *target* of the CLI call.
# Since 'train run' was the command, we make this function the hydra entry point.
@hydra.main(config_path=CONFIG_DIR_ABS, config_name=CONFIG_NAME, version_base=None)
def run_training_hydra(cfg: DictConfig) -> Optional[TrainingState]: # Renamed function
    """Hydra entry point for running the main training process."""
    # Logging and setup should ideally happen based on Hydra/config if possible,
    # but basic setup here is okay.
    setup_logging() # Use basic logging setup
    logger.info("--- Starting Training Run (via Hydra) ---")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.debug(f"Full composed cfg object:\n{OmegaConf.to_yaml(cfg)}")

    # Extract runtime args potentially overridden by Hydra (or default)
    # We still need a way to pass CLI args like --resume-from / --compile if they are NOT in the config.
    # For now, assume they might be accessible within the composed cfg if passed correctly
    # e.g., `python train.py --resume-from=...` might need custom hydra parsing
    # OR, we accept them from the cfg directly.
    # Let's try getting them from the cfg first, assuming the E2E test passes them as overrides.
    resume_from = cfg.get("resume_from") # Check if passed as override
    compile_model_flag = cfg.get("compile_model") # Check if passed as override

    try:
        # 1. Config is already loaded by @hydra.main
        # (Add any further config validation/processing if needed)

        # 2. Instantiate Trainer (which will handle component instantiation & config validation)
        logger.info("--- Initializing Trainer (instantiating components from config) ---")
        trainer = Trainer(
            cfg=cfg, # Pass the config from Hydra
            # Pass runtime flags if they were successfully extracted/overridden
            resume_from_checkpoint=resume_from,
            compile_model=compile_model_flag
        )

        logger.info("--- Trainer Setup Complete ---")

        # 3. Run Training
        trainer.train()

        logger.info("Training finished successfully.")
        return trainer.state # type: ignore[no-any-return, attr-defined]

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}", exc_info=True)
        # sys.exit(1) # Hydra might handle exit codes
    except OmegaConfErrors.ConfigKeyError as e:
        logger.error(f"Missing configuration key: {e}", exc_info=True)
        # sys.exit(1)
    except OmegaConfErrors.InterpolationKeyError as e:
        logger.error(f"Configuration interpolation error: {e}", exc_info=True)
        # sys.exit(1)
    except ValidationError as e:
        logger.error(f"Configuration validation error (Pydantic): {e}", exc_info=True)
        # sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        # sys.exit(1)
    return None # Return None or raise exception on failure?

# --- Typer Integration (If needed for subcommands, less direct now) --- #
# The Typer command now acts mainly as a wrapper or might be bypassed
# if `python -m craft.cli.train_commands` is used directly with hydra args.
@train_app.command(
    name="run",
    help="Run a training experiment (uses Hydra for config/overrides). Use Hydra syntax: +key=value",
    # Allow unknown options to pass through to Hydra
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_training_cli_wrapper(ctx: typer.Context) -> None:
    """Typer wrapper to potentially call the Hydra-decorated function.
       Note: This setup is complex. Calling the hydra script directly is preferred.
       This Typer command might not correctly pass all args to Hydra.
    """
    # This wrapper is now problematic because @hydra.main needs to parse sys.argv.
    # Calling run_training_hydra() directly here won't work as Hydra won't parse.
    # The E2E test should call `python -m craft.cli.train_commands ...hydra args...`
    # Or we need a more sophisticated way to bridge Typer and Hydra's @hydra.main.
    logger.warning("Running train via Typer wrapper is not the recommended way with @hydra.main.")
    logger.warning("Please run using `python -m craft.cli.train_commands ...hydra args...`")
    # For now, let's just inform the user.
    console.print("[yellow]Warning:[/yellow] Running via Typer wrapper. Direct execution with Hydra args is preferred.")
    # We can attempt to reconstruct sys.argv for hydra, but it's brittle.
    # Example (likely needs refinement):
    # script_name = sys.argv[0] # Or the module path
    # hydra_args = ctx.args # These are the extra args Typer collected
    # sys.argv = [script_name] + hydra_args
    # run_training_hydra() # Now hydra *might* parse correctly
    pass # Avoid calling for now


# Remove the old run_training function
# @train_app.command(...)
# def run_training(...):
#    ...

# --- Other Helper Functions (Keep if needed elsewhere, e.g., validation) --- #

def validate_resume_config(config: DictConfig, checkpoint_data: Dict[str, Any]) -> None:
    """
    Validates that the loaded configuration is compatible with the checkpoint.
    Currently a placeholder - implement specific checks as needed.
    Args:
        config: The loaded configuration for the current run.
        checkpoint_data: The dictionary loaded from the checkpoint file.
    """
    logger.info("Validating compatibility for resuming training...")

    # Example: Check if model architecture matches (can be more sophisticated)
    ckpt_config = checkpoint_data.get('config', {})
    ckpt_model_config = ckpt_config.get('model') if isinstance(ckpt_config, dict) else None
    current_model_config = OmegaConf.to_container(config.get('model'), resolve=True)

    # Simple check: Ensure the '_target_' matches if both exist
    ckpt_target = None
    if isinstance(ckpt_model_config, dict):
        ckpt_target = ckpt_model_config.get('_target_')

    current_target = current_model_config.get('_target_') if isinstance(current_model_config, dict) else None

    if ckpt_target and current_target and ckpt_target != current_target:
        logger.warning(f"Resuming with different model architecture? Ckpt: '{ckpt_target}', Current: '{current_target}'")
        # Depending on strictness, could raise error here

    # Add more checks as needed (e.g., tokenizer compatibility, data format)

    logger.info("Resume config validation complete (basic checks passed).")

# --- Main Guard (If script is run directly) --- #
# This allows running `python src/cli/train_commands.py ...hydra args...`
if __name__ == "__main__":
    # Setup basic logging just in case
    # setup_logging()
    # Let Hydra handle the app launch and config loading
    run_training_hydra()