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
train_app = typer.Typer(
    name="train",
    help="Commands for training models.",
    add_completion=False,
    no_args_is_help=True
)

# Calculate absolute config path relative to this file
# Assumes conf directory is at the project root, 3 levels up from this file's directory
# src/craft/cli/ -> src/craft/ -> src/ -> <root>
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
CONFIG_DIR_ABS = str(_PROJECT_ROOT / "conf")
CONFIG_NAME = "config" # Main config file name

# Register custom OmegaConf resolvers (Commented out as source file is missing)
# register_resolvers()
# enable_missing_variable_logging()

# --- Hydra Configuration Loading --- #
# This might be better placed in a shared CLI utility module if reused elsewhere

def load_hydra_config(
    config_path: str = "../../../conf", # Relative path from this file to conf dir
    config_name: str = "config", # Main config file name
    experiment_name: Optional[str] = None, # The experiment config to load from conf/experiment
    overrides: Optional[List[str]] = None
) -> DictConfig:
    """Loads Hydra configuration safely.

    Args:
        config_path: Relative path to the main configuration directory.
        config_name: Name of the main config file (e.g., 'config.yaml').
        experiment_name: Name of the experiment config under conf/experiment/ to load.
        overrides: List of command-line overrides (e.g., ['training.batch_size=64']).

    Returns:
        The composed OmegaConf DictConfig object.
    """
    if overrides is None:
        overrides = []

    # # Construct the absolute path to the config directory - REMOVED, not needed for initialize
    # abs_config_path = Path(__file__).parent.parent.parent / config_path
    # if not abs_config_path.is_dir():
    #     raise FileNotFoundError(f"Hydra config directory not found at: {abs_config_path}")

    # Add experiment override if provided
    if experiment_name:
        experiment_override = f"experiment={experiment_name}"
        # Avoid duplicate experiment overrides
        if not any(ov.startswith("experiment=") for ov in overrides):
             overrides.insert(0, experiment_override)
        else:
             logger.warning(f"Experiment '{experiment_name}' provided via arg, but also present in overrides. Using arg.")
             # Replace existing experiment override
             overrides = [ov for ov in overrides if not ov.startswith("experiment=")]
             overrides.insert(0, experiment_override)

    hydra.core.global_hydra.GlobalHydra.instance().clear() # Clear previous Hydra instance if any
    # Initialize Hydra using the relative config_path
    hydra.initialize(config_path=config_path, job_name="craft_cli_train", version_base=None) # Added version_base=None
    try:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        OmegaConf.resolve(cfg) # Resolve interpolations immediately
        logger.info(f"Hydra configuration loaded successfully for experiment: {cfg.get('experiment_name', 'N/A')}")
        logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")
        return cfg
    except Exception as e:
        logger.error(f"Error composing Hydra configuration: {e}", exc_info=True)
        raise

# --- Typer Command --- #

@train_app.command(
    name="run", # Or just keep it implicit if train_app() is called directly
    help="Run a training experiment based on configuration."
)
def run_training(
    ctx: typer.Context,
    experiment: Optional[str] = typer.Option(
        None,
        "--experiment", "-e",
        help="Name of the experiment configuration file (e.g., 'my_experiment') located in 'conf/experiment/'."
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Path to a checkpoint file or directory, or 'latest'/'best', to resume training from."
    ),
    compile_model: bool = typer.Option(
        False,
        "--compile",
        help="Enable model compilation with torch.compile()."
    ),
    overrides: Optional[List[str]] = typer.Argument(
        None,
        help="Hydra overrides for configuration (e.g., 'training.batch_size=32' 'optimizer.lr=1e-4')."
    ),
) -> Optional[TrainingState]:
    """Loads configuration and runs the main training process."""
    setup_logging() # Use basic logging setup
    logger.info("Starting training command...")

    try:
        # 1. Load Hydra Configuration
        cfg = load_hydra_config(
            experiment_name=experiment,
            overrides=overrides if overrides else []
        )

        # <<< DEBUG PRINT >>>
        logger.debug(f"Full composed cfg object:\n{OmegaConf.to_yaml(cfg)}") # Print the whole config
        # <<< END DEBUG PRINT >>>

        # 4. Instantiate Trainer (which will handle component instantiation & config validation)
        logger.info("--- Initializing Trainer (instantiating components from config) ---")
        trainer = Trainer(
            cfg=cfg, # Pass the raw OmegaConf DictConfig
            resume_from_checkpoint=resume_from,
            compile_model=compile_model
            # experiment_name=cfg.get("experiment_name", "default_exp") # Removed, Trainer gets from validated cfg
        )

        logger.info("--- Trainer Setup Complete ---")

        # 5. Run Training
        trainer.train()

        logger.info("Training finished successfully.")
        # TODO: Verify Trainer.state attribute and type hint
        return trainer.state # type: ignore[no-any-return, attr-defined]

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        raise typer.Exit(code=1)
    except OmegaConfErrors.ConfigKeyError as e:
        logger.error(f"Configuration key error: {e} - Please check your YAML files and overrides.")
        raise typer.Exit(code=1)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: \n{e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        logger.error(f"Value error during setup: {e}")
        logger.debug(traceback.format_exc()) # Add traceback for debugging
        raise typer.Exit(code=1)
    except ImportError as e:
        logger.error(f"Import error: {e}. Check installations and module paths.")
        logger.debug(traceback.format_exc())
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        # Consider cleanup or saving state here if needed
        raise typer.Exit(code=130) # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        # Log the full traceback
        # logger.exception("Traceback:") # Already done by exc_info=True
        raise typer.Exit(code=1)

# --- Helper Functions --- #

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

# Entry point for the train commands (if run as a script, though usually accessed via main cli)
if __name__ == "__main__":
    train_app()