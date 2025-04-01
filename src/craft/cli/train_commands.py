# src/cli/train_commands.py
import typer
import logging
import sys
import subprocess
from typing import List, Optional

# Rich console for better output formatting
from rich.console import Console

# --- Remove unused imports ---
# from pathlib import Path
# import hydra
# from omegaconf import DictConfig, OmegaConf
# from src.utils import set_seed, setup_device
# from src.data.base import prepare_dataloaders_from_config # Now handled by train_runner
# from src.models.factory import create_model_from_config # Now handled by train_runner
# from src.training.base import create_trainer_from_config # Now handled by train_runner

# Get the logger for this module
logger = logging.getLogger(__name__)
console = Console()

# Create a Typer app for training commands
train_app = typer.Typer()

@train_app.command("language")
def train_language_model(
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume training from latest/specified checkpoint."),
    checkpoint_path: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Specific checkpoint path to resume from."),
    # Accept overrides as a list of strings
    overrides: Optional[List[str]] = typer.Option(None, "--override", help="Hydra config overrides (e.g., 'experiment=chatgot_100m' 'data.batch_size=16').") 
):
    """
    Train a language model using the configuration defined by an experiment.
    Delegates to src.training.train_runner.py.
    
    Example: 
        python -m src.cli.run train language --override experiment=chatgot_100m
        python -m src.cli.run train language --override experiment=chatgot_100m --override data.batch_size=16
        python -m src.cli.run train language --override experiment=chatgot_100m --resume
        python -m src.cli.run train language --override experiment=chatgot_100m --checkpoint path/to/checkpoint.pt
    """
    console.print("Initiating training via src.training.train_runner.py...")

    # --- Construct Command for train_runner.py --- 
    command = [
        sys.executable,  # Use the same python interpreter
        "-m",
        "src.training.train_runner" # Target script
    ]
    
    # --- Handle Overrides --- 
    # Start with overrides passed via the --override option
    final_overrides = overrides if overrides is not None else []
    
    # Add resume/checkpoint override logic
    if resume and checkpoint_path:
        console.print(f"[yellow]Warning:[/yellow] Both --resume and --checkpoint provided. Using --checkpoint: {checkpoint_path}")
        final_overrides.append(f"resume_from={checkpoint_path}")
    elif resume:
        console.print("Resume requested. Setting resume_from=latest.")
        final_overrides.append("resume_from=latest")
    elif checkpoint_path:
         console.print(f"Specific checkpoint provided. Setting resume_from={checkpoint_path}.")
         final_overrides.append(f"resume_from={checkpoint_path}")
    # else: no resume override needed
    
    command.extend(final_overrides) # Add all collected overrides

    # --- Execute train_runner.py --- 
    console.print(f"Running command: {' '.join(command)}")
    try:
        # Use subprocess.run, stream output directly
        process = subprocess.run(
            command,
            check=True, # Raise exception on non-zero exit code
            text=True, # Decode stdout/stderr as text
            stdout=sys.stdout, # Redirect runner's stdout to cli's stdout
            stderr=sys.stderr  # Redirect runner's stderr to cli's stderr
        )
        console.print("[green]Training runner script finished successfully.[/green]")
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Could not find Python executable '{sys.executable}'.")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error:[/bold red] Training runner script failed with exit code {e.returncode}.")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        logger.exception("Unexpected error while running train_runner.py")

    # Removed the old direct training logic
    # ... (old hydra init, data prep, model creation, trainer code) ... 