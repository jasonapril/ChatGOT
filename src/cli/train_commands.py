# src/cli/train_commands.py
import typer
import logging
from typing import Optional, List
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from rich.console import Console

# Use absolute imports based on src being in PYTHONPATH or adjusted relative paths
# Assuming src/ is runnable via python -m src.cli.run
from ..models.factory import create_model_from_config
from ..data.base import prepare_dataloaders_from_config
from ..training.base import create_trainer_from_config
from ..utils.common import set_seed, setup_device

# Create Typer app for training commands
train_app = typer.Typer(help="Commands for model training")
console = Console() # Use a console instance, maybe import from run.py?
                    # For now, create a local one.
logger = logging.getLogger(__name__) # Get logger

@train_app.command("language")
def train_language_model(
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume training from checkpoint"),
    checkpoint_path: Optional[str] = typer.Option(None, "--checkpoint", help="Path to specific checkpoint for resuming"),
    overrides: List[str] = typer.Option(None, "--override", help="Hydra overrides (e.g., training=minimal_test)"),
):
    """Train a language model using Hydra, composing from conf/config.yaml."""
    
    # Define the path to the main config directory relative to this script's location
    # src/cli/train_commands.py -> conf/ is ../../conf
    # It's often more robust to use hydra.utils.get_original_cwd() or package resources
    # For simplicity assuming standard structure where conf is relative to project root
    
    # Determine absolute path to conf dir relative to this file
    # __file__ -> src/cli/train_commands.py
    # -> src/cli/ -> src/ -> project_root/
    conf_dir_abs = str(Path(__file__).parent.parent.parent / "conf")
    main_config_name = "config"

    console.print(f"Initializing Hydra with main config='{main_config_name}' from '{conf_dir_abs}'")
    if overrides:
        console.print(f"Applying overrides: {overrides}")

    # Manually initialize Hydra relative to the main config directory
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear() # Clear previous Hydra instance if any
        with hydra.initialize_config_dir(config_dir=conf_dir_abs, version_base=None):
            # Always compose the main 'config.yaml', applying command-line overrides
            cfg = hydra.compose(config_name=main_config_name, overrides=overrides or [])
            console.print("Hydra configuration composed successfully.")

            # Now use the composed cfg object
            output_dir = Path.cwd() # Hydra changes CWD
            console.print(f"Hydra run directory (CWD): {output_dir}")

            # Set seed
            set_seed(cfg.get("seed", 42))
            
            # Set up device
            device_name = cfg.get("system", {}).get("device", "auto") 
            device = setup_device(device_name)
            
            # Prepare data
            console.print(f"Preparing data from configuration...")
            train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders_from_config(
                data_config=cfg.data, 
                batch_size=cfg.data.batch_size, 
                num_workers=cfg.data.get("num_workers", 0)
            )
            
            # Create model
            console.print(f"Creating model from configuration...")
            model = create_model_from_config(cfg.model)
            
            # Create trainer
            console.print(f"Setting up trainer...")
            trainer_cfg = cfg.training
            OmegaConf.set_struct(trainer_cfg, False) # Allow modifications
            trainer_cfg.hydra_output_dir = str(output_dir) 
            trainer_cfg.device = str(device) 
            OmegaConf.set_struct(trainer_cfg, True)

            trainer = create_trainer_from_config(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                config=trainer_cfg,
            )
            
            # Resume training logic
            actual_checkpoint_path = None
            if resume:
                if checkpoint_path:
                    chkpt_path_obj = Path(checkpoint_path)
                    if chkpt_path_obj.exists():
                         actual_checkpoint_path = chkpt_path_obj
                    else:
                         if not chkpt_path_obj.is_absolute():
                              try:
                                   original_cwd = hydra.utils.get_original_cwd()
                                   resolved_path = Path(original_cwd) / checkpoint_path
                                   if resolved_path.exists():
                                        actual_checkpoint_path = resolved_path
                                   else:
                                        console.print(f"[yellow]Warning:[/yellow] Checkpoint path '{checkpoint_path}' (resolved to '{resolved_path}') not found.")
                              except Exception as e:
                                   console.print(f"[yellow]Warning:[/yellow] Error resolving relative checkpoint path: {e}")
                         else:
                              console.print(f"[yellow]Warning:[/yellow] Specified checkpoint path '{checkpoint_path}' not found.")
            
            if actual_checkpoint_path:
                console.print(f"Resuming training from checkpoint: {actual_checkpoint_path}")
                trainer.load_checkpoint(str(actual_checkpoint_path))
            elif resume:
                console.print("[yellow]Warning:[/yellow] Resume requested but no valid checkpoint found or specified.")

            # Train model
            console.print(f"Starting training...")
            metrics = trainer.train()
            
            console.print(f"Training completed!")
            if 'train_loss' in metrics and metrics['train_loss']:
                console.print(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
            if 'val_loss' in metrics and metrics['val_loss']:
                console.print(f"Final validation loss: {metrics['val_loss'][-1]:.4f}")

    except hydra.errors.ConfigCompositionException as e:
        logger.exception("Hydra configuration composition failed.")
        console.print(f"[bold red]Config Error:[/bold red] {e}")
        raise
    except Exception as e:
        logger.exception("Training failed.")
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise 