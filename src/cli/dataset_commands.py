# src/cli/dataset_commands.py
import typer
import logging
from typing import Optional

# Create Typer app for dataset commands
dataset_app = typer.Typer(help="Commands for dataset operations")
logger = logging.getLogger(__name__) # Get logger

@dataset_app.command("prepare")
def prepare_dataset(
    input_file: str = typer.Option(..., "--input", "-i", help="Input data file"),
    output_dir: str = typer.Option("data/processed", "--output-dir", "-o", help="Output directory"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Data processing configuration"),
):
    """Prepare a dataset for training."""
    # Use absolute import based on src being in PYTHONPATH or adjusted relative paths
    from ..data.processors import prepare_data 
    
    # Load configuration if provided
    config = None
    if config_path:
        # This function was removed, so comment out its usage
        # config = load_experiment_config(config_path)
        logging.warning(f"Config path ({config_path}) provided to 'dataset prepare', but config loading is currently disabled in CLI.")
        pass
    
    # Prepare data
    try:
         prepare_data(input_file, output_dir, config)
         logger.info(f"Dataset preparation complete for {input_file}. Output in {output_dir}")
    except Exception as e:
         logger.exception(f"Dataset preparation failed for {input_file}.")
         raise 