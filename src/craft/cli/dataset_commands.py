# src/cli/dataset_commands.py
import typer
import logging
import os
import glob
import traceback
from typing import Optional

# Create Typer app for dataset commands
dataset_app = typer.Typer(help="Commands for dataset operations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Get logger

@dataset_app.command("prepare")
def prepare_dataset(
    input_file: str = typer.Option(..., "--input", "-i", help="Input data file"),
    output_dir: str = typer.Option("data/processed", "--output-dir", "-o", help="Output directory"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Data processing configuration"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing by deleting existing files"),
):
    """Prepare a dataset for training."""
    try:
        # Use absolute import based on src being in PYTHONPATH or adjusted relative paths
        from ..data.processors import prepare_data 
        
        # Load configuration if provided
        config = None
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            logger.info(f"Configuration: {config}")
        
        # Process with character-level tokenization
        char_output_dir = os.path.join(output_dir, "char_level")
        logger.info(f"Creating character-level output directory: {char_output_dir}")
        os.makedirs(char_output_dir, exist_ok=True)
        
        if force and os.path.exists(char_output_dir):
            logger.info(f"Cleaning up existing character-level files in {char_output_dir}")
            # Delete all .pkl files
            for pkl_file in glob.glob(os.path.join(char_output_dir, "*.pkl")):
                os.remove(pkl_file)
                logger.info(f"Deleted {pkl_file}")
        
        # Create character-level config
        char_config = {"data": {"type": "text", "format": "character"}}
        logger.info(f"Using character-level config: {char_config}")
        try:
            char_output_paths = prepare_data(input_file, char_output_dir, char_config)
            logger.info(f"Character-level dataset preparation complete. Output in {char_output_dir}")
            logger.info(f"Character-level output paths: {char_output_paths}")
            
            # Verify the files were created
            for path in char_output_paths.values():
                if not os.path.exists(path):
                    logger.error(f"Expected file was not created: {path}")
                else:
                    logger.info(f"Verified file exists: {path}")
                    
        except Exception as e:
            logger.exception(f"Character-level dataset preparation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Process with subword tokenization
        if config:
            subword_output_dir = os.path.join(output_dir, "subword_level")
            logger.info(f"Creating subword-level output directory: {subword_output_dir}")
            os.makedirs(subword_output_dir, exist_ok=True)
            
            if force and os.path.exists(subword_output_dir):
                logger.info(f"Cleaning up existing subword-level files in {subword_output_dir}")
                # Delete all .pkl files
                for pkl_file in glob.glob(os.path.join(subword_output_dir, "*.pkl")):
                    os.remove(pkl_file)
                    logger.info(f"Deleted {pkl_file}")
                # Delete tokenizer files
                tokenizer_dir = os.path.join(subword_output_dir, "tokenizer")
                if os.path.exists(tokenizer_dir):
                    for file in os.listdir(tokenizer_dir):
                        os.remove(os.path.join(tokenizer_dir, file))
                        logger.info(f"Deleted {file} from tokenizer directory")
            
            try:
                subword_output_paths = prepare_data(input_file, subword_output_dir, config)
                logger.info(f"Subword-level dataset preparation complete. Output in {subword_output_dir}")
                logger.info(f"Subword-level output paths: {subword_output_paths}")
                
                # Verify the files were created
                for path in subword_output_paths.values():
                    if not os.path.exists(path):
                        logger.error(f"Expected file was not created: {path}")
                    else:
                        logger.info(f"Verified file exists: {path}")
                        
            except Exception as e:
                logger.exception(f"Subword-level dataset preparation failed: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
    except Exception as e:
        logger.exception(f"Dataset preparation failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 