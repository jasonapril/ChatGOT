# src/cli/dataset_commands.py
import typer
import logging
import os
import sys # Import sys
import glob
import traceback
import pickle # Import pickle
import numpy as np # Import numpy
from typing import Optional, List, Tuple # Import List, Tuple
# import yaml # No longer needed if config file isn't used
from pathlib import Path # Use Path

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
    input_path: Path = typer.Option(..., "--input-path", "-i", help="Path to the raw input data file.", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to save the processed data.", file_okay=False, resolve_path=True),
    type: str = typer.Option("char", "--type", "-t", help="Type of processing: 'char' or 'subword'.", case_sensitive=False),
    split_ratios: Optional[List[float]] = typer.Option(None, "--split-ratios", help="Train/Val/Test ratios (e.g., '0.9,0.05,0.05'). Default for char: 0.9,0.05,0.05. Required for subword.", callback=lambda v: [float(x) for x in v.split(',')] if v else None),
    tokenizer_path: Optional[Path] = typer.Option(None, "--tokenizer-path", help="Required for type='subword'. Path to the directory containing the pre-trained tokenizer files.", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing by deleting existing files/dirs in the output directory"),
):
    """Prepare a dataset for training (char or subword tokenization and splitting)."""
    logger.info(f"Starting dataset preparation...")
    logger.info(f"  Input Path: {input_path}")
    logger.info(f"  Output Dir: {output_dir}")
    logger.info(f"  Processing Type: {type}")

    # --- Argument Validation --- #
    if type == "subword" and not tokenizer_path:
        logger.error("--tokenizer-path is required when --type='subword'")
        raise typer.Exit(code=1)
    if type == "char" and tokenizer_path:
        logger.warning("--tokenizer-path is ignored when --type='char'")

    # Validate and set default splits
    splits_tuple: Tuple[float, float, float]
    if type == 'char':
        if split_ratios is None:
            splits_tuple = (0.9, 0.05, 0.05)
            logger.info(f"Using default char split ratios: {splits_tuple}")
        elif len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6:
            splits_tuple = tuple(split_ratios)
            logger.info(f"Using provided char split ratios: {splits_tuple}")
        else:
            logger.error(f"Invalid split ratios for char type: {split_ratios}. Must be 3 numbers summing to 1.0 or omitted.")
            raise typer.Exit(code=1)
    elif type == 'subword':
        if split_ratios is None or len(split_ratios) != 3 or not abs(sum(split_ratios) - 1.0) < 1e-6:
             logger.error(f"Invalid split ratios for subword type: {split_ratios}. Must provide 3 numbers summing to 1.0 via --split-ratios.")
             raise typer.Exit(code=1)
        splits_tuple = tuple(split_ratios)
        logger.info(f"Using provided subword split ratios: {splits_tuple}")
    else:
         logger.error(f"Invalid processing type specified: {type}")
         raise typer.Exit(code=1)
    # ------------------------ #

    try:
        # Use absolute import based on src being in PYTHONPATH or adjusted relative paths
        from ..data.char_processor import process_char_data
        from ..data.tokenizers.sentencepiece import SentencePieceTokenizer # Assuming only SP for subword now
        from ..utils.io import ensure_directory

        # Ensure output directory exists
        ensure_directory(output_dir)
        logger.info(f"Ensured output directory exists: {output_dir}")

        # Handle --force: Clean the target directory
        if force and output_dir.exists():
            logger.warning(f"Force flag set. Cleaning up existing files/dirs in {output_dir}")
            # Delete .pkl files
            for pkl_file in output_dir.glob("*.pkl"):
                try: pkl_file.unlink(); logger.info(f"Deleted {pkl_file}")
                except OSError as e: logger.error(f"Error deleting file {pkl_file}: {e}")
            # Delete metadata.json
            metadata_file = output_dir / "metadata.json"
            if metadata_file.exists():
                 try: metadata_file.unlink(); logger.info(f"Deleted {metadata_file}")
                 except OSError as e: logger.error(f"Error deleting file {metadata_file}: {e}")
            # Delete tokenizer directory
            tokenizer_dir = output_dir / "tokenizer"
            if tokenizer_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(tokenizer_dir); logger.info(f"Deleted tokenizer directory: {tokenizer_dir}")
                except OSError as e: logger.error(f"Error deleting tokenizer directory {tokenizer_dir}: {e}")

        # Execute the appropriate data preparation
        logger.info(f"Starting data preparation with type: {type}")

        if type == 'char':
            output_paths = process_char_data(
                input_path=str(input_path),
                output_dir=str(output_dir),
                splits=splits_tuple
            )

        elif type == 'subword':
            # --- Subword logic replicated from script --- #
            # 1. Load Tokenizer
            logger.info(f"Loading SentencePiece tokenizer from: {tokenizer_path}")
            try:
                tokenizer = SentencePieceTokenizer.load(str(tokenizer_path))
                logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}", exc_info=True)
                raise typer.Exit(code=1)

            # 2. Read input data
            logger.info(f"Reading raw data file {input_path}...")
            try:
                text = input_path.read_text(encoding='utf-8')
                logger.info(f"Read {len(text):,} characters.")
            except Exception as e:
                logger.error(f"Could not read input file {input_path}: {e}")
                raise typer.Exit(code=1)

            # 3. Encode data
            logger.info("Encoding data...")
            token_ids = tokenizer.encode(text)
            token_ids_np = np.array(token_ids, dtype=np.int32)
            logger.info(f"Generated {len(token_ids_np):,} tokens.")

            # 4. Split data
            n = len(token_ids_np)
            train_end = int(splits_tuple[0] * n)
            val_end = train_end + int(splits_tuple[1] * n)
            train_ids = token_ids_np[:train_end]
            val_ids = token_ids_np[train_end:val_end]
            test_ids = token_ids_np[val_end:]
            logger.info(f"Split sizes: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")

            # 5. Save splits to .pkl files
            output_paths = {}
            for split_name, split_data in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
                output_filepath = output_dir / f"{split_name}.pkl"
                logger.info(f"Saving {split_name} split to {output_filepath}...")
                try:
                    with open(output_filepath, 'wb') as f:
                        pickle.dump(split_data, f)
                    output_paths[split_name] = str(output_filepath)
                except IOError as e:
                     logger.error(f"Failed to save {split_name} split to {output_filepath}: {e}")
                     raise typer.Exit(code=1)
            # --- End Subword Logic --- #

        logger.info(f"Dataset preparation complete. Output in {output_dir}")
        logger.info(f"Output files created: {list(output_paths.values())}")

    except ImportError as e:
        logger.exception(f"Failed to import necessary modules. Ensure 'craft' is installed correctly: {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
         logger.exception(f"Input file or configuration file not found: {e}")
         raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during dataset preparation: {str(e)}")
        raise typer.Exit(code=1) 