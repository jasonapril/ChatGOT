# src/cli/dataset_commands.py
import typer
import logging
import os
import sys
import glob
import traceback
import pickle
import numpy as np
import json
from typing import Optional, List, Tuple, cast # Added cast
from pathlib import Path

# Import the SentencePiece training function
from ..data.tokenizers.sentencepiece_trainer import train_sentencepiece_model
# Import SentencePieceTokenizer for prepare command
from ..data.tokenizers.sentencepiece import SentencePieceTokenizer
# Import char processor
from ..data.char_processor import process_char_data
# Import IO utils
from ..utils.io import ensure_directory

# Create Typer app for dataset commands
dataset_app = typer.Typer(help="Commands for dataset operations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = typer.echo

@dataset_app.command("train-tokenizer")
def train_tokenizer_command(
    # Accept comma-separated string directly
    input_files: str = typer.Option(..., "--input-files", "-i", help="Comma-separated list of input text file paths."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to save the trained tokenizer model and vocab files.", file_okay=False, resolve_path=True),
    vocab_size: int = typer.Option(..., "--vocab-size", "-v", help="Vocabulary size for the tokenizer."),
    model_prefix: str = typer.Option("spm", "--model-prefix", help="Prefix for the output model and vocab files (e.g., 'spm')."),
    model_type: str = typer.Option("bpe", "--model-type", help="SentencePiece model type (e.g., 'bpe', 'unigram', 'char', 'word')."),
    character_coverage: float = typer.Option(0.9995, "--char-coverage", help="Character coverage for SentencePiece training."),
    num_threads: Optional[int] = typer.Option(None, "--num-threads", help="Number of threads for SentencePiece training (uses os.cpu_count() if None)."),
    force: bool = typer.Option(False, "--force", "-f", help="Force training even if output files already exist."),
) -> None: # Added return type hint
    """Train a SentencePiece tokenizer model.
    
    Example:
        python -m craft.cli.run dataset train-tokenizer -i data/input1.txt,data/input2.txt -o data/tokenizer -v 32000
    """
    console(f"Starting SentencePiece tokenizer training...")
    # Basic validation on input_files string
    if not input_files:
         logger.error("--input-files cannot be empty.")
         raise typer.Exit(code=1)
    # Log the files being used
    logger.info(f"  Input Files: {input_files}") 
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Vocab Size: {vocab_size}")
    logger.info(f"  Model Type: {model_type}")
    logger.info(f"  Model Prefix: {model_prefix}")
    logger.info(f"  Character Coverage: {character_coverage}")
    logger.info(f"  Num Threads: {'Default (os.cpu_count)' if num_threads is None else num_threads}")
    logger.info(f"  Force: {force}")

    try:
        # Pass arguments directly to the updated function
        model_path: Path
        vocab_path: Path
        model_path, vocab_path = train_sentencepiece_model(
            input_files=input_files, 
            output_dir=output_dir,
            vocab_size=vocab_size,
            model_prefix=model_prefix,
            model_type=model_type,
            character_coverage=character_coverage,
            num_threads=num_threads, # Pass Optional[int] directly
            force=force
        )
        
        console(f"SentencePiece tokenizer training successful!")
        console(f"  Model saved to: {model_path}")
        console(f"  Vocabulary saved to: {vocab_path}")

    except ImportError as e:
        if "sentencepiece" in str(e):
             logger.error("SentencePiece library not found. Please install it: pip install sentencepiece")
        else:
             logger.exception(f"Import error: {e}")
        raise typer.Exit(code=1)
    except FileExistsError as e:
         logger.error(f"Error: {e}") # Message already includes details
         raise typer.Exit(code=1)
    except FileNotFoundError as e:
         logger.error(f"Error: {e}")
         raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during tokenizer training: {str(e)}")
        raise typer.Exit(code=1)

@dataset_app.command("prepare")
def prepare_dataset(
    input_path: Path = typer.Option(..., "--input-path", "-i", help="Path to the raw input data file.", exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory to save the processed data.", file_okay=False, resolve_path=True),
    type: str = typer.Option("char", "--type", "-t", help="Type of processing: 'char' or 'subword'.", case_sensitive=False),
    split_ratios: Optional[str] = typer.Option(None, "--split-ratios", help="Train/Val/Test ratios (e.g., '0.9,0.05,0.05'). Default for char: 0.9,0.05,0.05. Required for subword.", callback=lambda v: [float(x) for x in v.split(',')] if v else None),
    tokenizer_path: Optional[Path] = typer.Option(None, "--tokenizer-path", help="Required for type='subword'. Path to the trained SentencePiece tokenizer model prefix (e.g., /path/to/spm)."), # Clarified help text
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing by deleting existing files/dirs in the output directory"),
) -> None: # Added return type hint
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
        elif isinstance(split_ratios, list) and len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6:
            splits_tuple = cast(Tuple[float, float, float], tuple(split_ratios))
            logger.info(f"Using provided char split ratios: {splits_tuple}")
        else:
            logger.error(f"Invalid split ratios for char type: {split_ratios}. Must be 3 numbers summing to 1.0 or omitted.")
            raise typer.Exit(code=1)
    elif type == 'subword':
        if not isinstance(split_ratios, list) or len(split_ratios) != 3 or not abs(sum(split_ratios) - 1.0) < 1e-6:
             logger.error(f"Invalid split ratios for subword type: {split_ratios}. Must provide 3 numbers summing to 1.0 via --split-ratios.")
             raise typer.Exit(code=1)
        splits_tuple = cast(Tuple[float, float, float], tuple(split_ratios))
        logger.info(f"Using provided subword split ratios: {splits_tuple}")
    else:
         logger.error(f"Invalid processing type specified: {type}")
         raise typer.Exit(code=1)
    # ------------------------ #

    try:
        # Ensure output directory exists (pass string)
        ensure_directory(str(output_dir))
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
            # Delete tokenizer directory (Only makes sense for char type where tokenizer is saved within output)
            if type == 'char':
                tokenizer_dir = output_dir / "tokenizer"
                if tokenizer_dir.is_dir():
                    try:
                        import shutil
                        shutil.rmtree(tokenizer_dir); logger.info(f"Deleted char tokenizer directory: {tokenizer_dir}")
                    except OSError as e: logger.error(f"Error deleting char tokenizer directory {tokenizer_dir}: {e}")

        # Execute the appropriate data preparation
        logger.info(f"Starting data preparation with type: {type}")

        if type == 'char':
            output_paths = process_char_data(
                input_path=str(input_path),
                output_dir=str(output_dir),
                splits=splits_tuple
            )

        elif type == 'subword':
            # 1. Load Tokenizer
            # Tokenizer path now points to the model prefix (e.g., /path/to/spm), not a directory
            logger.info(f"Loading SentencePiece tokenizer model from prefix: {tokenizer_path}")
            try:
                # Ensure tokenizer_path is not None before passing to str()
                if tokenizer_path is None:
                     raise ValueError("tokenizer_path cannot be None for subword type.")
                # Resolve to absolute path before loading and storing
                tokenizer_abs_path = tokenizer_path.resolve()
                tokenizer = SentencePieceTokenizer.load_from_prefix(str(tokenizer_abs_path))
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
            split_data_map = {'train': train_ids, 'val': val_ids, 'test': test_ids}
            for split_name, split_data in split_data_map.items(): # Use consistent name split_data_map
                if len(split_data) == 0:
                     logger.warning(f"Split '{split_name}' has size 0. Skipping save.")
                     continue
                output_filepath = output_dir / f"{split_name}.pkl"
                logger.info(f"Saving {split_name} split to {output_filepath}...")
                try:
                    with open(output_filepath, 'wb') as f:
                        pickle.dump(split_data, f)
                    output_paths[split_name] = str(output_filepath)
                except IOError as e:
                     logger.error(f"Failed to save {split_name} split to {output_filepath}: {e}")
                     raise typer.Exit(code=1)

            # 6. Prepare and Save Metadata
            logger.info("Preparing and saving metadata.json for subword dataset...")
            metadata = {
                'input_file': str(input_path.resolve()),
                'data_format': 'subword',
                'vocab_size': tokenizer.get_vocab_size(),
                'tokenizer_type': 'SentencePiece',
                'tokenizer_model_path': str(tokenizer_abs_path), # Store absolute path
                'total_tokens': n,
                'split_ratios': list(splits_tuple),
                'split_sizes': {
                    'train': len(train_ids),
                    'val': len(val_ids),
                    'test': len(test_ids)
                 },
            }
            metadata_path = output_dir / "metadata.json"
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"Saved metadata to {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to save metadata.json: {e}", exc_info=True)
                # Optionally raise error or just warn
                raise typer.Exit(code=1) # Exit if metadata fails, as it's needed

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