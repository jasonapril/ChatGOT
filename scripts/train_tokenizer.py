"""
Script to train a tokenizer (Subword BPE or SentencePiece) on a dataset.
"""
import argparse
import logging
import os
import sys
from typing import List

# Add project root to path to allow importing craft modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from craft.data.tokenizers import SubwordTokenizer, SentencePieceTokenizer, BaseTokenizer

# --- Setup Enhanced Logging --- #
# Set root logger level to DEBUG to capture more info
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(name)s][%(levelname)s] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to stdout
logger = logging.getLogger(__name__)
logger.info("--- Logger configured for DEBUG level --- ")
# --- End Enhanced Logging --- #

def train_tokenizer(
    tokenizer_type: str,
    input_files: List[str],
    output_dir: str,
    vocab_size: int,
    model_prefix: str = "tokenizer_model", # Only used by SentencePiece
    **kwargs # Catch other potential config args
) -> BaseTokenizer:
    """
    Initializes and trains the specified tokenizer.

    Args:
        tokenizer_type: Type of tokenizer ('subword' or 'sentencepiece').
        input_files: List of paths to input text files for training.
        output_dir: Directory to save the trained tokenizer files.
        vocab_size: Desired vocabulary size.
        model_prefix: Prefix for model files (used by SentencePiece).
        kwargs: Additional arguments for tokenizer configuration.

    Returns:
        The trained tokenizer instance.
        
    Raises:
        ValueError: If an unsupported tokenizer type is provided or input files are missing.
    """
    if not input_files:
        logger.error("Input file list cannot be empty for training.")
        raise ValueError("At least one input file must be provided for training.")
    
    # Check if input files exist
    logger.debug(f"Checking existence of input files: {input_files}")
    for f in input_files:
        if not os.path.exists(f):
            logger.error(f"Input file not found: {f}")
            raise ValueError(f"Input file not found: {f}")
        logger.debug(f"Input file confirmed: {f}")

    logger.info(f"Starting tokenizer training...")
    logger.info(f"  Type: {tokenizer_type}")
    logger.info(f"  Input Files: {input_files}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Vocab Size: {vocab_size}")
    logger.debug(f"  Additional Kwargs: {kwargs}")
    
    config = {
        "vocab_size": vocab_size,
        "model_prefix": model_prefix, # SP needs this
        **kwargs # Pass through other args like special tokens if provided
    }

    tokenizer: BaseTokenizer
    if tokenizer_type == "subword":
        logger.info("Initializing SubwordTokenizer (BPE)...")
        # Pass config directly to SubwordTokenizer
        tokenizer = SubwordTokenizer(config=config)
    elif tokenizer_type == "sentencepiece":
        logger.info("Initializing SentencePieceTokenizer...")
         # Ensure model_prefix is passed if using SentencePiece
        config['model_prefix'] = model_prefix # Explicitly set for SP
        tokenizer = SentencePieceTokenizer(config=config) 
        logger.info(f"  SentencePiece Model Prefix set to: {model_prefix}")
    else:
        logger.error(f"Unsupported tokenizer type provided: {tokenizer_type}")
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}. Choose 'subword' or 'sentencepiece'.")

    try:
        logger.info(f"Calling {tokenizer_type}.train method with output_dir: {output_dir}")
        tokenizer.train(files=input_files, output_dir=output_dir)
        logger.info(f"Tokenizer training reported success.")
        
        # Verify output directory and file existence after training
        logger.info(f"Verifying output directory: {output_dir}")
        if not os.path.isdir(output_dir):
             logger.error(f"Output directory {output_dir} was NOT created after training.")
             # Optional: attempt to create it now?
             # try:
             #     os.makedirs(output_dir, exist_ok=True)
             #     logger.info(f"Manually created output directory: {output_dir}")
             # except OSError as e:
             #     logger.error(f"Failed to manually create output directory {output_dir}: {e}")
        else:
             logger.info(f"Output directory verified: {output_dir}")
             # Further check for specific files if needed (e.g., tokenizer.json for Subword)
             expected_file = os.path.join(output_dir, 'tokenizer.json') if tokenizer_type == 'subword' else os.path.join(output_dir, f'{model_prefix}.model')
             logger.info(f"Verifying expected output file: {expected_file}")
             if not os.path.exists(expected_file):
                 logger.error(f"Expected output file {expected_file} does NOT exist after training.")
             else:
                 logger.info(f"Expected output file verified: {expected_file}")

        logger.info(f"Tokenizer training process finished. Model expected in {output_dir}")
        return tokenizer

    except Exception as e:
        logger.exception(f"Tokenizer training failed during the train call: {e}") # Use logger.exception for traceback
        raise # Re-raise the exception after logging

if __name__ == "__main__":
    # --- Sanity Check Print --- #
    print("--- train_tokenizer.py script started --- ", flush=True)
    # --- End Sanity Check Print --- #

    parser = argparse.ArgumentParser(description="Train a tokenizer (Subword BPE or SentencePiece).")
    
    parser.add_argument(
        "--tokenizer_type", 
        type=str, 
        required=True, 
        choices=["subword", "sentencepiece"],
        help="Type of tokenizer to train."
    )
    parser.add_argument(
        "--input_files", 
        type=str, 
        required=True, 
        nargs='+', # Allows multiple input files
        help="Path(s) to the input text file(s) for training."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the trained tokenizer model and config."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        required=True, 
        help="Desired vocabulary size for the tokenizer."
    )
    parser.add_argument(
        "--model_prefix", 
        type=str, 
        default="tokenizer_model", 
        help="Model prefix name (primarily for SentencePiece)."
    )
    # Add arguments for special tokens if needed (optional override)
    # parser.add_argument("--pad_token", type=str, default="<pad>", help="Padding token.")
    # parser.add_argument("--unk_token", type=str, default="<unk>", help="Unknown token.")
    # parser.add_argument("--bos_token", type=str, default="<s>", help="Beginning-of-sequence token.")
    # parser.add_argument("--eos_token", type=str, default="</s>", help="End-of-sequence token.")
    
    # Add verbose flag
    # parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()
    
    # Reconfigure logger level if verbose flag was added and set
    # if args.verbose:
    #     logging.getLogger().setLevel(logging.DEBUG)
    #     logger.info("Verbose logging enabled.")
    # else:
    #     logging.getLogger().setLevel(logging.INFO)

    # Prepare kwargs dictionary (ensure it doesn't overwrite explicit args)
    kwargs_for_tokenizer = vars(args).copy()
    explicit_args = ['tokenizer_type', 'input_files', 'output_dir', 'vocab_size', 'model_prefix']
    for arg_name in explicit_args:
        if arg_name in kwargs_for_tokenizer:
            del kwargs_for_tokenizer[arg_name]

    try:
        train_tokenizer(
            tokenizer_type=args.tokenizer_type,
            input_files=args.input_files,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            model_prefix=args.model_prefix,
            **kwargs_for_tokenizer # Pass remaining args like special tokens if provided via CLI
        )
        logger.info("Script finished successfully.")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1) # Exit with error code if training fails 