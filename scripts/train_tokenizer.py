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

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        raise ValueError("At least one input file must be provided for training.")
    
    # Check if input files exist
    for f in input_files:
        if not os.path.exists(f):
            raise ValueError(f"Input file not found: {f}")

    logger.info(f"Starting tokenizer training...")
    logger.info(f"  Type: {tokenizer_type}")
    logger.info(f"  Input Files: {input_files}")
    logger.info(f"  Output Directory: {output_dir}")
    logger.info(f"  Vocab Size: {vocab_size}")
    
    config = {
        "vocab_size": vocab_size,
        "model_prefix": model_prefix, # SP needs this
        **kwargs # Pass through other args like special tokens if provided
    }

    tokenizer: BaseTokenizer
    if tokenizer_type == "subword":
        logger.info("Initializing SubwordTokenizer (BPE)...")
        tokenizer = SubwordTokenizer(config=config)
    elif tokenizer_type == "sentencepiece":
        logger.info("Initializing SentencePieceTokenizer (BPE)...")
        tokenizer = SentencePieceTokenizer(config=config)
        logger.info(f"  SentencePiece Model Prefix: {model_prefix}")
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}. Choose 'subword' or 'sentencepiece'.")

    try:
        logger.info(f"Calling {tokenizer_type} train method...")
        tokenizer.train(files=input_files, output_dir=output_dir)
        logger.info(f"Tokenizer training completed successfully. Model saved to {output_dir}")
        
        # Optionally, save the simple config used (without extra kwargs for now)
        # This helps remember the settings used for this specific training run.
        # config_save_path = os.path.join(output_dir, 'training_config.json')
        # try:
        #     import json
        #     basic_config_to_save = {'tokenizer_type': tokenizer_type, 'vocab_size': vocab_size, 'model_prefix': model_prefix if tokenizer_type == 'sentencepiece' else None}
        #     with open(config_save_path, 'w', encoding='utf-8') as f:
        #         json.dump(basic_config_to_save, f, indent=4)
        #     logger.info(f"Basic training config saved to {config_save_path}")
        # except Exception as e:
        #     logger.warning(f"Could not save basic training config: {e}")

        return tokenizer

    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}", exc_info=True)
        raise # Re-raise the exception after logging

if __name__ == "__main__":
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
    
    args = parser.parse_args()
    
    # Convert args to dict, filtering out None values if necessary
    kwargs_for_tokenizer = vars(args).copy()
    # Remove args not directly part of tokenizer config (handled separately)
    del kwargs_for_tokenizer['tokenizer_type']
    del kwargs_for_tokenizer['input_files']
    del kwargs_for_tokenizer['output_dir']
    # Remove vocab_size as well, since it's passed explicitly
    del kwargs_for_tokenizer['vocab_size']
    # Remove model_prefix too, as it's also passed explicitly
    del kwargs_for_tokenizer['model_prefix']
    # Keep model_prefix in kwargs for simplicity if not passed explicitly later

    # Filter out any keys with None values if desired, or handle defaults in tokenizer init
    # kwargs_for_tokenizer = {k: v for k, v in kwargs_for_tokenizer.items() if v is not None}

    train_tokenizer(
        tokenizer_type=args.tokenizer_type,
        input_files=args.input_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        **kwargs_for_tokenizer # Pass remaining args
    )

    logger.info("Script finished.") 