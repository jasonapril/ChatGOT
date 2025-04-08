# src/craft/data/tokenizers/sentencepiece_trainer.py
import logging
import json
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, cast

import sentencepiece as spm # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

DEFAULT_SPECIAL_TOKENS = {
    # Define common defaults here if needed, consistent with base.py
    "pad": "<pad>",
    "bos": "<s>",
    "eos": "</s>",
    "unk": "<unk>",
    # Add others like mask, sep if relevant
}

def train_sentencepiece_model(
    # Renamed from input_file to input_files to match CLI and SP behavior
    input_files: str, # Comma-separated string of file paths
    output_dir: Union[str, Path],
    vocab_size: int,
    model_prefix: str, # Prefix for the output model/vocab files
    model_type: str = "unigram", # e.g., 'unigram', 'bpe', 'char', 'word'
    character_coverage: float = 0.9995,
    # Use a map for explicit special token definition during training
    special_tokens_map: Optional[Dict[str, str]] = None,
    # Explicit flags for how SP handles default special tokens
    add_bos_as_control: bool = True, # Corresponds to --control_symbols for BOS
    add_eos_as_control: bool = True, # Corresponds to --control_symbols for EOS
    add_unk_as_control: bool = True, # Corresponds to --control_symbols for UNK
    user_defined_symbols: Optional[List[str]] = None, # For tokens not in the standard map
    num_threads: Optional[int] = None, # Added
    force: bool = False, # Added
    # Add other relevant spm training args as needed
    # e.g., max_sentence_length, etc.
) -> Tuple[Path, Path]: # Changed return type
    """
    Trains a SentencePiece model using the provided configuration and saves it.

    Args:
        input_files: Comma-separated string of paths to input text corpora.
        output_dir: Directory to save the trained model (.model, .vocab) and metadata (.json).
        vocab_size: The target vocabulary size.
        model_prefix: Prefix for the output filenames (e.g., 'sp_model').
        model_type: SentencePiece model type ('unigram', 'bpe', 'char', 'word').
        character_coverage: Character coverage parameter for training.
        special_tokens_map: Dictionary mapping roles (e.g., 'pad', 'unk') to token strings.
                             Defaults will be used if not provided.
        add_bos_as_control: Treat the BOS token as a control symbol in SP training.
        add_eos_as_control: Treat the EOS token as a control symbol in SP training.
        add_unk_as_control: Treat the UNK token as a control symbol in SP training.
        user_defined_symbols: List of additional symbols to treat as special tokens.
        num_threads: Number of threads for training. Defaults to os.cpu_count().
        force: Overwrite existing model/vocab files if True.

    Returns:
        Tuple containing the Path to the created model file and vocab file.

    Raises:
        FileNotFoundError: If input files are not found.
        FileExistsError: If output files exist and force is False.
        ValueError: If required arguments are missing.
        RuntimeError: If SentencePiece training fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path_prefix = output_dir / model_prefix
    model_file_path = output_dir / f"{model_prefix}.model"
    vocab_file_path = output_dir / f"{model_prefix}.vocab"

    # Check input files exist (SP handles comma-separated string internally)
    # Basic check on the first file listed as a sanity check
    first_input_file = Path(input_files.split(',')[0].strip())
    if not first_input_file.is_file():
        raise FileNotFoundError(f"First input training file not found: {first_input_file}")

    # Check if output files exist and handle --force
    if not force:
        if model_file_path.exists() or vocab_file_path.exists():
            raise FileExistsError(
                f"Output file(s) already exist: {model_file_path} or {vocab_file_path}. Use --force to overwrite."
            )
    elif model_file_path.exists():
         logger.warning(f"Overwriting existing model file: {model_file_path}")
         model_file_path.unlink()
    elif vocab_file_path.exists():
         logger.warning(f"Overwriting existing vocab file: {vocab_file_path}")
         vocab_file_path.unlink()

    # --- Prepare Training Arguments for SentencePiece --- #
    # Final map combining defaults and user overrides
    final_special_map = DEFAULT_SPECIAL_TOKENS.copy()
    if special_tokens_map:
        final_special_map.update(special_tokens_map)

    # Construct control_symbols list based on flags and map
    control_symbols = []
    if add_bos_as_control and final_special_map.get("bos"):
        control_symbols.append(final_special_map["bos"])
    if add_eos_as_control and final_special_map.get("eos"):
        control_symbols.append(final_special_map["eos"])
    if add_unk_as_control and final_special_map.get("unk"):
        control_symbols.append(final_special_map["unk"])
    # Note: PAD is often handled differently (via --pad_id) but can be added here too if needed.
    if final_special_map.get("pad"):
         # Decide if pad should also be control or just user defined
         # control_symbols.append(final_special_map["pad"])
         pass # Often better handled by pad_id

    # Construct user_defined_symbols (ensure uniqueness, include non-control special tokens)
    all_user_symbols = set(user_defined_symbols or [])
    for role, token in final_special_map.items():
        if token and token not in control_symbols:
            all_user_symbols.add(token)
    user_defined_symbols_arg = ",".join(sorted(list(all_user_symbols))) if all_user_symbols else ""

    control_symbols_arg = ",".join(control_symbols) if control_symbols else ""

    # Determine num_threads
    effective_num_threads = num_threads if num_threads is not None else os.cpu_count() or 1

    # Build the command string or dictionary for spm.SentencePieceTrainer.train()
    train_args_dict = {
        "input": input_files, # Pass comma-separated string directly
        "model_prefix": str(model_path_prefix),
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "num_threads": effective_num_threads, # Added num_threads
        # Only add arguments if they have a value to avoid passing empty strings
        **({"control_symbols": control_symbols_arg} if control_symbols_arg else {}),
        **({"user_defined_symbols": user_defined_symbols_arg} if user_defined_symbols_arg else {}),
        # --- Default IDs (Match SP conventions) ---
        # Usually BOS=1, EOS=2, UNK=0. PAD often -1 (disabled) or 3.
        "bos_id": 1 if add_bos_as_control else -1, # Disable if not control
        "eos_id": 2 if add_eos_as_control else -1, # Disable if not control
        "unk_id": 0 if add_unk_as_control else -1, # Disable if not control
        "pad_id": -1, # Default: disable padding ID in SP model itself
        # -------------------------------------------
        # Add other relevant training parameters here
        # e.g., "max_sentence_length": max_sentence_length,
    }

    # Convert dict to string format suitable for spm.SentencePieceTrainer.train()
    train_cmd = " ".join([f"--{k}={v}" for k, v in train_args_dict.items()])

    # Corrected logging
    logger.info("Starting SentencePiece training...")
    logger.info(f"Executing command: spm.SentencePieceTrainer.train('{train_cmd}')")

    # --- Execute Training --- #
    try:
        spm.SentencePieceTrainer.train(train_cmd)
        logger.info(f"SentencePiece training complete. Model saved to prefix: {model_path_prefix}")
    except Exception as e:
        logger.error(f"SentencePiece training failed: {e}", exc_info=True)
        # Clean up potentially incomplete model/vocab files
        try:
             if model_file_path.exists(): model_file_path.unlink()
             if vocab_file_path.exists(): vocab_file_path.unlink()
        except Exception as cleanup_e:
             logger.error(f"Error during cleanup after failed SP training: {cleanup_e}")
        raise RuntimeError("SentencePiece training failed") from e

    # --- Verify Output Files --- #
    if not model_file_path.exists() or not vocab_file_path.exists():
        logger.error(f"SentencePiece training finished, but output files not found: {model_file_path}, {vocab_file_path}")
        raise RuntimeError("SentencePiece training completed but output files are missing.")

    # --- Save Metadata --- #
    metadata = {
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "special_tokens_map": final_special_map, # Save the map used
        "training_args": train_args_dict # Save the args dict for reference
    }
    metadata_path = output_dir / f"{model_prefix}.json"
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved training metadata to: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save training metadata to {metadata_path}: {e}")
        # Don't necessarily fail the whole process if metadata save fails, but warn.

    return model_file_path, vocab_file_path # Return paths to model and vocab 