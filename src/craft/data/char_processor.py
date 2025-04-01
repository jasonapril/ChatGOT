"""
Processor for character-level language modeling data.
"""

import os
import pickle
import logging
from typing import List, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def process_char_level_data(input_path: str, output_dir: str, splits: Tuple[float, float, float] = (0.9, 0.05, 0.05)) -> Dict[str, str]:
    """
    Processes a raw text file for character-level language modeling.

    Reads the input file, creates a character vocabulary, converts the text to
    token IDs, splits into train/val/test sets, and saves each split along with
    vocabulary metadata to pickle files in the output directory.

    Args:
        input_path: Path to the raw input text file.
        output_dir: Directory to save the processed .pkl files (train.pkl, val.pkl, test.pkl).
        splits: A tuple representing the fraction for (train, validation, test) splits.
                Must sum to 1.0.

    Returns:
        A dictionary mapping split names ('train', 'val', 'test') to their output file paths.

    Raises:
        ValueError: If splits do not sum to 1.0 or input file not found.
        IOError: If reading the input file or writing output files fails.
    """
    logger.info(f"Starting character-level processing for: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split ratios (train/val/test): {splits}")

    if not os.path.exists(input_path):
        raise ValueError(f"Input file not found: {input_path}")

    if not np.isclose(sum(splits), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, but got {splits} (sum={sum(splits)})" )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read the raw data
        logger.info("Reading raw data...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = f.read()
        logger.info(f"Read {len(data):,} characters.")

        # Build vocabulary
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        logger.info(f"Vocabulary size: {vocab_size}")
        # logger.debug(f"Vocabulary: {''.join(chars)}")

        # Convert text to token IDs
        logger.info("Tokenizing data...")
        token_ids = np.array([char_to_idx[ch] for ch in data], dtype=np.uint16) # Use uint16 if vocab_size < 65536
        logger.info(f"Generated {len(token_ids):,} tokens.")

        # Calculate split points
        n = len(token_ids)
        train_end = int(splits[0] * n)
        val_end = train_end + int(splits[1] * n)

        train_ids = token_ids[:train_end]
        val_ids = token_ids[train_end:val_end]
        test_ids = token_ids[val_end:]
        logger.info(f"Split sizes: Train={len(train_ids):,}, Val={len(val_ids):,}, Test={len(test_ids):,}")

        # Prepare metadata
        meta = {
            'vocab_size': vocab_size,
            'idx_to_char': idx_to_char,
            'char_to_idx': char_to_idx,
            'chars': chars, # Include the actual chars list
        }

        # Save splits to pickle files
        output_paths = {}
        for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            output_filename = f"{split_name}.pkl"
            output_filepath = os.path.join(output_dir, output_filename)
            logger.info(f"Saving {split_name} split to {output_filepath}...")

            # Combine token IDs and metadata for saving
            save_data = {
                'token_ids': split_ids,
                **meta # Embed metadata in each split file
            }

            with open(output_filepath, 'wb') as f:
                pickle.dump(save_data, f)
            output_paths[split_name] = output_filepath
            logger.info(f"Saved {split_name} split ({len(split_ids):,} tokens).")

        logger.info("Character-level processing completed successfully.")
        return output_paths

    except Exception as e:
        logger.error(f"Error during character-level processing: {e}", exc_info=True)
        raise IOError(f"Processing failed for {input_path}") from e 