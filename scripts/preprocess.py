import os
import json
import argparse
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vocabulary(input_path: str, output_path: str) -> None:
    """
    Reads a text file, calculates the character vocabulary, and saves it.

    Args:
        input_path: Path to the raw input text file.
        output_path: Path to save the output JSON file (containing vocab info).
    """
    logger.info(f"Starting vocabulary creation from: {input_path}")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Read {len(text):,} characters from {input_path}")
    except Exception as e:
        logger.error(f"Failed to read input file {input_path}: {e}", exc_info=True)
        raise

    if not text:
        logger.warning("Input file is empty. Cannot create vocabulary.")
        # Save an empty vocab file?
        vocab_data = {
            'char_to_idx': {},
            'idx_to_char': {},
            'vocab_size': 0
        }
    else:
        # Calculate character counts and sorted vocabulary
        char_counts = Counter(text)
        chars = sorted(char_counts.keys())
        vocab_size = len(chars)
        logger.info(f"Found {vocab_size} unique characters.")

        # Create mappings
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}

        vocab_data = {
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size': vocab_size
        }

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Save vocabulary data to JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Vocabulary saved successfully to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save vocabulary to {output_path}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create character vocabulary from a text file.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input text file."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the output vocabulary JSON file."
    )
    args = parser.parse_args()

    create_vocabulary(args.input_path, args.output_path) 