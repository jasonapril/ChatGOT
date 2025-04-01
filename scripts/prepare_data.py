import argparse
import logging
import os
import sys

# Add project root to path to allow importing 'craft'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the processor function
try:
    # from craft.data.processors import process_char_level_data # Old location
    from craft.data.char_processor import process_char_level_data # New location
except ImportError as e:
    print(f"Error: Could not import processing functions. {e}")
    print("Ensure the 'craft' package is installed correctly (e.g., pip install -e .)")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare raw data for use with the Craft framework.")
    parser.add_argument('--input-path', type=str, required=True, help="Path to the raw input data file or directory.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save the processed data.")
    parser.add_argument('--type', type=str, default='char', choices=['char', 'subword'], help="Type of processing to perform (default: char).")
    # parser.add_argument('--config', type=str, help="Path to a data processing configuration file (optional).") # Keep for future use?

    # Arguments specific to char-level processing
    parser.add_argument('--split-ratios', type=float, nargs=3, default=[0.9, 0.05, 0.05], help="Train/Val/Test split ratios (e.g., 0.9 0.05 0.05).")

    # Arguments specific to subword processing (add later)
    # parser.add_argument('--vocab-size', type=int, default=10000, help="Vocabulary size for subword tokenizers.")
    # parser.add_argument('--tokenizer-type', type=str, default='bpe', choices=['bpe', 'wordpiece'], help="Subword tokenizer type.")

    return parser.parse_args()

def main():
    """Main data preparation script execution."""
    args = parse_args()
    logger.info(f"Starting data preparation script:")
    logger.info(f"  Input path: {args.input_path}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Processing type: {args.type}")

    try:
        if args.type == 'char':
            logger.info("Performing character-level processing...")
            logger.info(f"  Split ratios: {args.split_ratios}")
            # Validate splits sum to 1
            if not abs(sum(args.split_ratios) - 1.0) < 1e-6:
                 parser.error(f"Split ratios must sum to 1.0. Got: {args.split_ratios}")

            process_char_level_data(
                input_path=args.input_path,
                output_dir=args.output_dir,
                splits=tuple(args.split_ratios)
            )
            logger.info("Character-level processing finished successfully.")

        elif args.type == 'subword':
            logger.warning("Subword processing not yet implemented in this script.")
            # Add logic here later, potentially calling a different function
            # from craft.data.processors or using the config argument.
            pass

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        sys.exit(1)
    except IOError as ioe:
        logger.error(f"File I/O error: {ioe}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data preparation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 