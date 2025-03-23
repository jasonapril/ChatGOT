#!/usr/bin/env python3
"""
Data Processing Script
=====================

This script processes raw Game of Thrones script data to prepare it for training.
It uses the data_processor module from the src package.

Usage:
    python process_data.py
"""

import argparse
import logging
import sys
import os
from src.logger import setup_logger, log_section_header, force_flush_logs
from src.data_processor import process_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process Game of Thrones script data for training")
    
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing raw text files")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                        help="Directory to save processed data")
    parser.add_argument("--file_pattern", type=str, default="*.txt",
                        help="Glob pattern for files to include")
    parser.add_argument("--min_char_freq", type=int, default=5,
                        help="Minimum character frequency for vocabulary")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Length of training sequences")
    parser.add_argument("--stride", type=int, default=None,
                        help="Step size between sequences (default: seq_length/2)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of data to use for training (vs. validation)")
    parser.add_argument("--output_filename", type=str, default="got_char_data.pkl",
                        help="Filename for the processed data")
    parser.add_argument("--log_file", type=str, default="data_processing.log",
                        help="Path to log file")
                        
    return parser.parse_args()

def main():
    """Main function for processing data."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logger(args.log_file)
    log_section_header("DATA PROCESSING")
    
    # Log arguments
    logging.info("Processing data with the following parameters:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    try:
        # Process data
        output_path = process_data(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            min_char_freq=args.min_char_freq,
            seq_length=args.seq_length,
            stride=args.stride,
            train_ratio=args.train_ratio,
            output_filename=args.output_filename
        )
        
        # Log success
        logging.info(f"Data processing completed successfully")
        print(f"\nData processing completed successfully!")
        print(f"Processed data saved to: {output_path}")
        force_flush_logs()
        
        return 0
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"\nError processing data: {str(e)}")
        force_flush_logs()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 