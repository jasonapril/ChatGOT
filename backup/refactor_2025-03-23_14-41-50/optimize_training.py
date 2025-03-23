#!/usr/bin/env python
"""
Transformer Training Optimizer
============================

This script combines all optimization techniques to maximize throughput
of transformer model training, including:

1. Optimal batch size determination
2. CUDA-specific optimizations
3. Gradient accumulation strategies
4. Memory-efficient implementation selection

Usage:
    python optimize_training.py --data_path processed_data/got_char_data.pkl
"""

import argparse
import logging
import os
import sys
import time
import torch
from typing import Dict, Any, Tuple, Optional

# Import local modules
from src.logger import setup_logger
from src.model import create_transformer_model
from src.data_handler import load_data
from src.cuda_optimizations import apply_all_cuda_optimizations
from src.batch_size_finder import find_optimal_batch_size
from src.utils import setup_device, set_seed

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Optimize transformer model training for maximum throughput."
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the processed data file.")
    parser.add_argument("--sequence_length", type=int, default=1024,
                        help="Maximum sequence length (default: 1024)")
    parser.add_argument("--min_batch", type=int, default=1,
                        help="Minimum batch size to try.")
    parser.add_argument("--max_batch", type=int, default=256,
                        help="Maximum batch size to try.")
    parser.add_argument("--test_batches", type=int, default=10,
                        help="Number of batches to process during throughput testing.")
    parser.add_argument("--log_file", type=str, default="optimization_results.log",
                        help="Path to log file.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if CUDA is available.")
    parser.add_argument("--skip_batch_search", action="store_true",
                        help="Skip batch size search and only apply CUDA optimizations.")
    parser.add_argument("--train_after_optimize", action="store_true",
                        help="Start training after optimization with optimal settings.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train if --train_after_optimize is used.")
    
    return parser.parse_args()

def main():
    """
    Main function for optimization.
    """
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logger(args.log_file)
    logging.info("\n" + "="*80)
    logging.info("STARTING OPTIMIZATION FOR MAXIMUM TRAINING THROUGHPUT")
    logging.info("="*80 + "\n")
    
    # Set random seed
    set_seed(42)
    
    # Apply CUDA optimizations
    logging.info("Applying CUDA optimizations...")
    cuda_results = apply_all_cuda_optimizations()
    
    # Set up device
    device, is_cuda, num_gpus = setup_device(args.force_cpu)
    
    if not is_cuda:
        logging.warning("CUDA not available, optimization will be limited")
    else:
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Find optimal batch size if not skipped
    optimal_batch_size = None
    if not args.skip_batch_search:
        logging.info("\n" + "="*80)
        logging.info("FINDING OPTIMAL BATCH SIZE")
        logging.info("="*80 + "\n")
        
        optimal_batch_size = find_optimal_batch_size(args)
        
        logging.info(f"\nOPTIMAL BATCH SIZE: {optimal_batch_size}")
    
    # Log optimization results
    logging.info("\n" + "="*80)
    logging.info("OPTIMIZATION RESULTS")
    logging.info("="*80 + "\n")
    
    if optimal_batch_size:
        logging.info(f"Optimal batch size: {optimal_batch_size}")
    
    logging.info("\nCUDA Optimizations:")
    for name, success in cuda_results.items():
        logging.info(f"- {name}: {'Enabled' if success else 'Disabled'}")
    
    # Provide training command with optimal settings
    cmd = f"python -m src.train --data_path {args.data_path} --sequence_length {args.sequence_length}"
    
    if optimal_batch_size:
        cmd += f" --batch_size {optimal_batch_size}"
    
    cmd += " --use_amp --force_aggressive_memory"
    
    logging.info("\nRecommended training command:")
    logging.info(cmd)
    
    # Start training if requested
    if args.train_after_optimize:
        logging.info("\n" + "="*80)
        logging.info("STARTING TRAINING WITH OPTIMAL SETTINGS")
        logging.info("="*80 + "\n")
        
        # Import train module
        try:
            from src.train import main as train_main
            sys.argv = cmd.split()[1:]  # Skip "python"
            train_main()
        except Exception as e:
            logging.exception(f"Error during training: {e}")
    
    # Print final results to console
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    if optimal_batch_size:
        print(f"\nOptimal batch size: {optimal_batch_size}")
    
    print("\nCUDA Optimizations:")
    for name, success in cuda_results.items():
        print(f"- {name}: {'Enabled' if success else 'Disabled'}")
    
    print("\nRecommended training command:")
    print(cmd)
    print("\nSee log file for details:", args.log_file)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Error during optimization: {e}")
        sys.exit(1) 