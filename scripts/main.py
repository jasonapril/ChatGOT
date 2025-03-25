#!/usr/bin/env python
"""
Pipeline CLI Entry Point
=======================

This is the main command-line interface for running the complete
text generation pipeline. It parses arguments and configures 
the pipeline before execution.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

# Add the parent directory to the path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.core.pipeline_core import Pipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ChatGoT: Text Generation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pipeline configuration
    parser.add_argument('--pipeline-dir', type=str, default='./pipeline_runs/run_{date}',
                        help='Directory to store pipeline artifacts')
    parser.add_argument('--force-restart', action='store_true',
                        help='Force restart the pipeline from the beginning')
    parser.add_argument('--resume', action='store_true',
                        help='Resume the pipeline from the last completed stage')
    parser.add_argument('--stage', type=str, choices=['process', 'optimize', 'train', 'generate'],
                        help='Start the pipeline from a specific stage')
    parser.add_argument('--skip-process', action='store_true',
                        help='Skip the data processing stage')
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Skip the optimization stage and use provided hyperparameters')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    
    # Data processing stage
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input text file for processing')
    parser.add_argument('--vocab-size', type=int, default=5000,
                        help='Vocabulary size for tokenization')
    parser.add_argument('--seq-length', type=int, default=128,
                        help='Sequence length for training samples')
    
    # Optimization stage
    parser.add_argument('--auto-optimize', action='store_true',
                        help='Automatically optimize hyperparameters')
    parser.add_argument('--optimization-time-budget', type=int, default=600,
                        help='Time budget for optimization in seconds')
    
    # Model parameters (used if not auto-optimizing)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size of the model')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in the model')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    
    # Generation parameters
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of text samples to generate')
    parser.add_argument('--max-length', type=int, default=500,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter (0 to disable)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p sampling parameter (1.0 to disable)')
    parser.add_argument('--seed-text', type=str, default='',
                        help='Seed text for generation')
    
    args = parser.parse_args()
    
    # Process the date template in the pipeline directory
    if '{date}' in args.pipeline_dir:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.pipeline_dir = args.pipeline_dir.replace('{date}', date_str)
    
    return args

def main():
    """Main entry point for the pipeline."""
    args = parse_args()
    
    # Create the pipeline
    pipeline = Pipeline(args)
    
    # Run the pipeline
    start_time = time.time()
    try:
        pipeline.run()
        logging.info(f"Pipeline completed successfully in {time.time() - start_time:.2f} seconds")
        return 0
    except KeyboardInterrupt:
        logging.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 