#!/usr/bin/env python
"""
Quick Activation Checkpointing Test
==================================

This script provides a quick test of activation checkpointing's memory efficiency,
using a minimal model configuration to finish in minutes rather than hours.
"""

import os
import sys
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from src.model import create_transformer_model, TransformerModel
from src.utils import set_seed, setup_device

def parse_args():
    parser = argparse.ArgumentParser(description="Quick test of activation checkpointing memory efficiency")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length for testing")
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["tiny", "small", "medium"],
                        help="Model size to use for testing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def get_model_config(size):
    """Get model configuration based on size."""
    if size == "tiny":
        return {
            "d_model": 128,
            "n_head": 2,
            "d_hid": 512,
            "n_layers": 2
        }
    elif size == "small":
        return {
            "d_model": 256,
            "n_head": 4,
            "d_hid": 1024,
            "n_layers": 4
        }
    else:  # medium
        return {
            "d_model": 384,
            "n_head": 6,
            "d_hid": 1536,
            "n_layers": 6
        }

def measure_memory_usage(model, batch_size, seq_length, device, use_checkpointing):
    """Measure peak memory usage during a forward and backward pass."""
    # Create random input data
    input_ids = torch.randint(0, 100, (batch_size, seq_length), device=device)
    target_ids = torch.randint(0, 100, (batch_size, seq_length), device=device)
    
    # Clear cache and reset peak memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Set model to training mode
    model.train()
    
    # Start timer
    start_time = time.time()
    
    # Forward pass
    outputs = model(input_ids)
    
    # Reshape outputs and targets for loss calculation
    outputs = outputs.view(-1, outputs.size(-1))
    targets = target_ids.view(-1)
    
    # Calculate loss
    loss = criterion(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # End timer
    end_time = time.time()
    
    # Get peak memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # in MB
    else:
        peak_memory = 0
    
    return {
        'peak_memory_mb': peak_memory,
        'time_sec': end_time - start_time,
        'use_checkpointing': use_checkpointing
    }

def main():
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up device
    device, is_cuda, _ = setup_device(False)
    
    if not is_cuda:
        logging.warning("CUDA is not available. Activation checkpointing benchmark is most useful on GPU.")
    
    # Get model config
    config = get_model_config(args.model_size)
    
    logging.info(f"Starting quick activation checkpointing test with {args.model_size} model")
    logging.info(f"Model config: d_model={config['d_model']}, n_layers={config['n_layers']}")
    logging.info(f"Batch size: {args.batch_size}, Sequence length: {args.seq_length}")
    
    # Test increasing batch sizes to see where the model breaks without checkpointing
    logging.info("\nTesting maximum batch size without activation checkpointing...")
    batch_sizes = []
    standard_max_batch = 0
    
    # Start with the specified batch size and increase until OOM
    test_batch_size = args.batch_size
    while test_batch_size <= 256:  # Cap at 256 to avoid excessive testing
        try:
            logging.info(f"Testing batch size: {test_batch_size}")
            
            # Create model without activation checkpointing
            model_standard = create_transformer_model(
                vocab_size=100,  # Arbitrary vocabulary size for testing
                d_model=config['d_model'],
                n_head=config['n_head'],
                d_hid=config['d_hid'],
                n_layers=config['n_layers'],
                dropout=0.1,
                use_activation_checkpointing=False
            )
            model_standard = model_standard.to(device)
            
            # Measure memory usage
            results = measure_memory_usage(
                model=model_standard,
                batch_size=test_batch_size,
                seq_length=args.seq_length,
                device=device,
                use_checkpointing=False
            )
            
            logging.info(f"  Peak memory: {results['peak_memory_mb']:.2f} MB")
            logging.info(f"  Time: {results['time_sec']:.4f} sec")
            
            batch_sizes.append(test_batch_size)
            standard_max_batch = test_batch_size
            
            # Clean up
            del model_standard
            torch.cuda.empty_cache()
            
            # Increase batch size for next iteration (by 50%)
            test_batch_size = int(test_batch_size * 1.5)
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "cuda runtime error" in str(e).lower():
                logging.info(f"  Out of memory at batch size {test_batch_size}")
                break
            else:
                logging.exception("Unexpected error during testing")
                break
    
    # Now test with activation checkpointing enabled
    logging.info("\nTesting maximum batch size WITH activation checkpointing...")
    checkpointing_max_batch = 0
    
    # Start with the last successful batch size for standard model
    test_batch_size = standard_max_batch
    while test_batch_size <= 512:  # Cap higher for checkpointing
        try:
            logging.info(f"Testing batch size: {test_batch_size}")
            
            # Create model with activation checkpointing
            model_checkpointed = create_transformer_model(
                vocab_size=100,  # Arbitrary vocabulary size for testing
                d_model=config['d_model'],
                n_head=config['n_head'],
                d_hid=config['d_hid'],
                n_layers=config['n_layers'],
                dropout=0.1,
                use_activation_checkpointing=True
            )
            model_checkpointed = model_checkpointed.to(device)
            
            # Measure memory usage
            results = measure_memory_usage(
                model=model_checkpointed,
                batch_size=test_batch_size,
                seq_length=args.seq_length,
                device=device,
                use_checkpointing=True
            )
            
            logging.info(f"  Peak memory: {results['peak_memory_mb']:.2f} MB")
            logging.info(f"  Time: {results['time_sec']:.4f} sec")
            
            checkpointing_max_batch = test_batch_size
            
            # Clean up
            del model_checkpointed
            torch.cuda.empty_cache()
            
            # Increase batch size for next iteration (by 50%)
            test_batch_size = int(test_batch_size * 1.5)
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "cuda runtime error" in str(e).lower():
                logging.info(f"  Out of memory at batch size {test_batch_size}")
                break
            else:
                logging.exception("Unexpected error during testing")
                break
    
    # Print results summary
    logging.info("\n" + "="*50)
    logging.info("QUICK ACTIVATION CHECKPOINTING TEST RESULTS")
    logging.info("="*50)
    
    batch_size_improvement = round((checkpointing_max_batch / standard_max_batch - 1) * 100)
    
    logging.info(f"\nModel size: {args.model_size}")
    logging.info(f"Configuration: d_model={config['d_model']}, n_layers={config['n_layers']}, n_head={config['n_head']}")
    logging.info(f"Maximum batch size WITHOUT activation checkpointing: {standard_max_batch}")
    logging.info(f"Maximum batch size WITH activation checkpointing: {checkpointing_max_batch}")
    logging.info(f"Batch size improvement: {batch_size_improvement}%")
    
    logging.info("\nRECOMMENDATIONS:")
    if batch_size_improvement > 30:
        logging.info("- Activation checkpointing provides significant memory savings.")
        logging.info("- Highly recommended for training with larger batch sizes.")
    elif batch_size_improvement > 10:
        logging.info("- Activation checkpointing provides moderate memory savings.")
        logging.info("- Recommended for memory-constrained scenarios.")
    else:
        logging.info("- Activation checkpointing provides limited memory benefits in this scenario.")
        logging.info("- Consider using other memory optimization techniques or adjusting model size.")

if __name__ == "__main__":
    main() 