"""
Batch Size Optimizer
===================

This script finds the optimal batch size that maximizes throughput without 
causing out-of-memory errors. It uses a more direct approach that tests
specific promising batch sizes for maximizing throughput.
"""

import argparse
import logging
import torch
import time
import os
import sys
import traceback
from typing import Dict, Any, Tuple, Optional, List
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.logger import setup_logger
from src.model import create_transformer_model
from src.data_handler import load_data
from src.trainer import train_epoch
from src.utils import setup_device, set_seed

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for batch size finder.
    """
    parser = argparse.ArgumentParser(
        description="Find optimal batch size for transformer training."
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
    parser.add_argument("--test_batches", type=int, default=5,
                        help="Number of batches to process during throughput testing.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if CUDA is available.")
    
    return parser.parse_args()

def measure_throughput(
    model, 
    batch_size: int, 
    device: torch.device, 
    data_path: str, 
    sequence_length: int,
    test_batches: int
) -> Tuple[float, bool]:
    """
    Measure model throughput for a given batch size.
    
    Args:
        model: The model to test
        batch_size: Batch size to test
        device: Device to run on
        data_path: Path to data file
        sequence_length: Maximum sequence length
        test_batches: Number of batches to process
        
    Returns:
        Tuple of (tokens_per_second, success_flag)
    """
    # Reset CUDA memory before each test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create optimizer with lightweight settings for testing
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    try:
        # Load small amount of data for test
        train_loader, _, _, _ = load_data(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=0,  # Use 0 to avoid memory overhead from workers
            device_type=device.type
        )
        
        # Create a small subset for quick testing
        limited_batches = []
        for i, batch in enumerate(train_loader):
            limited_batches.append(batch)
            if i >= test_batches:
                break
        
        if len(limited_batches) == 0:
            return 0, False
        
        # Skip the first batch because it includes compilation time
        warmup_batch = limited_batches[0]
        warmup_input, warmup_target = warmup_batch
        warmup_input = warmup_input.to(device)
        warmup_target = warmup_target.to(device)
        
        # Forward and backward to warm up cuDNN and alloc memory
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type):
                outputs = model(warmup_input)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), warmup_target.view(-1))
        loss.backward()
        torch.cuda.synchronize()
        
        # Clear memory from warmup
        del warmup_input, warmup_target, loss, outputs
        optimizer.zero_grad(set_to_none=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Run a second warmup pass to ensure consistent timing
        if len(limited_batches) > 1:
            second_warmup = limited_batches[1]
            second_input, second_target = second_warmup
            second_input = second_input.to(device)
            second_target = second_target.to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(second_input)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), second_target.view(-1))
            loss.backward()
            torch.cuda.synchronize()
            
            # Clear memory again
            del second_input, second_target, loss, outputs
            optimizer.zero_grad(set_to_none=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Benchmark the remaining batches with precise timing
        # Start from the third batch if available, otherwise use what we have
        start_idx = min(2, len(limited_batches) - 1)
        if start_idx >= len(limited_batches):
            return 0, True  # Not enough batches, but no OOM error
        
        # Use CUDA events for more accurate timing
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        
        total_tokens = 0
        
        # Process test batches
        for batch in limited_batches[start_idx:]:
            input_data, target = batch
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Forward and backward pass with AMP
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(input_data)
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target.view(-1))
            loss.backward()
            
            # Count tokens
            total_tokens += input_data.size(0) * input_data.size(1)
            
            # Clear some memory but don't fully delete tensors for consistent timing
            optimizer.zero_grad(set_to_none=True)
        
        # Ensure all operations completed for accurate timing
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed = elapsed_ms / 1000.0  # Convert to seconds
        else:
            elapsed = time.time() - start_time
        
        # Calculate throughput
        tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return tokens_per_second, True
        
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA out of memory" in str(e) or "cudaMalloc failed" in str(e):
            logging.info(f"Batch size {batch_size} caused OOM: {str(e)}")
            return 0, False
        else:
            logging.error(f"Error during benchmark: {str(e)}")
            traceback.print_exc()
            return 0, False

def find_optimal_batch_size(args):
    """
    Find the optimal batch size using a more direct approach to maximize throughput.
    """
    # Set up device
    device, is_cuda, _ = setup_device(args.force_cpu)
    
    if not is_cuda:
        logging.warning("Running on CPU, optimization not applicable")
        return 16  # Default for CPU
    
    # Load a small part of the data to get vocab size
    _, _, char_to_idx, _ = load_data(
        data_path=args.data_path,
        batch_size=1,
        device_type='cpu'  # Load on CPU to avoid memory issues
    )
    
    vocab_size = len(char_to_idx)
    
    # Get GPU details for better recommendations
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    logging.info(f"GPU: {gpu_name} with {total_memory:.2f} GB memory")
    
    # Create model with conservative memory settings first
    model = create_transformer_model(
        vocab_size=vocab_size,
        max_seq_length=args.sequence_length,
        d_model=768,
        n_head=12,
        d_hid=3072,
        n_layers=12,
        dropout=0.2,
        memory_efficient=True  # Always use memory-efficient setting for search
    )
    
    # Move model to device
    model = model.to(device)
    
    # First, find the maximum batch size that doesn't OOM
    # Start with a smaller set of batch sizes based on GPU memory
    if total_memory < 6:  # Less than 6GB (like GTX 1650 Ti)
        logging.info("Low GPU memory detected, testing conservative batch sizes")
        batch_sizes_to_test = [4, 8, 12, 16, 20, 24, 28, 32]
    elif total_memory < 10:  # 6-10 GB
        batch_sizes_to_test = [8, 16, 24, 32, 48, 64]
    else:  # More than 10GB
        batch_sizes_to_test = [16, 32, 48, 64, 96, 128]
    
    # Filter based on min/max batch constraints
    batch_sizes_to_test = [bs for bs in batch_sizes_to_test 
                          if args.min_batch <= bs <= args.max_batch]
    
    logging.info(f"Testing batch sizes: {batch_sizes_to_test}")
    
    # Track results
    results = []
    max_working_batch = 0
    
    # Test each batch size
    for batch_size in batch_sizes_to_test:
        logging.info(f"Testing batch size: {batch_size}")
        
        # Reset CUDA for clean test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure throughput with more accurate settings for real comparison
        throughput, success = measure_throughput(
            model=model, 
            batch_size=batch_size, 
            device=device, 
            data_path=args.data_path,
            sequence_length=args.sequence_length,
            test_batches=max(5, args.test_batches)  # Use at least 5 batches for accuracy
        )
        
        if success:
            max_working_batch = max(max_working_batch, batch_size)
            logging.info(f"Batch size {batch_size} successful: {throughput:.2f} tokens/sec")
            results.append((batch_size, throughput))
        else:
            logging.info(f"Batch size {batch_size} failed (OOM)")
            # Stop testing larger batches if we hit OOM
            break
    
    # If we found working batch sizes, we can try a few more around the best one
    if results:
        # Find the batch size with highest throughput
        best_batch_size, best_throughput = max(results, key=lambda x: x[1])
        
        # Try a few more batch sizes around the best one
        additional_batch_sizes = []
        
        # Try smaller and larger sizes around the best one
        step_size = 2
        for offset in range(-3*step_size, 3*step_size+1, step_size):
            if offset != 0:  # Skip the one we already tested
                candidate = best_batch_size + offset
                if (args.min_batch <= candidate <= args.max_batch and 
                    candidate <= max_working_batch and 
                    candidate not in batch_sizes_to_test):
                    additional_batch_sizes.append(candidate)
        
        # Test additional batch sizes
        if additional_batch_sizes:
            logging.info(f"Testing additional batch sizes around best: {additional_batch_sizes}")
            
            for batch_size in additional_batch_sizes:
                throughput, success = measure_throughput(
                    model=model, 
                    batch_size=batch_size, 
                    device=device, 
                    data_path=args.data_path,
                    sequence_length=args.sequence_length,
                    test_batches=max(5, args.test_batches)
                )
                
                if success:
                    logging.info(f"Batch size {batch_size} successful: {throughput:.2f} tokens/sec")
                    results.append((batch_size, throughput))
                else:
                    logging.info(f"Batch size {batch_size} failed (OOM)")
    
    # Select optimal batch size based on throughput
    if not results:
        logging.warning("No successful batch sizes found, using minimum batch size")
        optimal_batch = args.min_batch
        optimal_throughput = 0
    else:
        # Sort by throughput and choose the best
        results.sort(key=lambda x: x[1], reverse=True)
        optimal_batch, optimal_throughput = results[0]
        
        # Log all results for comparison
        logging.info("\nBatch size throughput results:")
        for batch_size, throughput in results:
            logging.info(f"  Batch size {batch_size}: {throughput:.2f} tokens/sec")
    
    # Report results
    logging.info(f"\nOptimal batch size: {optimal_batch}")
    logging.info(f"Maximum throughput: {optimal_throughput:.2f} tokens/sec")
    
    # Suggest additional optimizations
    if is_cuda:
        logging.info("\nAdditional performance tips:")
        logging.info("  - Try using --use_amp for faster training with minimal precision loss")
        if optimal_throughput < 1000:  # Low throughput threshold
            logging.info("  - Consider reducing model size with --d_model and --n_layers")
        if "memory_efficient=True" in str(model):
            logging.info("  - For higher throughput at the cost of memory, try memory_efficient=False")
    
    return optimal_batch

def main():
    """
    Main function for batch size finder.
    """
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file if args.log_file else "batch_size_optimization.log"
    setup_logger(log_file)
    
    # Set random seed
    set_seed(42)
    
    # Find optimal batch size
    optimal_batch = find_optimal_batch_size(args)
    
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print(f"Optimal batch size: {optimal_batch}")
    print(f"Use this batch size for maximum throughput with sequence length {args.sequence_length}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Error during batch size optimization: {e}")
        sys.exit(1) 