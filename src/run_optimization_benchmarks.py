#!/usr/bin/env python
"""
Optimization Benchmarks
======================

This script runs comprehensive benchmarks for different optimization techniques,
allowing for fair comparisons between them.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_transformer_model
from src.data_handler import load_data
from src.utils import set_seed, setup_device
from src.benchmark_logger import get_benchmark_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive optimization benchmarks")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the processed data file")
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["tiny", "small", "medium", "full-gpt2"],
                        help="Model size to use for benchmarks")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of steps for each benchmark")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all optimization combinations")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline benchmark (no optimizations)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Run benchmark with 8-bit optimizer")
    parser.add_argument("--use_activation_checkpointing", action="store_true",
                        help="Run benchmark with activation checkpointing")
    parser.add_argument("--use_combined", action="store_true",
                        help="Run benchmark with both 8-bit optimizer and activation checkpointing")
    
    return parser.parse_args()

def get_model_config(size):
    """Get model configuration parameters based on size."""
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
    elif size == "medium":
        return {
            "d_model": 384,
            "n_head": 6,
            "d_hid": 1536,
            "n_layers": 6
        }
    else:  # full-gpt2
        return {
            "d_model": 768,
            "n_head": 12,
            "d_hid": 3072,
            "n_layers": 12
        }

def train_loop(model, dataloader, optimizer, device, num_steps=50, get_memory=False):
    """Training loop that measures throughput and optionally memory usage."""
    model.train()
    total_tokens = 0
    start_time = time.time()
    
    # Reset peak memory stats if measuring memory
    if get_memory and device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_steps:
            break
            
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Shift targets for next-token prediction
        shifted_outputs = outputs[:, :-1, :].contiguous()
        shifted_targets = targets[:, 1:].contiguous()
        
        # Reshape for loss calculation
        B, T, C = shifted_outputs.shape
        shifted_outputs = shifted_outputs.view(-1, C)
        shifted_targets = shifted_targets.view(-1)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(shifted_outputs, shifted_targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track tokens processed
        total_tokens += inputs.numel()
        
        # Print progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            print(f"Step {i+1}/{num_steps} | Loss: {loss.item():.4f} | Tokens/sec: {tokens_per_sec:.2f}")
    
    # Calculate final throughput
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    
    # Get peak memory usage if requested
    peak_memory = None
    if get_memory and device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
    
    return tokens_per_sec, peak_memory

def run_benchmark(model_config, use_8bit=False, use_activation_checkpointing=False, 
                 args=None, device=None, train_loader=None):
    """Run a benchmark with specified optimizations."""
    
    # Create benchmark name
    opt_name = "baseline"
    if use_8bit and use_activation_checkpointing:
        opt_name = "8bit+checkpointing"
    elif use_8bit:
        opt_name = "8bit"
    elif use_activation_checkpointing:
        opt_name = "checkpointing"
        
    benchmark_name = f"{args.model_size}_{opt_name}_b{args.batch_size}"
    
    # Log benchmark start
    logging.info(f"Running benchmark: {benchmark_name}")
    logging.info(f"Model config: d_model={model_config['d_model']}, n_layers={model_config['n_layers']}")
    logging.info(f"Optimizations: 8bit={use_8bit}, checkpointing={use_activation_checkpointing}")
    
    # Create model with vocab size from model_config
    model = create_transformer_model(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_head=model_config["n_head"],
        d_hid=model_config["d_hid"],
        n_layers=model_config["n_layers"],
        dropout=0.1,
        use_activation_checkpointing=use_activation_checkpointing
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    if use_8bit and device.type == 'cuda':
        try:
            import bitsandbytes as bnb
            logging.info("Creating 8-bit AdamW optimizer")
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        except ImportError:
            logging.warning("bitsandbytes not available. Falling back to standard optimizer.")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
    else:
        logging.info("Creating standard AdamW optimizer")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
    
    # Warm up
    logging.info("Warming up...")
    _ = train_loop(model, train_loader, optimizer, device, num_steps=5)
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Run benchmark
    logging.info(f"Running benchmark for {args.num_steps} steps...")
    tokens_per_sec, peak_memory = train_loop(
        model, train_loader, optimizer, device, 
        num_steps=args.num_steps, get_memory=True
    )
    
    # Log result with benchmark logger
    benchmark_logger = get_benchmark_logger()
    benchmark_logger.log_benchmark(
        name=benchmark_name,
        throughput=tokens_per_sec,
        batch_size=args.batch_size,
        model_config=model_config,
        optimizations={
            "8bit_optimizer": use_8bit,
            "activation_checkpointing": use_activation_checkpointing
        },
        memory_usage_mb=peak_memory
    )
    
    # Clean up
    del model
    del optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return tokens_per_sec, peak_memory

def main():
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("Starting optimization benchmarks")
    logging.info(f"Model size: {args.model_size}, Batch size: {args.batch_size}, Steps: {args.num_steps}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up device
    device, is_cuda, _ = setup_device(False)
    
    if not is_cuda:
        logging.warning("CUDA is not available. Benchmarks will be slower and memory usage won't be measured.")
    
    # Get model configuration
    model_config = get_model_config(args.model_size)
    
    # Load data
    logging.info(f"Loading data from {args.data_path}")
    train_loader, _, char_to_idx, _ = load_data(
        args.data_path,
        batch_size=args.batch_size,
        device_type=device.type,
        num_workers=4
    )
    
    # Get vocabulary size from character mapping
    vocab_size = len(char_to_idx)
    logging.info(f"Vocabulary size: {vocab_size}")
    
    # Add vocab_size to model_config for easier reference
    model_config["vocab_size"] = vocab_size
    
    # Determine which benchmarks to run
    benchmarks_to_run = []
    if args.run_all or args.baseline:
        benchmarks_to_run.append({"use_8bit": False, "use_activation_checkpointing": False})
    if args.run_all or args.use_8bit:
        benchmarks_to_run.append({"use_8bit": True, "use_activation_checkpointing": False})
    if args.run_all or args.use_activation_checkpointing:
        benchmarks_to_run.append({"use_8bit": False, "use_activation_checkpointing": True})
    if args.run_all or args.use_combined:
        benchmarks_to_run.append({"use_8bit": True, "use_activation_checkpointing": True})
    
    # If no benchmarks specified, default to running all
    if not benchmarks_to_run:
        logging.info("No specific benchmarks selected. Running all combinations.")
        benchmarks_to_run = [
            {"use_8bit": False, "use_activation_checkpointing": False},
            {"use_8bit": True, "use_activation_checkpointing": False},
            {"use_8bit": False, "use_activation_checkpointing": True},
            {"use_8bit": True, "use_activation_checkpointing": True}
        ]
    
    # Run benchmarks
    logging.info(f"Running {len(benchmarks_to_run)} benchmarks...")
    
    for i, benchmark in enumerate(benchmarks_to_run):
        logging.info(f"Benchmark {i+1}/{len(benchmarks_to_run)}")
        try:
            run_benchmark(
                model_config=model_config,
                use_8bit=benchmark["use_8bit"],
                use_activation_checkpointing=benchmark["use_activation_checkpointing"],
                args=args,
                device=device,
                train_loader=train_loader
            )
        except Exception as e:
            logging.error(f"Error running benchmark: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Compare results
    benchmark_logger = get_benchmark_logger()
    benchmark_logger.compare_benchmarks()

if __name__ == "__main__":
    main() 