#!/usr/bin/env python
"""
Benchmark Activation Checkpointing
=================================

This script benchmarks the performance impact of using activation checkpointing,
measuring both memory usage and throughput.
"""

import os
import sys
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import local modules
from src.model import create_transformer_model, TransformerModel
from src.data_handler import load_data
from src.trainer import train_epoch
from src.utils import set_seed, setup_device

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark activation checkpointing")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the processed data file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of steps to measure throughput")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def measure_memory_and_throughput(model, train_loader, device, optimizer, num_steps=50, use_checkpointing=False):
    """
    Measure memory usage and training throughput.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training
        device: Device to train on
        optimizer: Optimizer to use
        num_steps: Number of steps to measure
        use_checkpointing: Whether activation checkpointing is enabled
        
    Returns:
        Dict containing throughput and peak memory usage
    """
    # Clear cache before measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Create a simple scheduler that does nothing
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    
    # Start timer
    start_time = time.time()
    
    # Train for specified number of steps
    avg_loss, tokens_per_sec = train_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        device=device,
        epoch=0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Get peak memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # in MB
    else:
        peak_memory = 0
    
    return {
        'avg_loss': avg_loss,
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_mb': peak_memory,
        'train_time_sec': train_time,
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
    
    # Load data
    logging.info(f"Loading data from {args.data_path}")
    train_loader, _, char_to_idx, _ = load_data(
        args.data_path,
        batch_size=args.batch_size,
        device_type=device.type,
        num_workers=4
    )
    
    # Limit the number of batches for benchmarking
    def limit_batches(loader, num_steps):
        for i, batch in enumerate(loader):
            if i >= num_steps:
                break
            yield batch
    
    limited_loader = list(limit_batches(train_loader, args.num_steps))
    limited_train_loader = torch.utils.data.DataLoader(
        [(x, y) for x, y in limited_loader],
        batch_size=1,
        collate_fn=lambda b: (torch.cat([x[0] for x in b]), torch.cat([x[1] for x in b]))
    )
    
    # Get vocabulary size
    vocab_size = len(char_to_idx)
    
    # Run benchmark without activation checkpointing
    logging.info("=== Running benchmark WITHOUT activation checkpointing ===")
    model_standard = create_transformer_model(
        vocab_size=vocab_size,
        d_model=768,
        n_head=12,
        d_hid=3072,
        n_layers=12,
        dropout=0.1,
        use_activation_checkpointing=False
    )
    model_standard = model_standard.to(device)
    
    optimizer_standard = optim.AdamW(
        model_standard.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    results_standard = measure_memory_and_throughput(
        model=model_standard,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer_standard,
        num_steps=args.num_steps,
        use_checkpointing=False
    )
    
    # Clear GPU memory
    del model_standard
    del optimizer_standard
    torch.cuda.empty_cache()
    
    # Run benchmark with activation checkpointing
    logging.info("=== Running benchmark WITH activation checkpointing ===")
    model_checkpointed = create_transformer_model(
        vocab_size=vocab_size,
        d_model=768,
        n_head=12,
        d_hid=3072,
        n_layers=12,
        dropout=0.1,
        use_activation_checkpointing=True
    )
    model_checkpointed = model_checkpointed.to(device)
    
    optimizer_checkpointed = optim.AdamW(
        model_checkpointed.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    results_checkpointed = measure_memory_and_throughput(
        model=model_checkpointed,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer_checkpointed,
        num_steps=args.num_steps,
        use_checkpointing=True
    )
    
    # Print results
    logging.info("\n" + "="*50)
    logging.info("BENCHMARK RESULTS")
    logging.info("="*50)
    
    logging.info("\nWithout Activation Checkpointing:")
    logging.info(f"- Throughput: {results_standard['tokens_per_sec']:.2f} tokens/sec")
    logging.info(f"- Peak Memory: {results_standard['peak_memory_mb']:.2f} MB")
    logging.info(f"- Avg Loss: {results_standard['avg_loss']:.4f}")
    logging.info(f"- Training Time: {results_standard['train_time_sec']:.2f} sec")
    
    logging.info("\nWith Activation Checkpointing:")
    logging.info(f"- Throughput: {results_checkpointed['tokens_per_sec']:.2f} tokens/sec")
    logging.info(f"- Peak Memory: {results_checkpointed['peak_memory_mb']:.2f} MB")
    logging.info(f"- Avg Loss: {results_checkpointed['avg_loss']:.4f}")
    logging.info(f"- Training Time: {results_checkpointed['train_time_sec']:.2f} sec")
    
    # Calculate the impact
    memory_reduction = (results_standard['peak_memory_mb'] - results_checkpointed['peak_memory_mb']) / results_standard['peak_memory_mb'] * 100
    throughput_change = (results_checkpointed['tokens_per_sec'] - results_standard['tokens_per_sec']) / results_standard['tokens_per_sec'] * 100
    
    logging.info("\nImpact:")
    logging.info(f"- Memory Reduction: {memory_reduction:.2f}%")
    logging.info(f"- Throughput Change: {throughput_change:.2f}%")
    
    # Provide recommendations
    logging.info("\nRECOMMENDATIONS:")
    if memory_reduction > 20:
        logging.info("- Activation checkpointing provides significant memory savings.")
        if throughput_change < -10:
            logging.info("- However, there is a noticeable throughput penalty.")
            logging.info("- Recommended for memory-constrained scenarios where you need larger batch sizes.")
        else:
            logging.info("- With minimal throughput impact.")
            logging.info("- Highly recommended for all training scenarios.")
    else:
        logging.info("- Activation checkpointing provides limited memory benefits in this scenario.")
        logging.info("- Consider using other memory optimization techniques or adjusting model size.")

if __name__ == "__main__":
    main() 