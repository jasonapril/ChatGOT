#!/usr/bin/env python
"""
Optimizer Test Script
====================
Tests the 8-bit optimizer functionality to measure throughput improvements.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_transformer_model
from src.data_handler import load_data
from src.utils import set_seed, setup_device

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 8-bit optimizer performance")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the processed data file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train for")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of steps to measure throughput")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit optimizer")
    parser.add_argument("--use_activation_checkpointing", action="store_true",
                        help="Use activation checkpointing to reduce memory usage")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def train_loop(model, dataloader, optimizer, device, num_steps=100):
    model.train()
    total_tokens = 0
    start_time = time.time()
    
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
    
    return tokens_per_sec

def main():
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Log optimizer configuration
    if args.use_8bit:
        logging.info("Testing with 8-bit optimizer")
    else:
        logging.info("Testing with standard AdamW optimizer")
        
    if args.use_activation_checkpointing:
        logging.info("Using activation checkpointing to reduce memory usage")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up device
    device, is_cuda, _ = setup_device(False)
    
    # Load data
    logging.info(f"Loading data from {args.data_path}")
    train_loader, _, char_to_idx, _ = load_data(
        args.data_path,
        batch_size=args.batch_size,
        device_type=device.type,
        num_workers=4
    )
    
    # Create model
    vocab_size = len(char_to_idx)
    seq_length = next(iter(train_loader))[0].size(1)
    
    logging.info(f"Creating model with vocabulary size {vocab_size}, sequence length {seq_length}")
    model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=768,  # standard GPT-2 Small dimension
        n_head=12,     # standard GPT-2 Small heads
        d_hid=3072,   # standard GPT-2 Small hidden dim
        n_layers=12,  # standard GPT-2 Small layers
        dropout=0.1,
        use_activation_checkpointing=args.use_activation_checkpointing
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    if args.use_8bit and is_cuda:
        try:
            import bitsandbytes as bnb
            print("Creating 8-bit AdamW optimizer")
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
            print("8-bit optimizer successfully created")
        except ImportError:
            print("bitsandbytes not available. Falling back to standard optimizer.")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=5e-5,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
    else:
        print("Creating standard AdamW optimizer")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
    
    # Run training loop
    print(f"Starting training with {'8-bit' if args.use_8bit else 'standard'} optimizer")
    print(f"Batch size: {args.batch_size}")
    
    # Warm up
    print("Warming up...")
    _ = train_loop(model, train_loader, optimizer, device, num_steps=10)
    
    # Clear GPU cache
    if is_cuda:
        torch.cuda.empty_cache()
    
    # Run benchmark
    print(f"Running benchmark for {args.num_steps} steps...")
    tokens_per_sec = train_loop(model, train_loader, optimizer, device, num_steps=args.num_steps)
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Optimizer: {'8-bit AdamW' if args.use_8bit else 'AdamW'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print("="*50)

if __name__ == "__main__":
    main() 