#!/usr/bin/env python3
"""
Diagnostic script for the ChatGoT model.
This script checks CUDA availability and reports model parameters.

Usage:
    python diagnostic.py
"""

import os
import sys
import torch
import logging
from src.utils import diagnose_cuda_status, setup_device, set_seed
from src.model import create_transformer_model

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Run CUDA diagnostic
    logging.info("Running CUDA diagnostic...")
    diagnose_cuda_status()
    
    # Set up device
    device, is_cuda, num_gpus = setup_device(force_cpu=False)
    logging.info(f"Using device: {device}" + (f" ({num_gpus} GPUs available)" if is_cuda else ""))
    
    # Create model and report parameters
    logging.info("Creating test model to check parameter count...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Standard character-level vocabulary size for English text + special tokens
    vocab_size = 128
    
    # Create model with our target configuration
    model = create_transformer_model(
        vocab_size=vocab_size,
        max_seq_length=1024,   # Standard GPT-2 context window
        d_model=768,      # GPT-2 Small
        n_head=12,         # GPT-2 Small
        d_hid=3072,       # GPT-2 Small
        n_layers=12,      # Standard GPT-2 Small
        dropout=0.1,      # GPT-2 Standard dropout
        memory_efficient=True
    )
    
    # Move to device
    model = model.to(device)
    
    # Calculate and report parameter count
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameter count: {num_params:,}")
    
    # Break down parameters by layer
    logging.info("Parameter breakdown:")
    total = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total += param_count
        logging.info(f"{name}: {param_count:,}")
    
    logging.info(f"Total confirmed parameters: {total:,}")
    
    # Calculate memory usage
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    logging.info(f"Model size: {size_mb:.2f} MB")
    
    # Estimate memory needed during training
    batch_size = 16
    sequence_length = 256
    estimated_memory = size_mb * 4  # Roughly 4x model size needed for training
    logging.info(f"Estimated memory needed for batch_size={batch_size}, seq_length={sequence_length}: ~{estimated_memory:.2f} MB")
    
    logging.info("Diagnostic complete!")

if __name__ == "__main__":
    main() 