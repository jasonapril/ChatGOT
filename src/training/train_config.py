#!/usr/bin/env python
"""
Training Configuration Module
=========================

This module handles all the configuration and argument parsing for the training process,
with options for:

1. Model architecture settings
2. Training hyperparameters
3. Optimization techniques
4. System and hardware configurations
5. Text generation parameters

Usage:
    Typically used by train_runner.py to parse command-line arguments.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Tuple, Optional, List

from src.utils import create_output_dir

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training.
    
    Returns:
        Parsed arguments
    """
    # Default model parameters - to be imported from model
    default_model_params = {
        "d_model": 768,  # Default d_model
        "n_head": 12,    # Default n_head
        "d_hid": 3072,   # Default d_hid
        "n_layers": 12,  # Default n_layers
        "dropout": 0.1,  # Default dropout
        "max_seq_length": 1024, # Default sequence length
        "layer_norm_eps": 1e-5, # Default layer_norm_eps
    }
    
    parser = argparse.ArgumentParser(
        description="Train a character-level transformer model on text data with maximum performance."
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the processed data file.")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (0-1).")
    parser.add_argument("--sequence_length", type=int, default=default_model_params["max_seq_length"],
                        help=f"Maximum sequence length (default: {default_model_params['max_seq_length']})")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")
    
    # Model configuration
    parser.add_argument("--d_model", type=int, default=default_model_params["d_model"],
                        help=f"Model embedding dimension (default: {default_model_params['d_model']})")
    parser.add_argument("--n_head", type=int, default=default_model_params["n_head"],
                        help=f"Number of attention heads (default: {default_model_params['n_head']})")
    parser.add_argument("--n_layers", type=int, default=default_model_params["n_layers"],
                        help=f"Number of transformer layers (default: {default_model_params['n_layers']})")
    parser.add_argument("--d_hid", type=int, default=default_model_params["d_hid"],
                        help=f"Hidden dimension of feedforward layers (default: {default_model_params['d_hid']})")
    parser.add_argument("--dropout", type=float, default=default_model_params["dropout"],
                        help=f"Dropout probability (default: {default_model_params['dropout']})")
    parser.add_argument("--disable_mem_efficient", action="store_true",
                        help="Disable memory efficient attention.")
    parser.add_argument("--layer_norm_eps", type=float, default=default_model_params["layer_norm_eps"],
                        help="Layer normalization epsilon.")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (None for auto).")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping.")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Number of steps to accumulate gradients (None for auto).")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision (may increase speed if supported by hardware)")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 instead of float16 for mixed precision.")
    parser.add_argument("--max_memory_usage", type=float, default=None,
                        help="Maximum memory usage as fraction of total GPU memory (0-1).")
    parser.add_argument("--force_aggressive_memory", action="store_true",
                        help="Force aggressive memory pre-allocation for GTX 1650 Ti.")
    parser.add_argument("--optimize_cuda", action="store_true",
                        help="Apply all CUDA optimizations before training.")
    parser.add_argument("--optimize_batch_size", action="store_true",
                        help="Find optimal batch size before training.")
    parser.add_argument("--test_batches", type=int, default=5,
                        help="Number of batches to test during batch size optimization.")
    
    # System and output configuration
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs and checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a specific checkpoint.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if CUDA is available.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level.")
    
    # OneCycle LR scheduler
    parser.add_argument("--use_onecycle", action="store_true",
                        help="Use OneCycleLR scheduler.")
    parser.add_argument("--max_lr", type=float, default=1e-3,
                        help="Maximum learning rate for OneCycleLR scheduler.")
    parser.add_argument("--div_factor", type=float, default=25,
                        help="Determines the initial learning rate for OneCycleLR.")
    
    # Text generation options
    parser.add_argument("--generate", action="store_true",
                        help="Generate text after training.")
    parser.add_argument("--generate_length", type=int, default=500,
                        help="Length of generated text.")
    parser.add_argument("--generate_seed", type=str, default="TYRION: ",
                        help="Seed text for generation.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for text generation.")
    
    # Custom gradient accumulation options
    parser.add_argument("--effective_batch_size", type=int, default=None,
                        help="Target effective batch size (batch_size * gradient_accumulation_steps).")
    
    # New torch.compile arguments
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Use torch.compile to optimize model (requires PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Compilation mode for torch.compile")
    
    # 8-bit optimizer options
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                        help="Use 8-bit optimizer to reduce memory usage (requires bitsandbytes)")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer implementation (requires bitsandbytes)")
    parser.add_argument("--use_8bit_adamw", action="store_true", 
                        help="Use 8-bit AdamW optimizer implementation (requires bitsandbytes)")
    
    # Activation checkpointing
    parser.add_argument("--use_activation_checkpointing", action="store_true",
                        help="Use activation checkpointing to reduce memory usage (trades compute for memory)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir is None:
        args.output_dir = create_output_dir("runs")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set log file if not specified
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "training.log")
    
    return args

def prepare_training_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert parsed arguments into a configuration dictionary.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of training configuration
    """
    config = vars(args).copy()
    
    # Set defaults where None was specified
    if config['batch_size'] is None:
        config['batch_size'] = 32  # Default batch size if not auto-determined
    
    if config['gradient_accumulation_steps'] is None:
        config['gradient_accumulation_steps'] = 1
    
    # Add additional config items as needed
    config['track_memory'] = True
    
    return config 