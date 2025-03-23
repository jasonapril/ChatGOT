#!/usr/bin/env python3
"""
ChatGoT Text Generation Script
==============================

A standalone script for generating text from a trained model.
This script allows you to generate text without running the full training process.

Usage:
    python generate_text.py --checkpoint checkpoints/best_model.pt --seed "TYRION: " --length 1000
"""

import argparse
import os
import sys
import torch
import logging
from src.logger import setup_logger, log_section_header, force_flush_logs
from src.model import create_transformer_model
from src.trainer import generate_text
from src.utils import setup_device

def parse_args():
    """Parse command line arguments for text generation."""
    parser = argparse.ArgumentParser(description="Generate text from a trained ChatGoT model.")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file.")
    
    # Optional arguments
    parser.add_argument("--seed", type=str, default="TYRION: ",
                        help="Seed text to start generation.")
    parser.add_argument("--length", type=int, default=1000,
                        help="Number of characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for sampling (higher = more random).")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Limit sampling to top k tokens (0 = disabled).")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability threshold.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Penalty for repeating tokens (1.0 = no penalty).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save generated text (default: print to console).")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force CPU usage even if CUDA is available.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file.")
    
    return parser.parse_args()

def main():
    """Main function for text generation."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_file = args.log_file or "generation.log"
    setup_logger(log_file, "INFO")
    
    # Log start of generation
    log_section_header("TEXT GENERATION")
    logging.info(f"Loading model from checkpoint: {args.checkpoint}")
    logging.info(f"Seed text: '{args.seed}'")
    logging.info(f"Generation length: {args.length}")
    logging.info(f"Temperature: {args.temperature}")
    
    # Set up device
    device, is_cuda, _ = setup_device(args.force_cpu)
    logging.info(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        logging.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        print(f"Error: Failed to load checkpoint: {e}")
        sys.exit(1)
    
    # Extract model parameters and data
    model_args = checkpoint.get('args', {})
    char_to_idx = checkpoint.get('char_to_idx', {})
    idx_to_char = checkpoint.get('idx_to_char', {})
    
    if not char_to_idx or not idx_to_char:
        logging.error("Character mappings not found in checkpoint!")
        print("Error: Character mappings not found in checkpoint!")
        sys.exit(1)
    
    # Create model with same configuration
    vocab_size = len(char_to_idx)
    
    # Extract model hyperparameters from checkpoint
    d_model = model_args.get('d_model', 256)
    n_head = model_args.get('n_head', 8)
    n_layers = model_args.get('n_layers', 6)
    d_hid = model_args.get('d_hid', 1024)
    dropout = model_args.get('dropout', 0.1)
    context_length = model_args.get('context_length', 256)
    
    logging.info(f"Recreating model with {vocab_size} characters, {d_model} dimensions, {n_layers} layers")
    
    # Create model
    model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        d_hid=d_hid,
        dropout=dropout,
        max_seq_length=context_length,
        device=device
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("Model weights loaded successfully")
    
    # Generate text
    logging.info("Generating text...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate text with specified parameters
    generated_text = generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seed_text=args.seed,
        max_length=args.length,
        temperature=args.temperature,
        device=device
    )
    
    # Save or print generated text
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        logging.info(f"Generated text saved to: {args.output_file}")
        print(f"Generated text saved to: {args.output_file}")
    else:
        # Print a separator for readability
        print("\n" + "="*80)
        print("GENERATED TEXT:")
        print("-"*80)
        print(generated_text)
        print("="*80 + "\n")
    
    logging.info("Text generation completed")
    force_flush_logs()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nText generation interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred during text generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 