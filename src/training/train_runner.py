#!/usr/bin/env python
"""
Training Runner Module
=========================

This module provides the main entry point for model training with all optimizations.
It coordinates the entire training process including:

1. Data loading and preparation
2. Model creation and optimization
3. Training loop execution
4. Checkpoint saving and loading
5. Progress monitoring and reporting

Usage:
    python -m src.training.train_runner --data_path processed_data/got_char_data.pkl
"""

import argparse
import logging
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm

# Import local modules
from src.logger import setup_logger, log_section_header, force_flush_logs, format_time
from src.model import create_transformer_model, TransformerModel
from src.data_handler import load_data
from src.trainer import train_epoch, evaluate, generate_text
from src.utils.model import set_seed
from src.utils.device import setup_device
from src.utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from src.utils.io import create_output_dir, save_args
from src.training.train_config import parse_args, prepare_training_config
from src.training.optimizations import (
    optimize_training_parameters,
    setup_mixed_precision,
    optimize_memory_usage,
    setup_8bit_optimizer,
    setup_torch_compile,
    configure_activation_checkpointing
)

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logger(args.log_file, args.log_level)
    
    # Log training start
    logging.info(f"Starting optimized training with PyTorch {torch.__version__}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup device (CPU/GPU)
    device, is_cuda, device_name = setup_device(args.force_cpu)
    logging.info(f"Using device: {device_name}")
    
    # Load processed data
    log_section_header("LOADING DATA")
    logging.info(f"Loading processed data from {args.data_path}")
    
    data_start = time.time()
    try:
        data = load_data(args.data_path)
        char_to_idx = data["char_to_idx"]
        idx_to_char = data["idx_to_char"]
        text_data = data["data"]
        
        logging.info(f"Data loaded: {len(text_data)} characters, {len(char_to_idx)} unique characters")
        logging.info(f"Data loading completed in {time.time() - data_start:.2f} seconds")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Optimize training parameters
    log_section_header("OPTIMIZING PARAMETERS")
    args = optimize_training_parameters(args, device, is_cuda, char_to_idx)
    
    # Create dataloaders
    # (Implementation depends on how your data_handler module works)
    
    # Create model
    log_section_header("CREATING MODEL")
    model = create_transformer_model(
        vocab_size=len(char_to_idx),
        d_model=args.d_model,
        n_head=args.n_head,
        d_hid=args.d_hid,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_length=args.sequence_length,
        layer_norm_eps=args.layer_norm_eps,
        mem_efficient_attn=not args.disable_mem_efficient
    )
    
    # Log model architecture and size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created with {total_params:,} total parameters")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Apply activation checkpointing if requested
    configure_activation_checkpointing(args, model)
    
    # Move model to device
    model.to(device)
    
    # Optimize memory usage
    optimize_memory_usage(args, device)
    
    # Set up mixed precision training
    use_amp, scaler = setup_mixed_precision(args)
    
    # Set up optimizer
    log_section_header("SETTING UP OPTIMIZER")
    
    # Try 8-bit optimizer first
    optimizer = setup_8bit_optimizer(args, model)
    
    # Fall back to standard optimizer if 8-bit not used
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        logging.info(f"Using standard AdamW optimizer with learning rate {args.lr}")
    
    # Set up learning rate scheduler
    if args.use_onecycle:
        # OneCycle learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_dataloader) // args.gradient_accumulation_steps,
            pct_start=0.3,
            div_factor=args.div_factor,
            final_div_factor=1000
        )
        logging.info(f"Using OneCycleLR scheduler with max_lr={args.max_lr}")
    else:
        # Linear warmup scheduler
        from transformers import get_linear_schedule_with_warmup
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        logging.info(f"Using linear warmup scheduler with {args.warmup_steps} warmup steps")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume or args.resume_from:
        checkpoint_path = args.resume_from if args.resume_from else get_latest_checkpoint(args.checkpoint_dir)
        
        if checkpoint_path:
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
            
            if checkpoint:
                start_epoch = checkpoint.get('epoch', 0) + 1
                logging.info(f"Resuming training from epoch {start_epoch}")
            else:
                logging.warning("Failed to load checkpoint, starting from epoch 0")
        else:
            logging.warning("No checkpoint found, starting from scratch")
    
    # Save args for reference
    save_args(args, os.path.join(args.output_dir, "training_args.json"))
    
    # Apply torch.compile if requested
    model = setup_torch_compile(args, model)
    
    # Training loop
    log_section_header("STARTING TRAINING")
    logging.info(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
    logging.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logging.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    best_val_loss = float('inf')
    train_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_throughput = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
            device=device,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=use_amp,
            use_bfloat16=args.use_bfloat16,
            scaler=scaler
        )
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, device, use_amp=use_amp)
        
        # Log epoch results
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch {epoch+1}/{args.epochs} completed in {format_time(epoch_time)}")
        logging.info(f"Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
        logging.info(f"Training throughput: {train_throughput:.2f} tokens/sec")
        
        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=os.path.join(args.checkpoint_dir, f"best_model.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                args=args
            )
            logging.info(f"New best validation loss: {val_loss:.4f}, saved checkpoint")
        
        # Save regular checkpoint every save_every epochs
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                path=os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                args=args
            )
            logging.info(f"Saved checkpoint for epoch {epoch+1}")
        
        # Force logs to be written
        force_flush_logs()
    
    # Training completed
    total_time = time.time() - train_start_time
    logging.info(f"Training completed in {format_time(total_time)}")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Generate text if requested
    if args.generate:
        log_section_header("GENERATING TEXT")
        generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=args.generate_seed,
            max_length=args.generate_length,
            temperature=args.temperature,
            device=device
        )

if __name__ == "__main__":
    main() 