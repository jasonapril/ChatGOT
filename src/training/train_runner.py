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
from torch.utils.data import DataLoader

# Import local modules
from src.models.transformer import create_transformer_model, TransformerModel
from src.data.dataset import load_data
from src.utils import (
    set_seed,
    setup_device,
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    create_output_dir,
    save_args,
    setup_logging,
    ensure_directory,
    load_json,
    save_json
)
from src.training.train_config import parse_args, prepare_training_config
from src.training.optimizations import (
    optimize_training_parameters,
    setup_mixed_precision,
    optimize_memory_usage,
    setup_8bit_optimizer,
    setup_torch_compile,
    configure_activation_checkpointing
)
from src.data.language_dataset import LanguageDataset
from src.models.gpt_decoder import GPTDecoderModel
from src.training.language_trainer import LanguageTrainer

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

class TrainRunner:
    """Runner class for training language models."""
    
    def __init__(self, config_path: str):
        """
        Initialize the training runner.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = load_json(config_path)
        self.config_path = config_path
        
        # Set up output directory
        self.output_dir = self.config.get('output_dir', 'runs')
        ensure_directory(self.output_dir)
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
    
    def run(self) -> None:
        """Run the training process."""
        # Initialize model
        model = self._initialize_model()
        model.to(self.device)
        logging.info(f"Initialized model: {type(model).__name__}")
        
        # Load dataset
        train_dataset, val_dataset = self._load_datasets()
        
        # Initialize trainer
        trainer = self._initialize_trainer(model)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4)
            )
        
        # Train the model
        logging.info("Starting training")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.get('epochs', 10)
        )
        
        # Save the final model
        final_model_path = os.path.join(self.output_dir, 'final_model.pt')
        trainer.save_checkpoint(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        
        # Save training results
        results_path = os.path.join(self.output_dir, 'training_results.json')
        save_json(results, results_path)
        logging.info(f"Training results saved to {results_path}")
        
        logging.info("Training completed")
    
    def _initialize_model(self) -> torch.nn.Module:
        """
        Initialize the model based on configuration.
        
        Returns:
            Initialized model
        """
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'transformer')
        
        if model_type == 'transformer':
            return TransformerModel(model_config)
        elif model_type == 'gpt_decoder':
            return GPTDecoderModel(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_datasets(self) -> tuple:
        """
        Load datasets for training and validation.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        data_config = self.config.get('data', {})
        
        train_dataset = LanguageDataset(
            data_path=data_config.get('train_path'),
            vocab_size=data_config.get('vocab_size', 256),
            sequence_length=data_config.get('sequence_length', 128),
            is_character_level=data_config.get('is_character_level', True)
        )
        
        val_dataset = None
        if 'val_path' in data_config:
            val_dataset = LanguageDataset(
                data_path=data_config.get('val_path'),
                vocab_size=data_config.get('vocab_size', 256),
                sequence_length=data_config.get('sequence_length', 128),
                is_character_level=data_config.get('is_character_level', True),
                vocab=train_dataset.vocab  # Share vocabulary with training dataset
            )
        
        return train_dataset, val_dataset
    
    def _initialize_trainer(self, model: torch.nn.Module) -> LanguageTrainer:
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            
        Returns:
            Initialized trainer
        """
        # Get optimizer configuration
        optim_config = self.config.get('optimizer', {})
        optim_type = optim_config.get('type', 'adam')
        lr = optim_config.get('learning_rate', 0.001)
        
        # Initialize optimizer
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=optim_config.get('weight_decay', 0.0)
            )
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=optim_config.get('momentum', 0.9),
                weight_decay=optim_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_type}")
        
        # Initialize trainer
        trainer = LanguageTrainer(
            model=model,
            optimizer=optimizer,
            device=self.device,
            config=self.config
        )
        
        # Load checkpoint if specified
        if 'resume_from' in self.config:
            checkpoint_path = self.config['resume_from']
            if os.path.exists(checkpoint_path):
                logging.info(f"Resuming from checkpoint: {checkpoint_path}")
                trainer.load_checkpoint(checkpoint_path)
        
        return trainer

if __name__ == "__main__":
    main() 