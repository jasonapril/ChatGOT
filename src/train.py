"""
Main Training Script
===================

This script is the main entry point for training character-level transformer
models. It provides:

1. Robust command-line interface for configuration
2. Comprehensive training pipeline with memory optimization
3. Automatic model checkpointing and logging
4. CUDA-optimized training for maximum throughput
5. Text generation capabilities for model evaluation

The training process is optimized specifically for NVIDIA GTX 1650 Ti GPUs
with 4GB VRAM, but automatically scales to utilize larger GPUs if available.

Usage:
    python train.py --data_path processed_data/got_char_data.pkl --epochs 10
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

# Import local modules
from src.logger import setup_logger, log_section_header, force_flush_logs, format_time
from src.memory_management import get_memory_optimized_settings, preallocate_gpu_memory
from src.model import create_transformer_model, TransformerModel
from src.data_handler import load_data
from src.trainer import train_epoch, evaluate, generate_text, sample_text
from src.utils import (
    set_seed, 
    setup_device, 
    save_checkpoint, 
    load_checkpoint, 
    get_latest_checkpoint,
    create_output_dir,
    save_args,
    estimate_model_memory
)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    # Get default model parameters from TransformerModel
    default_model_params = {
        "d_model": TransformerModel.__init__.__defaults__[0],  # Default d_model
        "n_head": TransformerModel.__init__.__defaults__[1],    # Default n_head
        "d_hid": TransformerModel.__init__.__defaults__[2],    # Default d_hid
        "n_layers": TransformerModel.__init__.__defaults__[3], # Default n_layers
        "dropout": TransformerModel.__init__.__defaults__[4],  # Default dropout
        "max_seq_length": TransformerModel.__init__.__defaults__[5], # Default sequence length
        "layer_norm_eps": TransformerModel.__init__.__defaults__[6], # Default layer_norm_eps
    }
    
    parser = argparse.ArgumentParser(
        description="Train a character-level transformer model on text data."
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
                        help="Use automatic mixed precision training.")
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Use torch.compile to optimize model (requires PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Compilation mode for torch.compile")
    parser.add_argument("--use_bfloat16", action="store_true",
                        help="Use bfloat16 instead of float16 for mixed precision.")
    parser.add_argument("--max_memory_usage", type=float, default=None,
                        help="Maximum memory usage as fraction of total GPU memory (0-1).")
    parser.add_argument("--force_aggressive_memory", action="store_true",
                        help="Force aggressive memory pre-allocation for GTX 1650 Ti.")
    
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
    
    # Parse the arguments
    args = parser.parse_args()
    
    # If output_dir is not specified, create a timestamped one
    if args.output_dir is None:
        args.output_dir = create_output_dir("runs")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set log file if not specified
    if args.log_file is None:
        args.log_file = os.path.join(args.output_dir, "training.log")
    
    return args

def create_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    total_steps: int
) -> Tuple[optim.Optimizer, Optional[Any]]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The model to optimize
        args: Command line arguments
        total_steps: Total number of training steps
        
    Returns:
        Tuple of optimizer and scheduler
    """
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Create learning rate scheduler
    if args.use_onecycle:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            div_factor=args.div_factor,
            pct_start=0.3,
            final_div_factor=10000.0,
            anneal_strategy='cos'
        )
        logging.info(f"Using OneCycleLR scheduler with max_lr={args.max_lr}")
    else:
        # Linear warmup followed by constant LR
        def lr_lambda(current_step: int) -> float:
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logging.info(f"Using LambdaLR scheduler with warmup_steps={args.warmup_steps}")
    
    return optimizer, scheduler

def log_training_info(args: argparse.Namespace, model: nn.Module, device: torch.device, vocab_size: int) -> None:
    """
    Log detailed information about the training setup.
    
    Args:
        args: Command line arguments
        model: The model being trained
        device: The device being used
        vocab_size: Size of the vocabulary
    """
    log_section_header("TRAINING CONFIGURATION")
    
    # Log hardware information
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA: {torch.version.cuda}")
        logging.info(f"PyTorch CUDA: {torch.version.cuda}")
        
        # Get total GPU memory and calculate percentage used
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU Memory: {gpu_mem:.2f} GB")
        
        if args.max_memory_usage is not None:
            target_mem = gpu_mem * args.max_memory_usage
            logging.info(f"[MEMORY] Target GPU memory usage: {target_mem:.2f} GB ({args.max_memory_usage*100:.1f}%)")
    
    # Log model information
    logging.info(f"Model Architecture: Transformer (GPT-2 Style)")
    
    # Calculate and log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    
    # Log memory estimate
    memory_info = estimate_model_memory(model)
    logging.info(f"Estimated Model Size: {memory_info['model_size_float32_mb']:.2f} MB (FP32)")
    logging.info(f"Estimated Training Memory: {memory_info['training_memory_mb']:.2f} MB")
    
    # Log training parameters
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    logging.info(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Number of Epochs: {args.epochs}")
    
    # Log optimization settings
    logging.info(f"Using AMP: {args.use_amp}")
    if args.use_amp:
        precision = "bfloat16" if args.use_bfloat16 else "float16"
        logging.info(f"AMP Precision: {precision}")
    
    logging.info(f"Max Gradient Norm: {args.max_grad_norm}")
    
    # Log optimizer and scheduler information
    logging.info(f"Optimizer: AdamW")
    if args.use_onecycle:
        logging.info(f"LR Scheduler: OneCycleLR (max_lr={args.max_lr}, div_factor={args.div_factor})")
    else:
        logging.info(f"LR Scheduler: Linear warmup ({args.warmup_steps} steps) then constant")
    
    # Log paths information
    logging.info(f"Data Path: {args.data_path}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    logging.info(f"Log File: {args.log_file}")
    
    # Log memory optimization information
    if args.force_aggressive_memory:
        logging.info("[MEMORY] Aggressive memory optimization enabled")
    
    # Log reproducibility information
    logging.info(f"Random Seed: {args.seed}")
    
    force_flush_logs()

def main() -> None:
    """
    Main training function.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logger(args.log_file, args.log_level)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save the arguments
    save_args(args, args.output_dir)
    
    # Set up device
    device, is_cuda, num_gpus = setup_device(args.force_cpu)
    
    # Get optimized settings for memory usage
    if is_cuda:
        # Get GPU name for optimization
        gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
        
        # Get optimized settings based on GPU
        optimized_settings = get_memory_optimized_settings(gpu_name, args.force_aggressive_memory)
        
        # Apply settings if not overridden by command line
        if args.batch_size is None:
            args.batch_size = optimized_settings['batch_size']
            logging.info(f"[MEMORY] Using optimized batch size: {args.batch_size}")
        
        if args.gradient_accumulation_steps is None:
            args.gradient_accumulation_steps = optimized_settings['gradient_accumulation_steps']
            logging.info(f"[MEMORY] Using optimized gradient accumulation steps: {args.gradient_accumulation_steps}")
        
        if args.max_memory_usage is None:
            args.max_memory_usage = optimized_settings['max_memory_usage']
            logging.info(f"[MEMORY] Setting VRAM utilization target to {args.max_memory_usage*100:.1f}%")
        
        # Apply AMP settings if not specified
        if not hasattr(args, 'use_amp') or args.use_amp is None:
            args.use_amp = optimized_settings.get('use_amp', False)
            if args.use_amp:
                logging.info("[MEMORY] Enabling Automatic Mixed Precision for optimal throughput")
        
        # Pre-allocate memory if aggressive optimization is enabled
        if args.force_aggressive_memory:
            logging.info("[MEMORY] Pre-allocating maximum CUDA memory to force high throughput...")
            allocated_mb, allocated_pct = preallocate_gpu_memory(
                device, 
                target_memory_mb=torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) * args.max_memory_usage,
                max_memory_usage=args.max_memory_usage
            )
            logging.info(f"[MEMORY] Ready for training with {allocated_mb:.1f}MB reserved ({allocated_pct:.1f}% of VRAM)")
    else:
        # Set default values for CPU
        if args.batch_size is None:
            args.batch_size = 8
        if args.gradient_accumulation_steps is None:
            args.gradient_accumulation_steps = 1
        if args.max_memory_usage is None:
            args.max_memory_usage = 1.0
    
    # Load data
    log_section_header("LOADING DATA")
    train_loader, val_loader, char_to_idx, idx_to_char = load_data(
        args.data_path,
        batch_size=args.batch_size,
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        num_workers=args.num_workers
    )
    
    vocab_size = len(char_to_idx)
    
    # Create model
    log_section_header("CREATING MODEL")
    model = create_transformer_model(
        vocab_size=vocab_size,
        max_seq_length=args.sequence_length,
        d_model=args.d_model,
        n_head=args.n_head,
        d_hid=args.d_hid,
        n_layers=args.n_layers,
        dropout=args.dropout,
        memory_efficient=not args.disable_mem_efficient,
        layer_norm_eps=args.layer_norm_eps
    )
    
    # Move model to the appropriate device
    model = model.to(device)
    logging.info(f"Model moved to {device}")
    
    # If using CUDA, maximize memory usage by storing batch data directly on GPU
    if is_cuda and args.force_aggressive_memory:
        logging.info("[MEMORY] Enabling aggressive GPU memory strategies for data loading and processing")
        # Try to pre-fetch batches to GPU for better throughput
        torch.cuda.empty_cache()  # Ensure we have clean GPU memory
        current_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        logging.info(f"[MEMORY] Current VRAM utilization: {current_usage*100:.1f}%")
    
    # Calculate total steps for learning rate scheduler
    total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, total_steps)
    
    # Set up mixed precision training
    if args.use_amp:
        if device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            logging.info(f"Automatic Mixed Precision enabled with {torch.bfloat16 if args.use_bfloat16 else torch.float16}")
        else:
            # AMP not available for CPU, disabling
            args.use_amp = False
            scaler = None
            logging.info("Automatic Mixed Precision disabled (not supported on CPU)")
    else:
        scaler = None
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume or args.resume_from:
        log_section_header("RESUMING FROM CHECKPOINT")
        
        if args.resume_from:
            checkpoint_path = args.resume_from
            logging.info(f"Resuming from specified checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
            logging.info(f"Resuming from latest checkpoint: {checkpoint_path}")
        
        if checkpoint_path:
            checkpoint = load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device
            )
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('loss', float('inf'))
            
            # Ensure character mappings match
            loaded_char_to_idx = checkpoint.get('char_to_idx', {})
            if loaded_char_to_idx and loaded_char_to_idx != char_to_idx:
                logging.warning("Character mappings in checkpoint don't match current data!")
            
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            logging.warning("No checkpoint found, starting from scratch.")
    
    # Log training information
    log_training_info(args, model, device, vocab_size)
    
    # Enable cuDNN benchmarking for faster training if available
    if device.type == 'cuda' and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("[SPEED] Enabled cuDNN benchmarking for faster training")
    
    # Main training loop
    log_section_header("STARTING TRAINING")
    
    # Track training metrics
    train_losses = []
    val_losses = []
    best_checkpoint_path = None
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, tokens_per_sec = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.use_amp,
            use_bfloat16=args.use_bfloat16,
            use_torch_compile=args.use_torch_compile,
            compile_mode=args.compile_mode,
            scaler=scaler
        )
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        log_section_header(f"VALIDATION (EPOCH {epoch})")
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = format_time(epoch_time)
        
        # Log epoch results
        logging.info(f"Epoch {epoch} Summary:")
        logging.info(f"- Train Loss: {train_loss:.4f}")
        logging.info(f"- Validation Loss: {val_loss:.4f}")
        logging.info(f"- Tokens/second: {tokens_per_sec:.1f}")
        logging.info(f"- Epoch Time: {epoch_time_str}")
        
        # Generate sample text for monitoring
        if epoch % 1 == 0:  # Generate every epoch
            log_section_header(f"SAMPLE GENERATION (EPOCH {epoch})")
            sample_text_output = sample_text(
                model=model,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                device=device,
                seed_text="TYRION: ",
                max_length=100,
                temperature=args.temperature
            )
            
            # Log sample text to a separate file
            sample_file = os.path.join(args.output_dir, f"sample_epoch_{epoch}.txt")
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(sample_text_output)
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            logging.info(f"Saving checkpoint for epoch {epoch}...")
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss,
                args=args,
                tokenizer_data={'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char},
                checkpoint_dir=args.checkpoint_dir
            )
            
            # Save the best model separately
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model.pt")
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'args': vars(args),
                    'char_to_idx': char_to_idx,
                    'idx_to_char': idx_to_char
                }, best_checkpoint_path)
                
                logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Force flush logs to ensure they're written to disk
        force_flush_logs()
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    training_time_str = format_time(total_training_time)
    
    # Log final training summary
    log_section_header("TRAINING COMPLETE")
    logging.info(f"Total training time: {training_time_str}")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Best model saved to: {best_checkpoint_path}")
    
    # Generate text after training if requested
    if args.generate:
        log_section_header("GENERATING TEXT")
        
        # Use the best model for generation
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            # Load best model
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model for text generation")
        
        # Generate text
        generated_text = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=args.generate_seed,
            max_length=args.generate_length,
            temperature=args.temperature,
            device=device
        )
        
        # Save generated text to file
        generation_file = os.path.join(args.output_dir, "generated_text.txt")
        with open(generation_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        
        logging.info(f"Generated text saved to: {generation_file}")
    
    logging.info("Training completed successfully!")
    force_flush_logs()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Error occurred during training: {e}")
        sys.exit(1) 