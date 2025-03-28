#!/usr/bin/env python
"""
Training script with sample generation

This script handles the training of transformer models with
periodic sample generation to visualize progress.
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
import functools
from datetime import datetime
import math
import types
import logging.handlers
import re
import gc

import hydra
import torch
import torch.nn.functional as F
import torch.distributed as dist
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# Try to import checkpoint functionality
try:
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
except ImportError:
    # Older PyTorch versions might have it here
    try:
        from torch.utils import checkpoint as torch_checkpoint
    except ImportError:
        torch_checkpoint = None

# Add the project root to the path so we can import the src package
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import CharDataset
from src.models.transformer import TransformerModel, create_transformer_model
from src.models.gpt_decoder import GPTDecoder, create_gpt_model
from src.utils.generation import generate_sample_text, sample_text
from src.utils.memory import get_memory_optimized_settings, preallocate_gpu_memory
from src.utils.metrics import calculate_tokens_per_second

# Set up logging
def setup_logging(config_name=None):
    """Set up logging with both console (colored) and file (clean) output."""
    # Create logs directory if it doesn't exist
    os.makedirs("outputs/logs", exist_ok=True)
    
    # Create a formatter without color codes for file logging
    file_formatter = logging.Formatter(
        '%(asctime)s [ %(levelname)s ] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a detailed file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"outputs/logs/train_{config_name}_{timestamp}.log" if config_name else f"outputs/logs/train_{timestamp}.log"
    
    # Set up file handler with rotation (10 MB max size, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Setup console handler with colors for terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [ %(levelname)s ] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add both handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logs will be saved to {log_filename}")
    
    return logger, log_filename

def load_config(config_path):
    """Load configuration from YAML file."""
    # First read and print the raw file contents
    with open(config_path, 'r') as f:
        config_str = f.read()
        print(f"Raw config content:\n{config_str}")
        
    # Then parse the YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Print the loaded config for debugging
    print(f"Parsed config:\n{config}")
    return config

def enable_gradient_checkpointing(model):
    """Safely enable gradient checkpointing with error handling."""
    # Get logger instance
    logger = logging.getLogger(__name__)
    
    if not hasattr(model, 'layers'):
        logger.warning("Model doesn't have 'layers' attribute, skipping gradient checkpointing")
        return False
    
    try:
        for i, layer in enumerate(model.layers):
            # Store the original forward function if not already stored
            if not hasattr(layer, 'forward_original'):
                layer.forward_original = layer.forward
                # Create safe wrapper with error handling
                def checkpoint_wrapper(fn, *args, **kwargs):
                    try:
                        return torch_checkpoint(fn, *args, **kwargs)
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.error(f"Error in gradient checkpointing: {e}")
                        # Fall back to standard forward pass
                        return fn(*args, **kwargs)
                
                layer.forward = functools.partial(checkpoint_wrapper, layer.forward_original)
                logger.info(f"Gradient checkpointing enabled for layer {i}")
        return True
    except Exception as e:
        logger.error(f"Failed to enable gradient checkpointing: {e}")
        # Restore original forward functions if any error occurs
        disable_gradient_checkpointing(model)
        return False

def disable_gradient_checkpointing(model):
    """Safely disable gradient checkpointing and restore original forward functions."""
    # Get logger instance
    logger = logging.getLogger(__name__)
    
    if not hasattr(model, 'layers'):
        return
    
    try:
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'forward_original'):
                layer.forward = layer.forward_original
                delattr(layer, 'forward_original')
                logger.info(f"Gradient checkpointing disabled for layer {i}")
        return True
    except Exception as e:
        logger.error(f"Error disabling gradient checkpointing: {e}")
        return False

def cleanup_old_checkpoints(checkpoint_dir, config_name, timestamp, max_to_keep=5):
    """Remove old checkpoints, keeping only the latest max_to_keep."""
    logger = logging.getLogger(__name__)
    
    try:
        # Find all checkpoints matching the pattern
        prefix = f"{config_name}_{timestamp}_step_"
        checkpoints = []
        
        for file in os.listdir(checkpoint_dir):
            if file.startswith(prefix) and file.endswith(".pt"):
                # Extract step number
                step_str = file[len(prefix):-3]
                if step_str.isdigit():
                    step = int(step_str)
                    checkpoints.append((step, os.path.join(checkpoint_dir, file)))
        
        # Sort by step number (descending)
        checkpoints.sort(reverse=True)
        
        # Keep only max_to_keep checkpoints
        if len(checkpoints) > max_to_keep:
            for _, checkpoint_path in checkpoints[max_to_keep:]:
                try:
                    os.remove(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {os.path.basename(checkpoint_path)}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    except Exception as e:
        logger.warning(f"Error during checkpoint cleanup: {e}")

# Add SafeGradScaler class for mixed precision training
class SafeGradScaler(GradScaler):
    """Enhanced GradScaler with NaN detection and fallback to full precision."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_counter = 0
        self.max_nan_before_fallback = 3
        self.fallback_triggered = False
        self.warning_logged = False
        
    def scale(self, loss):
        # Check for NaN before scaling
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.nan_counter += 1
            logger.warning(f"NaN/Inf detected in loss before scaling: {loss.item() if not torch.isinf(loss).all() else 'inf'} (occurrence {self.nan_counter}/{self.max_nan_before_fallback})")
            
            # If we've seen too many NaNs, disable mixed precision
            if self.nan_counter >= self.max_nan_before_fallback and not self.fallback_triggered:
                logger.error(f"Too many NaN/Inf values detected, disabling mixed precision")
                self._enabled = False
                self.fallback_triggered = True
                
                # Return unscaled loss to continue training in full precision
                return loss
        else:
            # Reset counter if loss is normal
            if self.nan_counter > 0:
                self.nan_counter = max(0, self.nan_counter - 1)  # Gradually decrease counter
                
        if not self._enabled and not self.warning_logged:
            logger.warning("Mixed precision is disabled. Using full precision for forward/backward passes.")
            self.warning_logged = True
            return loss
            
        return super().scale(loss)
    
    def step(self, optimizer, *args, **kwargs):
        # Additional safety around optimizer step
        if not self._enabled:
            # In full precision mode, just do a normal step
            return optimizer.step(*args, **kwargs)
            
        try:
            return super().step(optimizer, *args, **kwargs)
        except RuntimeError as e:
            if "inf or nan" in str(e).lower():
                logger.error(f"NaN/Inf detected during optimizer step, disabling mixed precision: {e}")
                self._enabled = False
                self.fallback_triggered = True
                
                # Attempt normal optimizer step as fallback
                try:
                    optimizer.zero_grad()
                    return optimizer.step(*args, **kwargs)
                except Exception as e2:
                    logger.error(f"Fallback optimizer step also failed: {e2}")
                    raise
            else:
                raise

def setup_tensorboard(config, config_name=None, log_file=None):
    """Set up TensorBoard logging."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = config.get('experiment_name', config_name or 'default')
    
    # Ensure the logs directory exists
    os.makedirs("outputs/logs/tensorboard", exist_ok=True)
    
    # Always use logs/tensorboard, ignoring any config setting
    log_dir = os.path.join('outputs/logs/tensorboard', f"{experiment_name}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log the tensorboard directory
    logger = logging.getLogger(__name__)
    logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    # Create a symlink to the log file in the tensorboard directory if available
    if log_file and os.path.exists(log_file):
        try:
            symlink_path = os.path.join(log_dir, "training.log")
            # On Windows, may need to use a different approach
            if os.name == 'nt':  # Windows
                # Just copy the file instead of symlink
                import shutil
                shutil.copy2(log_file, symlink_path)
            else:
                os.symlink(os.path.abspath(log_file), symlink_path)
            logger.info(f"Created link to log file in TensorBoard directory")
        except Exception as e:
            logger.warning(f"Could not create log file link in TensorBoard directory: {e}")
    
    return writer

def try_increase_batch_size(original_batch_size, current_batch_size, recent_losses, stable_window=5, variance_threshold=0.01):
    """Attempt to increase batch size if training has been stable."""
    if current_batch_size < original_batch_size and len(recent_losses) >= stable_window:
        # Check loss stability over recent window
        recent_losses_tensor = torch.tensor(recent_losses[-stable_window:])
        variance = torch.var(recent_losses_tensor).item()
        mean = torch.mean(recent_losses_tensor).item()
        
        # Calculate coefficient of variation (normalized variance)
        # This helps account for different loss scales
        cv = variance**0.5 / mean if mean > 0 else float('inf')
        
        # If loss is stable (low variance relative to mean)
        if cv < variance_threshold:
            new_batch_size = min(current_batch_size + 1, original_batch_size)
            return new_batch_size, True
    
    return current_batch_size, False

def get_gpu_memory_usage():
    """Get GPU memory usage statistics in a readable format."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    stats = {}
    
    # Current memory allocation
    stats['allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
    stats['cached'] = torch.cuda.memory_reserved() / (1024**2)  # MB
    
    # Peak memory stats during training
    stats['max_allocated'] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    stats['max_cached'] = torch.cuda.max_memory_reserved() / (1024**2)  # MB
    
    # Get device properties
    device_props = torch.cuda.get_device_properties(0)
    stats['total'] = device_props.total_memory / (1024**2)  # MB
    stats['device_name'] = device_props.name
    
    return stats

def train_with_samples(
    config_file: str,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 1,
    batch_size: int = None,
    log_interval: int = 100,
    sample_every: int = 1000,  
    sample_length: int = 100,
    sample_temperature: float = 0.8,
    grad_clip: float = 0.0,
    resume_from: str = None,  # New parameter to specify a checkpoint to resume from
):
    """Train with samples at specified intervals."""
    # Get config name from file path for logging
    config_name = os.path.basename(config_file).replace('.yaml', '')
    
    # Set up logging with both console and file output
    logger, log_file = setup_logging(config_name)
    
    logger.info(f"Loading config from {config_file}")
    # Load the config
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return

    logger.info(f"Setting up device {device_str}")
    device = torch.device(device_str)
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        logger.info("CUDA not available, using CPU")

    # Apply memory optimizations for CUDA
    if device_str == "cuda":
        logger.info("Applying memory optimizations for CUDA")
        # Check if PYTORCH_CUDA_ALLOC_CONF is set
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            logger.info(f"PYTORCH_CUDA_ALLOC_CONF already set to {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
        else:
            # Use memory allocation settings focused on performance, not memory savings
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF for maximum performance")
        
        # Completely disable memory pre-allocation - it causes performance issues
        logger.info("Memory pre-allocation disabled - using dynamic allocation for best performance")
        
        # Empty cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Implement memory tracking
        memory_stats = {}
        memory_stats['init'] = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"Initial CUDA memory usage: {memory_stats['init']:.2f} MB")
        
        # Set environment variables for better OOM handling but prioritize performance
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Set to 0 for better performance
        logger.info("Using asynchronous CUDA operations for better performance")

    # Parse configuration
    arch_config = config.get('architecture', {})
    # Check if architecture config might be in 'model' (for compatibility)
    if not arch_config and 'model' in config:
        arch_config = config.get('model', {})
        
    arch_type = arch_config.get('type', arch_config.get('architecture', 'gpt'))
    model_type = arch_config.get('model_type', 'transformer')
    logger.info(f"Architecture type: {arch_type}, Model type: {model_type}")

    # Get sequence length from config
    max_seq_length = arch_config.get('max_seq_length', 1024)
    logger.info(f"Using sequence length: {max_seq_length}")

    # Convert layer_norm_eps to float
    if 'layer_norm_eps' in arch_config and isinstance(arch_config['layer_norm_eps'], str):
        arch_config['layer_norm_eps'] = float(arch_config['layer_norm_eps'])
        logger.info(f"Converting layer_norm_eps to float: {arch_config['layer_norm_eps']}")

    # Override batch size if specified
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
        logger.info(f"Overriding batch size to {batch_size}")
    else:
        batch_size = config['training'].get('batch_size', 8)
    
    # Create dataset
    data_path = config.get('data', {}).get('path', 'data/shakespeare.txt')
    logger.info(f"Loading dataset from {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Loaded {len(text)} characters from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_path}: {e}")
        return
    
    # Create the dataset with the text content
    dataset = CharDataset(text=text, block_size=max_seq_length)
    
    # Set vocabulary size in config
    vocab_size = dataset.vocab_size
    arch_config['vocab_size'] = vocab_size
    logger.info(f"Dataset vocabulary size: {vocab_size}")
    
    # Extract model configuration
    d_model = arch_config.get('d_model', 768)
    n_head = arch_config.get('n_head', 12)
    n_layers = arch_config.get('n_layers', 12)
    d_hid = arch_config.get('d_hid', 3072)
    dropout = arch_config.get('dropout', 0.1)
    layer_norm_eps = arch_config.get('layer_norm_eps', 1e-5)
    activation = arch_config.get('activation', 'gelu')
    bias = arch_config.get('bias', True)
    
    # Training parameters
    learning_rate = float(config['training'].get('learning_rate', 3e-4))
    weight_decay = float(config['training'].get('weight_decay', 0.01))
    
    # Use a learning rate that's more effective for this dataset/model size
    # Too low can lead to no learning progress, too high can cause instability
    learning_rate = 5e-4  # Using a larger value to ensure visible learning progress
    logger.info(f"Using learning rate: {learning_rate}")
    
    # Re-enable gradient accumulation for better throughput (if it's set to 1)
    if config['training'].get('accumulate_grad_batches', 1) < 2:
        # Only override if set too low
        config['training']['accumulate_grad_batches'] = 2
        logger.info(f"Using gradient accumulation with steps=2 for better throughput")
    
    # Ensure gradient clipping is enabled, but not too aggressive
    if grad_clip <= 0 or grad_clip > 5.0:
        grad_clip = 1.0  # Use reasonable default if not specified
        logger.info(f"Setting gradient clipping to {grad_clip}")
    
    # Try a smaller batch size if the current one isn't working well with learning
    if batch_size > 6:
        batch_size_orig = batch_size
        batch_size = batch_size_orig  # Use the original batch size to utilize more GPU memory
        logger.info(f"Using full batch size of {batch_size} to maximize GPU utilization")
    
    # For higher GPU utilization, adjust the model size parameters if they're too small
    if d_model < 512 and d_hid < 2048:
        # Only adjust if current values are small
        old_d_model, old_d_hid = d_model, d_hid
        d_model = min(512, d_model * 1.5)  # Increase embedding dimension
        d_hid = min(2048, d_hid * 1.5)     # Increase hidden dimension
        logger.info(f"Adjusting model dimensions for better GPU utilization: d_model {old_d_model}->{d_model}, d_hid {old_d_hid}->{d_hid}")
        arch_config['d_model'] = d_model
        arch_config['d_hid'] = d_hid
    
    # Memory optimization parameters
    mixed_precision = config['training'].get('mixed_precision', False)
    gradient_checkpointing = config['training'].get('gradient_checkpointing', False)
    accumulate_grad_batches = config['training'].get('accumulate_grad_batches', 1)
    logger.info(f"Using gradient accumulation steps: {accumulate_grad_batches}")
    
    # Set up TensorBoard with our config name and log file
    writer = setup_tensorboard(config, config_name, log_file)
    
    # Create model based on model_type
    logger.info(f"Creating model with type: {model_type}")
    if model_type == "gpt_decoder":
        # Import the GPT decoder model if needed
        from src.models.gpt_decoder import create_gpt_model
        
        # Adjust initialization to start closer to random prediction
        # Use smaller init range for more conservative initial predictions
        init_range = arch_config.get('init_range', 0.02)
        init_range_adjusted = min(init_range, 0.01)  # Cap to avoid too large initial weights
        arch_config['init_range'] = init_range_adjusted
        
        model = create_gpt_model(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=n_head,
            d_hid=d_hid,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
            bias=bias
        )
        logger.info(f"Created optimized GPT decoder with initialization range: {init_range_adjusted}")
    else:
        # Default to standard transformer model
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_head=n_head,
            d_hid=d_hid,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
            bias=bias
        )
        logger.info("Created standard transformer model with cross-attention")
    model.to(device)
    
    # Enabling gradient checkpointing for memory efficiency
    if device.type == 'cuda' and torch_checkpoint is not None and gradient_checkpointing:
        logger.info("Attempting to enable gradient checkpointing for memory efficiency")
        if enable_gradient_checkpointing(model):
            logger.info("Successfully enabled gradient checkpointing")
        else:
            logger.warning("Failed to enable gradient checkpointing, continuing without it")
    else:
        if gradient_checkpointing:
            if device.type != 'cuda':
                logger.info("Gradient checkpointing only supported on CUDA devices")
            elif torch_checkpoint is None:
                logger.info("torch.utils.checkpoint not available in this PyTorch version")
            else:
                logger.info("Gradient checkpointing not requested in configuration")

    # Try to free up memory before training starts
    torch.cuda.empty_cache()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Using CrossEntropyLoss for character prediction")
    
    # Create data loaders
    num_workers = config.get('training', {}).get('num_workers', 2)  # Reduced from 4
    
    # Split dataset into train and val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Add learning rate scheduler after data loaders are created
    # Linear warmup for 10% of steps followed by cosine decay
    max_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=learning_rate * 0.1  # Final learning rate will be 10% of initial
    )
    
    # Replace the regular GradScaler with our SafeGradScaler
    # GradScaler for mixed precision training
    scaler = SafeGradScaler(enabled=mixed_precision) if mixed_precision else None
    
    # Add info log about mixed precision status
    if mixed_precision:
        logger.info("Mixed precision training enabled with SafeGradScaler (with NaN detection and fallback)")
    else:
        logger.info("Mixed precision training disabled, using full precision")
    
    # Add option to resume from checkpoint
    start_epoch = 0
    if resume_from:
        if os.path.isfile(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}")
            try:
                # First, make sure that any gradient checkpointing is completely disabled and reset
                # This is crucial to avoid the 'dict' object is not callable error
                if hasattr(model, 'layers'):
                    logger.info("Resetting all layer forward functions to avoid checkpoint loading issues")
                    # Create a fresh backup of the original forward method
                    for layer in model.layers:
                        # Reset the forward function to the original class implementation
                        if hasattr(layer, 'forward_original'):
                            # If we have a stored original forward, restore it
                            logger.info(f"  - Restoring original forward function for layer")
                            # Get the class method directly from the class (not the instance)
                            layer_cls = layer.__class__
                            original_forward = layer_cls.forward
                            # Replace the instance method with the class method
                            layer.forward = types.MethodType(original_forward, layer)
                            # Clean up the now unused attribute
                            delattr(layer, 'forward_original')
                
                # Force garbage collection before loading checkpoint
                gc.collect()
                torch.cuda.empty_cache()
                
                # Load the checkpoint
                checkpoint = torch.load(resume_from, map_location=device)
                
                # More robust state dict loading with strict=False to ignore missing/unexpected keys
                model_state_dict = checkpoint['model_state_dict']
                # Filter out unexpected keys before loading
                model_dict = model.state_dict()
                # Filter to include only keys that exist in the model
                filtered_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
                # Update the model's state dict with the filtered values
                model_dict.update(filtered_dict)
                # Load the filtered state dict
                model.load_state_dict(model_dict)
                
                # Disable gradient checkpointing after loading checkpoint
                if gradient_checkpointing:
                    logger.info("Re-enabling gradient checkpointing after loading checkpoint")
                    enable_gradient_checkpointing(model)
                else:
                    logger.info("Gradient checkpointing remains disabled after loading checkpoint")
                
                # Load optimizer state - if keys mismatch, start with fresh optimizer
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Move optimizer state to the correct device
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except Exception as opt_e:
                    logger.warning(f"Could not load optimizer state: {opt_e}")
                    logger.warning("Starting with fresh optimizer state")
                
                # Load other training state
                start_epoch = checkpoint.get('epoch', 0)
                global_step = checkpoint.get('global_step', 0)
                logger.info(f"Resuming from epoch {start_epoch}, global step {global_step}")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Starting training from scratch")
        else:
            logger.warning(f"Checkpoint file {resume_from} not found, starting from scratch")
    
    # Training loop
    global_step = global_step if 'global_step' in locals() else 0
    tokens_processed = 0
    start_time = time.time()
    
    # Initialize checkpoint naming variables
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create models directory if it doesn't exist
    os.makedirs("outputs/models", exist_ok=True)
    
    # Checkpoint settings - only time-based, not step-based
    last_checkpoint_time = time.time()
    checkpoint_interval_minutes = 2  # Save checkpoint every 2 minutes
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Model configuration: {arch_config}")
    logger.info(f"Training with batch size: {batch_size}, learning rate: {learning_rate}")
    logger.info(f"Checkpoints will be saved every {checkpoint_interval_minutes} minutes to: outputs/models/{config_name}_{timestamp}_step_X.pt")
    
    # Ensure all parameters have requires_grad=True
    for param in model.parameters():
        param.requires_grad = True
    
    # Verify parameter count and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    # Time-based sampling settings
    sample_interval_minutes = 2  # Generate sample every 2 minutes (increased from 1)
    last_sample_time = time.time()
    
    # Track tokens per second
    tokens_per_second_window = []
    window_size = 50  # Increase window size for more stability
    
    # For ETA calculation
    epoch_start_time = 0
    running_batch_times = []
    max_running_times = 100  # Keep more batch times for a more stable average
    eta_smoothing_factor = 0.8  # Much heavier smoothing - 80% previous, 20% current
    last_eta = None
    
    # Initialize median filter window for more stable ETA
    eta_window = []
    eta_window_size = 5

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            epoch_start_time = time.time()
            
            # Track loss trend for detecting training instability
            previous_losses = []
            max_loss_increase_count = 5  # How many consecutive increases before we intervene
            loss_increase_count = 0
            
            # Track original batch size for potential recovery
            original_batch_size = config['training'].get('batch_size', 8)
            batch_size_recovery_attempts = 0
            epochs_since_last_recovery = 0
            
            # Check if we should try to recover batch size at the start of the epoch
            if batch_size < original_batch_size and epoch > 0:
                # Only try recovery if we've had at least one stable epoch
                if len(previous_losses) >= 50 and epochs_since_last_recovery >= 1:
                    new_batch_size, increased = try_increase_batch_size(
                        original_batch_size=original_batch_size,
                        current_batch_size=batch_size,
                        recent_losses=previous_losses,
                        stable_window=50,  # Use a larger window for more confidence
                        variance_threshold=0.02
                    )
                    
                    if increased:
                        # Batch size increased, update data loaders
                        logger.info(f"Training has been stable. Increasing batch size from {batch_size} to {new_batch_size}")
                        batch_size = new_batch_size
                        batch_size_recovery_attempts += 1
                        epochs_since_last_recovery = 0
                        
                        # Recreate data loaders with new batch size
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available()
                        )
                        
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=torch.cuda.is_available()
                        )
                        
                        # Update config
                        config['training']['batch_size'] = batch_size
                
                epochs_since_last_recovery += 1
            
            for batch_idx, (x, y) in enumerate(train_loader):
                batch_start = time.time()
                
                # Move data to device
                x, y = x.to(device), y.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Get accumulation steps
                is_accumulating = (batch_idx % accumulate_grad_batches != 0)
                
                # Forward and loss computation with mixed precision
                try:
                    # Forward pass
                    model_outputs = model(x)
                    
                    # Log model output type to debug
                    if batch_idx == 0:  # Only log once
                        logger.info(f"Model output type: {type(model_outputs)}")
                        if isinstance(model_outputs, tuple):
                            logger.info(f"Model outputs tuple length: {len(model_outputs)}")
                            for i, item in enumerate(model_outputs):
                                logger.info(f"  Output item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                    
                    # Handle different model output formats
                    if isinstance(model_outputs, tuple):
                        if len(model_outputs) >= 2:
                            # Format could be (logits, loss) or (logits, _, attn_weights)
                            logits = model_outputs[0]
                            
                            # Check if second element is loss
                            if isinstance(model_outputs[1], torch.Tensor) and model_outputs[1].numel() == 1:
                                # Use provided loss
                                unscaled_loss = model_outputs[1]
                            else:
                                # Calculate loss from logits
                                unscaled_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                        else:
                            # Single item tuple
                            logits = model_outputs[0]
                            unscaled_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    else:
                        # Direct tensor output
                        logits = model_outputs
                        unscaled_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    
                    # Ensure unscaled_loss is tensor and has the right shape
                    if not isinstance(unscaled_loss, torch.Tensor):
                        logger.error(f"Loss is not a tensor: {unscaled_loss}")
                        unscaled_loss = torch.tensor(unscaled_loss, device=device)
                    
                    # Enhanced loss checks to detect NaN/Inf
                    if torch.isnan(unscaled_loss).any() or torch.isinf(unscaled_loss).any():
                        logger.warning(f"NaN/Inf loss detected after forward pass: {unscaled_loss if not torch.isinf(unscaled_loss).all() else 'inf'}")
                        
                        # Try to diagnose the issue
                        logits_info = f"Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}"
                        logger.warning(f"Abnormal outputs detected: {logits_info}")
                        
                        # If we're accumulating gradients, skip this batch but continue training
                        if is_accumulating:
                            logger.info("Skipping gradient accumulation for this batch")
                            continue
                    
                    # Continue with mixed precision handling
                    if scaler is not None:
                        # AMP: Scale the loss and compute gradients
                        scaled_loss = scaler.scale(unscaled_loss)
                        # Avoid accumulation on the first step of each batch
                        if not is_accumulating:
                            scaled_loss.backward(create_graph=False, retain_graph=False)
                        else:
                            # No need for model.no_sync() as we're not using distributed training
                            scaled_loss.backward(create_graph=False, retain_graph=False)
                    else:
                        # Regular full precision mode
                        if not is_accumulating:
                            unscaled_loss.backward(create_graph=False, retain_graph=False)
                        else:
                            # No need for model.no_sync() as we're not using distributed training
                            unscaled_loss.backward(create_graph=False, retain_graph=False)
                    
                    # Immediately delete variables no longer needed to free GPU memory
                    # removing this to avoid errors - delete after diagnostics
                    # del logits
                    # torch.cuda.empty_cache()
                    
                    # If no scaler or if scaler with fallback to full precision
                    if (scaler is None or not scaler._enabled) and (torch.isnan(unscaled_loss).any() or torch.isinf(unscaled_loss).any()):
                        logger.warning("NaN/Inf detected in full precision mode, skipping parameter update")
                        optimizer.zero_grad()  # Clear gradients to avoid accumulating bad values
                        continue
                    
                    # Only update after gradient_accumulation_steps
                    if (batch_idx + 1) % accumulate_grad_batches == 0 or batch_idx == len(train_loader) - 1:
                        # Apply gradient clipping to prevent exploding gradients
                        if grad_clip > 0:
                            if scaler is not None and scaler._enabled:
                                # For mixed precision
                                scaler.unscale_(optimizer)
                            
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        
                        # Step with scaler in mixed precision or regular optimizer step
                        if scaler is not None and scaler._enabled:
                            # Update weights with gradient scaler for mixed precision
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Regular optimizer step in full precision
                            optimizer.step()
                        
                        # Zero gradients
                        optimizer.zero_grad()
                        
                        # Step the scheduler if it's time-based
                        scheduler.step()
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Handle CUDA OOM errors
                        logger.error(f"CUDA out of memory error: {e}")
                        
                        # More aggressive memory cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        
                        # Log memory state before reduction
                        if device.type == 'cuda':
                            memory_stats = get_gpu_memory_usage()
                            logger.warning(f"Memory before batch reduction - Used: {memory_stats['allocated']:.1f} MB, "
                                           f"Cached: {memory_stats['cached']:.1f} MB, "
                                           f"Max: {memory_stats['max_allocated']:.1f} MB / {memory_stats['total']:.1f} MB")
                        
                        # Reduce batch size and try again
                        if batch_size > 1:
                            old_batch_size = batch_size
                            batch_size = max(1, batch_size - 1)
                            logger.warning(f"Reducing batch size from {old_batch_size} to {batch_size} due to OOM error")
                            
                            # Recreate data loaders with new batch size
                            train_loader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=torch.cuda.is_available()
                            )
                            
                            val_loader = DataLoader(
                                val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=torch.cuda.is_available()
                            )
                            
                            # Update config
                            config['training']['batch_size'] = batch_size
                            
                            # Skip to the next epoch
                            break
                        else:
                            logger.error("Batch size already at minimum, cannot reduce further")
                            raise
                    else:
                        # Log other types of errors
                        logger.error(f"Runtime error during training: {e}")
                        raise
                
                # Check for NaN or inf in loss
                if torch.isnan(unscaled_loss) or torch.isinf(unscaled_loss):
                    logger.error(f"Loss is {unscaled_loss}, stopping training")
                    raise ValueError("Loss is NaN or inf")
                
                # Detect training instability - check if loss is consistently increasing
                if len(previous_losses) >= 20:
                    previous_losses.pop(0)  # Remove oldest loss
                
                previous_losses.append(unscaled_loss)
                
                # Check if loss is consistently increasing (but only after the first 100 batches)
                if batch_idx > 100 and len(previous_losses) >= 5:
                    # Check if the most recent loss is higher than the average of the previous 5
                    recent_avg = sum(previous_losses[-6:-1]) / 5
                    current = previous_losses[-1]
                    
                    if current > recent_avg * 1.2:  # 20% increase threshold
                        loss_increase_count += 1
                        if loss_increase_count >= max_loss_increase_count:
                            logger.warning(f"Loss increasing consistently over {max_loss_increase_count} checks, reducing learning rate by 50%")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            loss_increase_count = 0  # Reset the counter
                    else:
                        loss_increase_count = 0  # Reset counter if loss isn't increasing
                
                # Diagnostic: Check logits for sanity
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        # Check if logits have suspicious patterns
                        logits_detached = logits.detach()
                        logits_mean = logits_detached.mean().item()
                        logits_std = logits_detached.std().item()
                        logits_max = logits_detached.max().item()
                        logits_min = logits_detached.min().item()
                        
                        logger.info(f"Logits stats - Mean: {logits_mean:.4f}, Std: {logits_std:.4f}, Min: {logits_min:.4f}, Max: {logits_max:.4f}")
                        
                        # Check loss calculation directly for verification
                        direct_loss = F.cross_entropy(logits_detached.view(-1, logits_detached.size(-1)), y.view(-1))
                        logger.info(f"Direct loss: {direct_loss.item():.4f} vs Unscaled model loss: {unscaled_loss:.4f}")
                        
                        # Track additional metrics - theoretical min loss and accuracy
                        theoretical_min_loss = math.log(vocab_size)
                        logger.info(f"Theoretical min loss: {theoretical_min_loss:.4f} (random guessing: {theoretical_min_loss})")
                        
                        # Calculate per-character accuracy
                        predictions = torch.argmax(logits_detached, dim=-1)
                        accuracy = (predictions == y).float().mean().item()
                        logger.info(f"Batch accuracy: {accuracy:.2%}")
                
                # Track memory usage periodically
                if batch_idx % 100 == 0 and device.type == 'cuda':
                    memory_stats = get_gpu_memory_usage()
                    logger.info(f"GPU Memory - Allocated: {memory_stats['allocated']:.1f} MB, "
                                f"Cached: {memory_stats['cached']:.1f} MB, "
                                f"Max: {memory_stats['max_allocated']:.1f} MB / {memory_stats['total']:.1f} MB "
                                f"({memory_stats['max_allocated']/memory_stats['total']*100:.1f}%)")
                    
                writer.add_scalar('system/gpu_memory_used_mb', memory_stats['allocated'], global_step)
                writer.add_scalar('system/gpu_memory_percent', 
                                 memory_stats['allocated']/memory_stats['total']*100, 
                                 global_step)
                
                # Add more comprehensive OOM protection, with memory cleanup
                if batch_idx % 500 == 0 and device.type == 'cuda':
                    # Force garbage collection and clear cache periodically
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info("Performed memory cleanup - GC collect and CUDA cache emptied")
                
                # Update counters
                global_step += 1
                batch_count += 1
                epoch_loss += unscaled_loss  # Track unscaled loss for reporting
                tokens_in_batch = x.numel()
                tokens_processed += tokens_in_batch
                
                # Calculate throughput
                batch_time = time.time() - batch_start
                running_batch_times.append(batch_time)
                if len(running_batch_times) > max_running_times:
                    running_batch_times.pop(0)
                
                tokens_per_second = tokens_in_batch / batch_time
                tokens_per_second_window.append(tokens_per_second)
                if len(tokens_per_second_window) > window_size:
                    tokens_per_second_window.pop(0)
                avg_tokens_per_second = sum(tokens_per_second_window) / len(tokens_per_second_window)
                
                # Simple but reliable ETA calculation based on average tokens/second
                batches_remaining = len(train_loader) - (batch_idx + 1)
                
                # Only calculate after a reasonable number of batches
                if len(tokens_per_second_window) >= 10:
                    # Simple average of recent token throughput
                    avg_token_rate = sum(tokens_per_second_window) / len(tokens_per_second_window)
                    
                    # Calculate remaining time
                    tokens_remaining = batches_remaining * batch_size * max_seq_length
                    seconds_remaining = tokens_remaining / avg_token_rate if avg_token_rate > 0 else 0
                    
                    # Convert to hours:minutes:seconds format
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(seconds_remaining))
                else:
                    eta_str = "Calculating..."
                
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    # Use unscaled loss for reporting to see true progress
                    logger.info(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {unscaled_loss:.4f} - {avg_tokens_per_second:.1f} tokens/s - ETA: {eta_str}")
                    writer.add_scalar('training/loss', unscaled_loss, global_step)
                    writer.add_scalar('training/loss_scaled', unscaled_loss.item(), global_step)
                    writer.add_scalar('training/throughput', avg_tokens_per_second, global_step)
                
                # Get current time for checkpoint and sampling decisions
                current_time = time.time()
                
                # Save checkpoint periodically based on time interval
                elapsed_since_last_checkpoint = current_time - last_checkpoint_time
                if elapsed_since_last_checkpoint >= (checkpoint_interval_minutes * 60):
                    last_checkpoint_time = current_time
                    # Save time-based checkpoint
                    checkpoint_path = os.path.join("outputs/models", f"{config_name}_{timestamp}_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': unscaled_loss,
                        'config': config,
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                    
                    # Cleanup old checkpoints, keeping only the 5 most recent ones
                    max_checkpoints = config.get('training', {}).get('max_checkpoints_to_keep', 5)
                    cleanup_old_checkpoints("outputs/models", config_name, timestamp, max_to_keep=max_checkpoints)
                
                # Generate sample text periodically (based ONLY on time interval, not step count)
                elapsed_since_last_sample = current_time - last_sample_time
                
                # Check if it's time to generate a sample (only by time, not by step count)
                if elapsed_since_last_sample >= (sample_interval_minutes * 60):
                    last_sample_time = current_time
                    
                    model.eval()
                    
                    # Choose a random context from the validation set
                    val_idx = random.randint(0, len(val_dataset) - 1)
                    context, _ = val_dataset[val_idx]
                    
                    # Instead of arbitrary position, find a proper line beginning
                    seed_context = dataset.decode(context.tolist())
                    
                    # Find line beginnings (either start of text or after newline)
                    line_starts = [0]  # Always include the start of text
                    for i in range(1, len(seed_context)):
                        if seed_context[i-1] == '\n':
                            line_starts.append(i)
                    
                    if line_starts and len(line_starts) > 1:
                        # Choose a random line start that gives enough context
                        valid_starts = [p for p in line_starts if p < len(seed_context) - 100]
                        if valid_starts:
                            start_pos = random.choice(valid_starts)
                            max_context_len = 100  # Increased context length for better seeds
                            end_pos = min(start_pos + max_context_len, len(seed_context))
                            seed_text = seed_context[start_pos:end_pos]
                        else:
                            # Fallback if no valid line starts
                            start_pos = line_starts[0]  # Use beginning of text
                            end_pos = min(start_pos + 100, len(seed_context))
                            seed_text = seed_context[start_pos:end_pos]
                        
                        # Convert back to token ids
                        context = torch.tensor([dataset.char_to_idx.get(c, 0) for c in seed_text], 
                                              dtype=torch.long).unsqueeze(0).to(device)
                    else:
                        # Fallback to first 100 tokens as before but with longer context
                        context_length = min(100, context.size(0))
                        context = context[:context_length].unsqueeze(0).to(device)
                        seed_text = dataset.decode(context[0].tolist())
                    
                    # Generate sample text with improved sampling
                    with torch.no_grad():  # Explicitly use no_grad
                        sample = generate_sample_text(
                            model=model,
                            context=context,
                            max_new_tokens=250,  # Increased sample length for better output
                            temperature=0.9,  # Higher temperature for early training
                            top_p=0.95,  # Higher top_p for more diversity 
                            top_k=40,    # Add top-k filtering
                            repetition_penalty=1.03,  # Light repetition penalty
                            tokenizer=dataset
                        )
                    
                    # Get only the newly generated part
                    generated_text = sample[len(seed_text):]
                    
                    # Calculate time spent training in a readable format
                    elapsed_seconds = time.time() - start_time
                    if elapsed_seconds < 3600:
                        # Less than an hour - show minutes
                        time_str = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60)}s"
                    elif elapsed_seconds < 86400:
                        # Less than a day - show hours and minutes
                        hours = int(elapsed_seconds // 3600)
                        mins = int((elapsed_seconds % 3600) // 60)
                        time_str = f"{hours}h {mins}m"
                    else:
                        # More than a day - show days, hours and minutes
                        days = int(elapsed_seconds // 86400)
                        hours = int((elapsed_seconds % 86400) // 3600)
                        mins = int((elapsed_seconds % 3600) // 60)
                        time_str = f"{days}d {hours}h {mins}m"
                    
                    logger.info(f"Sample after {time_str} of training (Epoch {epoch+1}, Batch {batch_idx}):")
                    logger.info(f"Seed: {seed_text}")
                    logger.info(f"Generated: {generated_text}\n")
                    writer.add_text(f'samples/time_{time_str.replace(" ", "_")}', 
                                   f"Seed: {seed_text}\nGenerated: {generated_text}", 
                                   global_step)
                    
                    # Log additional model diagnostics - make this faster
                    with torch.no_grad():
                        # Get a single batch from validation set (reuse context used for generation to save time)
                        # This avoids slow random batch selection
                        val_x, val_y = context, torch.cat([context[:, 1:], context[:, -1:]], dim=1)
                        
                        # Compute all metrics in a single forward pass
                        val_logits, val_loss = model(val_x, val_y)
                        val_perplexity = torch.exp(val_loss).item()
                        
                        # Quick accuracy calculation 
                        predictions = torch.argmax(val_logits, dim=-1)
                        accuracy = (predictions == val_y).float().mean().item()
                        
                        # Very fast top-5 sampling (only check 20 random positions)
                        if val_logits.numel() > 0:  # Make sure we have some logits to work with
                            batch_size, seq_len = val_y.size()
                            sample_size = min(20, batch_size * seq_len)  # Reduced from 100 to 20 positions
                            
                            if sample_size > 0:  # Make sure we have something to sample
                                # Sample a small subset of positions for efficiency
                                flat_logits = val_logits.view(-1, val_logits.size(-1))
                                flat_targets = val_y.view(-1)
                                
                                if flat_logits.size(0) > 0:  # Make sure we have data to sample from
                                    # Get random indices efficiently
                                    indices = torch.randint(0, flat_logits.size(0), (sample_size,), device=device)
                                    
                                    # Get top-k predictions only for sampled positions
                                    _, topk_indices = flat_logits[indices].topk(5, dim=-1)
                                    topk_correct = torch.sum(topk_indices == flat_targets[indices].unsqueeze(-1)).item()
                                    topk_accuracy = topk_correct / sample_size
                                else:
                                    topk_accuracy = 0.0
                            else:
                                topk_accuracy = 0.0
                        else:
                            topk_accuracy = 0.0
                            
                        # Log just the essential metrics
                        logger.info(f"Validation loss: {val_loss.item():.4f}, Perplexity: {val_perplexity:.2f}")
                        logger.info(f"Character prediction accuracy: {accuracy:.2%}")
                        logger.info(f"Top-5 character prediction accuracy (sampled): {topk_accuracy:.2%}")

                        # Log to TensorBoard - only log what's needed
                        writer.add_scalar('validation/loss', val_loss.item(), global_step)
                        writer.add_scalar('validation/perplexity', val_perplexity, global_step) 
                        writer.add_scalar('validation/accuracy', accuracy, global_step)
                    
                    # Print summary of recent training progress
                    if len(previous_losses) >= 5:
                        recent_losses = previous_losses[-5:]
                        avg_recent_loss = sum(recent_losses) / len(recent_losses)
                        loss_trend = recent_losses[-1] - recent_losses[0]
                        logger.info(f"Recent training: avg_loss={avg_recent_loss:.4f}, trend={loss_trend:.4f} ({'decreasing' if loss_trend < 0 else 'increasing'})")
                    
                    model.train()
            
            # Log epoch summary
            epoch_avg_loss = epoch_loss / batch_count
            elapsed = time.time() - start_time
            tokens_per_second = tokens_processed / elapsed
            
            logger.info(f"Epoch {epoch+1}/{epochs} complete - Avg loss: {epoch_avg_loss:.4f}")
            logger.info(f"Training throughput: {tokens_per_second:.1f} tokens/s")
            
            writer.add_scalar('training/epoch_loss', epoch_avg_loss, epoch)
            writer.add_scalar('training/epoch_throughput', tokens_per_second, epoch)
            
            # Generate end-of-epoch sample
            model.eval()
            logger.info(f"\nGenerating end of epoch {epoch+1} sample:")
            
            # Get a random seed from validation set
            val_idx = random.randint(0, len(val_dataset) - 1)
            context, _ = val_dataset[val_idx]
            context_length = min(100, context.size(0))  # Use longer context for better samples
            context = context[:context_length].unsqueeze(0).to(device)
            
            # Generate sample with lower temperature for better quality
            seed_text = dataset.decode(context[0].tolist())
            sample = generate_sample_text(
                model=model,
                context=context,
                max_new_tokens=200,  # Generate longer samples at end of epoch
                temperature=0.7,     # Lower temperature for more coherent text
                tokenizer=dataset
            )
            
            # Get only the newly generated part
            generated_text = sample[len(seed_text):]
            
            logger.info(f"Seed: {seed_text}")
            logger.info(f"Generated: {generated_text}\n")
            writer.add_text(f'samples/epoch_{epoch+1}', f"Seed: {seed_text}\nGenerated: {generated_text}", epoch)
            
            model.train()
            
            # Save model checkpoint with new naming convention
            checkpoint_path = os.path.join("outputs/models", f"{config_name}_{timestamp}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_avg_loss,
                'global_step': global_step,
                'config': config,  # Save config with the model
            }, checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Save the final model with new naming convention
        final_model_path = os.path.join("outputs/models", f"{config_name}_{timestamp}_final.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_avg_loss,
            'global_step': global_step,
            'config': config,  # Save config with the model
        }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Generate final sample
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample = generate_sample_text(
            model=model,
            context=context,
            max_new_tokens=1000,
            temperature=0.8,
            tokenizer=dataset
        )
        logger.info(f"\nFinal sample:\n{sample}\n")
        writer.add_text('samples', f"Final sample:\n{sample}", global_step)
        
        writer.close()
        
        logger.info("Training complete!")
        logger.info(f"Total time: {time.time() - start_time:.2f} seconds")
        logger.info(f"Final throughput: {tokens_processed / (time.time() - start_time):.1f} tokens/s")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Error during training: {e}")
    finally:
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a character-level language model with PyTorch')
    parser.add_argument('config_file', type=str, help='Path to the config file')
    parser.add_argument('--device', type=str, default=None, help='Device to train on (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--log_interval', type=int, default=None, help='Log interval')
    parser.add_argument('--sample_every', type=int, default=None, help='Sample interval')
    parser.add_argument('--sample_length', type=int, default=None, help='Sample length')
    parser.add_argument('--sample_temperature', type=float, default=None, help='Sample temperature')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping value')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint file to resume from')
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    kwargs = {}
    if args.device is not None:
        kwargs['device_str'] = args.device
    if args.epochs is not None:
        kwargs['epochs'] = args.epochs
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    if args.log_interval is not None:
        kwargs['log_interval'] = args.log_interval
    if args.sample_every is not None:
        kwargs['sample_every'] = args.sample_every
    if args.sample_length is not None:
        kwargs['sample_length'] = args.sample_length
    if args.sample_temperature is not None:
        kwargs['sample_temperature'] = args.sample_temperature
    if args.grad_clip is not None:
        kwargs['grad_clip'] = args.grad_clip
    if args.resume_from is not None:
        kwargs['resume_from'] = args.resume_from
    
    train_with_samples(args.config_file, **kwargs) 