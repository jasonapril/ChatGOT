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
import datetime
import math
import types

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
from src.models.gpt_decoder import GPTDecoder, create_gpt_decoder
from src.utils.generation import generate_sample_text, sample_text
from src.utils.memory import get_memory_optimized_settings, preallocate_gpu_memory
from src.utils.metrics import calculate_tokens_per_second

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ %(levelname)s ] %(message)s',  # Change dash to square brackets
    datefmt='%Y-%m-%d %H:%M:%S',  # Remove millisecond precision
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_tensorboard(config):
    """Set up TensorBoard logging."""
    experiment_name = config.get('experiment_name', os.path.basename(config_path).replace('.yaml', ''))
    log_dir = os.path.join(config.get('logging', {}).get('log_dir', 'runs'), 
                          f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set up file handler for logging to a file
    log_file_path = os.path.join(log_dir, 'training.log')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logs will be saved to {log_dir}")
    return writer

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
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [ %(levelname)s ] %(message)s'
    )
    logger = logging.getLogger(__name__)

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
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512")
        
        # Empty cache before we start
        torch.cuda.empty_cache()
        
        # Implement memory tracking
        memory_stats = {}
        memory_stats['init'] = torch.cuda.memory_allocated() / (1024**2)
        logger.info(f"Initial CUDA memory usage: {memory_stats['init']:.2f} MB")
        
        # Set environment variables for better OOM handling
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logger.info("Set CUDA_LAUNCH_BLOCKING=1 for better error reporting")

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
        batch_size = 6  # Smaller batch sizes often help with initial learning
        logger.info(f"Reducing batch size from {batch_size_orig} to {batch_size} to improve learning")
        config['training']['batch_size'] = batch_size
    
    # Memory optimization parameters
    mixed_precision = config['training'].get('mixed_precision', False)
    gradient_checkpointing = config['training'].get('gradient_checkpointing', False)
    accumulate_grad_batches = config['training'].get('accumulate_grad_batches', 1)
    logger.info(f"Using gradient accumulation steps: {accumulate_grad_batches}")
    
    # Set up TensorBoard
    log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    # Create model based on model_type
    logger.info(f"Creating model with type: {model_type}")
    if model_type == "gpt_decoder":
        # Import the GPT decoder model if needed
        from src.models.gpt_decoder import create_gpt_decoder
        
        # Adjust initialization to start closer to random prediction
        # Use smaller init range for more conservative initial predictions
        init_range = arch_config.get('init_range', 0.02)
        init_range_adjusted = min(init_range, 0.01)  # Cap to avoid too large initial weights
        arch_config['init_range'] = init_range_adjusted
        
        model = create_gpt_decoder(
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
        logger.warning("Gradient checkpointing is DISABLED due to compatibility issues")
        # Skip the actual checkpoint enabling code
    else:
        if gradient_checkpointing:
            logger.warning("Gradient checkpointing requested but disabled due to compatibility issues")
            
        if device.type != 'cuda':
            logger.info("Gradient checkpointing only supported on CUDA devices")
        elif torch_checkpoint is None:
            logger.info("torch.utils.checkpoint not available in this PyTorch version")

    # Try to free up memory before training starts
    torch.cuda.empty_cache()
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
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
    
    # GradScaler for mixed precision training
    scaler = GradScaler(enabled=mixed_precision) if mixed_precision else None
    
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
    config_name = os.path.basename(config_file).replace('.yaml', '')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Checkpoint settings
    save_checkpoint_steps = 100  # Save checkpoints every 100 steps
    last_checkpoint_time = time.time()
    checkpoint_interval_minutes = 2  # Save checkpoint every 2 minutes
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Model configuration: {arch_config}")
    logger.info(f"Training with batch size: {batch_size}, learning rate: {learning_rate}")
    logger.info(f"Checkpoints will be saved every {checkpoint_interval_minutes} minutes and every {save_checkpoint_steps} steps to: models/{config_name}_{timestamp}_step_X.pt")
    
    # Ensure all parameters have requires_grad=True
    for param in model.parameters():
        param.requires_grad = True
    
    # Verify parameter count and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    
    # Time-based sampling settings
    sample_interval_minutes = 5  # Generate sample every 5 minutes
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
            
            for batch_idx, (x, y) in enumerate(train_loader):
                batch_start = time.time()
                
                # Move data to device
                x, y = x.to(device), y.to(device)
                
                # Zero gradients
            optimizer.zero_grad()
            
                # Get accumulation steps
                is_accumulating = (batch_idx % accumulate_grad_batches != 0)
                
                # Forward pass with optional mixed precision
                if mixed_precision:
                    with autocast(device_type=device.type, dtype=torch.float16):
                        try:
                            logits, loss = model(x, y)
                            # Keep original loss for logging/monitoring
                            unscaled_loss = loss.item()
                            # Scale loss only for backprop
                            if accumulate_grad_batches > 1:
                                loss = loss / accumulate_grad_batches
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                logger.error("CUDA OOM during forward pass")
                                # Emergency memory cleanup
                                del logits, loss
                                torch.cuda.empty_cache()
                                # Try to reduce batch size
                                if batch_size > 1:
                                    new_batch_size = max(1, batch_size // 2)
                                    logger.info(f"Reducing batch size from {batch_size} to {new_batch_size}")
                                    batch_size = new_batch_size
                                    # Recreate dataloaders with new batch size
                                    # Continue with next epoch
                                    break
                            raise  # Re-raise the exception if it's not an OOM error
                else:
                    logits, loss = model(x, y)
                    # Keep original loss for logging/monitoring
                    unscaled_loss = loss.item()
                    # Scale loss only for backprop
                    if accumulate_grad_batches > 1:
                        loss = loss / accumulate_grad_batches
                
                # Check for NaN or inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Loss is {loss.item()}, stopping training")
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
                
                # Backward pass with optional mixed precision
                if mixed_precision:
                scaler.scale(loss).backward()
                    
                    # Only update weights on non-accumulation steps
                    if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1 == len(train_loader)):
                        scaler.unscale_(optimizer)
                        
                        # Diagnostic: check gradients
                        if batch_idx % 100 == 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                            logger.info(f"Gradient norm before clipping: {grad_norm:.4f}")
                            
                            # Check for parameters without gradients
                            params_without_grad = sum(1 for p in model.parameters() if p.grad is None)
                            if params_without_grad > 0:
                                logger.warning(f"{params_without_grad} parameters have no gradients!")
                            
                            # Sample some gradient values
                            grad_sample = []
                            for name, param in model.named_parameters():
                                if param.grad is not None:
                                    grad_sample.append((name, param.grad.abs().mean().item()))
                                    if len(grad_sample) >= 5:
                                        break
                            logger.info(f"Gradient samples: {grad_sample}")
                        
                        # Apply actual gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                    
                    # Only update weights on non-accumulation steps
                    if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1 == len(train_loader)):
                        # Diagnostic: check gradients
                        if batch_idx % 100 == 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                            logger.info(f"Gradient norm before clipping: {grad_norm:.4f}")
                            
                            # Check for parameters without gradients
                            params_without_grad = sum(1 for p in model.parameters() if p.grad is None)
                            if params_without_grad > 0:
                                logger.warning(f"{params_without_grad} parameters have no gradients!")
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                        optimizer.zero_grad()
                
                # Step the learning rate scheduler (after warmup period)
                if global_step >= warmup_steps:
                    scheduler.step()
                else:
                    # Linear warmup
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate * (global_step / warmup_steps)
                
                # Log current learning rate periodically
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('training/learning_rate', current_lr, global_step)
                
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
                    writer.add_scalar('training/loss_scaled', loss.item(), global_step)
                    writer.add_scalar('training/throughput', avg_tokens_per_second, global_step)
                
                # Generate sample text periodically (based on time interval)
                current_time = time.time()
                elapsed_since_last_sample = current_time - last_sample_time
                
                # Save checkpoint periodically based on time interval
                elapsed_since_last_checkpoint = current_time - last_checkpoint_time
                if elapsed_since_last_checkpoint >= (checkpoint_interval_minutes * 60) or (global_step % save_checkpoint_steps == 0 and global_step > 0):
                    last_checkpoint_time = current_time
                    # Save time-based checkpoint
                    checkpoint_path = os.path.join("models", f"{config_name}_{timestamp}_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': unscaled_loss,
                        'config': config,
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                
                if elapsed_since_last_sample >= (sample_interval_minutes * 60):
                    last_sample_time = current_time
                    
                    model.eval()
                    
                    # Choose a random context from the validation set
                    val_idx = random.randint(0, len(val_dataset) - 1)
                    context, _ = val_dataset[val_idx]
                    
                    # Instead of arbitrary position, find a proper word boundary
                    seed_context = dataset.decode(context.tolist())
                    # Find a period, question mark, or exclamation followed by space to ensure we start at sentence boundary
                    sentence_breaks = [i+2 for i, c in enumerate(seed_context[:-2]) 
                                      if c in ['.', '!', '?'] and seed_context[i+1] == ' ' and seed_context[i+2].isupper()]
                    
                    if sentence_breaks and len(sentence_breaks) > 1:
                        # Find a break point that gives us at least 30 chars but not too many
                        valid_breaks = [p for p in sentence_breaks if p < len(seed_context) - 100]
                        if valid_breaks:
                            start_pos = max(valid_breaks)
                            max_context_len = 80  # Arbitrary but reasonable length for seed
                            end_pos = min(start_pos + max_context_len, len(seed_context))
                            seed_text = seed_context[start_pos:end_pos]
                        else:
                            # Fallback if no valid breakpoints
                            start_pos = max(0, len(seed_context) - 100)
                            end_pos = len(seed_context)
                            seed_text = seed_context[start_pos:end_pos]
                        
                        # Convert back to token ids
                        context = torch.tensor([dataset.char_to_idx.get(c, 0) for c in seed_text], 
                                              dtype=torch.long).unsqueeze(0).to(device)
                    else:
                        # Fallback to first 50 tokens as before
                        context_length = min(50, context.size(0))
                        context = context[:context_length].unsqueeze(0).to(device)
                        seed_text = dataset.decode(context[0].tolist())
                    
                    # Generate sample text with improved sampling
                    with torch.no_grad():  # Explicitly use no_grad
                        sample = generate_sample_text(
                            model=model,
                            context=context,
                            max_new_tokens=sample_length,
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
                    
                    logger.info(f"\nSample after {time_str} of training (Epoch {epoch+1}, Batch {batch_idx}):")
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
            checkpoint_path = os.path.join("models", f"{config_name}_{timestamp}_epoch_{epoch+1}.pt")
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
        final_model_path = os.path.join("models", f"{config_name}_{timestamp}_final.pt")
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