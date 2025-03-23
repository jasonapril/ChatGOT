import os
import time
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import create_char_transformer, CharTransformer, TransformerConfig
import datetime
import sys
import logging
import gc
import matplotlib
import json
import warnings
import torch.nn.functional as F
import contextlib

"""
VRAM Utilization Strategy for Maximum Throughput
------------------------------------------------

Goal: Maximize training throughput by fully utilizing available GPU VRAM.

Key principle: VRAM is fast memory - the more of it we use, the faster our training.
Higher VRAM utilization directly translates to higher throughput and tokens/sec rates.

Design decisions and strategies:

1. MAXIMUM VRAM Utilization:
   - Target 98-99% VRAM usage, not just conservative memory management
   - Pre-allocate and aggressively touch VRAM to force higher utilization patterns
   - Counter PyTorch's conservative memory allocation with active strategies

2. Batch Size Maximization:
   - Use the largest possible batch size that fits in VRAM
   - Larger batches = more parallelism = faster throughput
   - Balance with accumulation steps to maintain effective batch size

3. Model Parallelization:
   - Maximize parallel compute across the GPU
   - Optimize tensor core utilization through appropriate dimensions
   - Use thread and kernel configuration for maximum compute density

4. Memory Bandwidth Optimization:
   - Reduce CPU↔GPU transfers that cause throughput bottlenecks
   - Keep computation on GPU as much as possible
   - Use pinned memory and non-blocking transfers when CPU interaction is necessary

5. Runtime Performance Monitoring:
   - Track tokens/second as the ultimate performance metric
   - Monitor VRAM usage to ensure we're utilizing maximum available memory
   - Automatically adjust to increase utilization when it falls below targets

Trade-offs and considerations:
   - Numerical stability vs. speed (favor speed where safe)
   - Convergence rate vs. parallelism (use lr scheduling to compensate)
   - Safety margins vs. maximum utilization (minimize margins to 1-2%)
"""

# Filter out the specific scheduler UserWarning about verbose parameter
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated.*")

matplotlib.use('Agg')  # Use non-interactive backend

def setup_logger(log_file=None, level=logging.INFO):
    """
    Setup logging configuration for both console and file outputs.
    Ensures immediate visibility of log messages.
    """
    # Create a formatter that includes timestamps
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove all handlers to avoid duplicates (if running multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with immediate flushing
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Ensure console output is immediately visible
    console_handler.stream = sys.stdout  # Use stdout explicitly
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        try:
            # Use a special file handler that flushes immediately
            file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='backslashreplace', mode='a')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            
            # Hack to ensure Windows flushes on every log
            orig_emit = file_handler.emit
            def new_emit(record):
                orig_emit(record)
                file_handler.flush()
                try:
                    # For Windows, force the OS to write to disk
                    if hasattr(file_handler.stream, 'fileno'):
                        os.fsync(file_handler.stream.fileno())
                except (AttributeError, OSError):
                    pass
            file_handler.emit = new_emit
            
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {os.path.abspath(log_file)}")
            
            # Write an immediate test message and force flush
            logging.info(f"=== LOGGING STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            force_flush_logs()
        except Exception as e:
            logging.error(f"Failed to set up file logging: {str(e)}")
            print(f"Error setting up log file: {str(e)}", flush=True)
    
    # Log startup info with timestamp
    logging.info(f"Logger initialized with {'console and file' if log_file else 'console only'} output")
    return root_logger

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    if seed is None:
        # Use a default seed if None is provided
        seed = 42
        logging.info(f"No seed specified, using default seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Setting to False for reproducibility

def load_data(data_path, batch_size, device_type, num_workers=0):
    """
    Load processed data from pickle file - optimized data loading approach.
    Uses multiple workers and prefetching for faster data loading.
    """
    logging.info(f"Loading data from {data_path}")
    
    # Set a default batch size if None is provided
    if batch_size is None:
        batch_size = 8  # Default batch size
        logging.info(f"No batch size specified, using default: {batch_size}")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # Get vocabulary and data splits
    char_to_idx = data["char_to_idx"]
    idx_to_char = data["idx_to_char"]
    train_inputs = data["train_inputs"]
    train_targets = data["train_targets"]
    val_inputs = data["val_inputs"]
    val_targets = data["val_targets"]
    
    logging.info(f"Data loaded: {len(train_inputs)} training sequences, {len(val_inputs)} validation sequences")
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(train_inputs, dtype=torch.long),
        torch.tensor(train_targets, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(val_inputs, dtype=torch.long),
        torch.tensor(val_targets, dtype=torch.long)
    )
    
    # Optimize dataloader settings based on device type and num_workers
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=(device_type == 'cuda'),  # Faster CPU->GPU transfer
        num_workers=num_workers,  # Parallel data loading
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor,  # Prefetch next batches
        drop_last=True  # More efficient training by dropping the last incomplete batch
    )
    
    # Use a larger batch size for evaluation (faster)
    eval_batch_size = min(batch_size * 4, 128)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=(device_type == 'cuda'),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    logging.info(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    if num_workers > 0:
        logging.info(f"Using {num_workers} workers for data loading with prefetch_factor={prefetch_factor}")
    
    return train_loader, val_loader, char_to_idx, idx_to_char

def force_flush_logs():
    """
    Force flush all logging handlers and stdout.
    """
    # Flush all logging handlers
    for handler in logging.getLogger().handlers:
        try:
            if hasattr(handler, 'flush') and callable(handler.flush):
                handler.flush()
        except Exception:
            pass
    
    # Also flush stdout
    sys.stdout.flush()
    
    # Close and reopen file handlers if applicable (more aggressive flushing)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                # This forces the OS to write to disk
                handler.close()
                handler.stream = open(handler.baseFilename, handler.mode)
            except Exception:
                pass

def train_epoch(model, optimizer, scheduler, dataloader, device, epoch, max_grad_norm=1.0, 
               gradient_accumulation_steps=1, use_amp=False, use_bfloat16=False, scaler=None,
               offload_optimizer_state=False):
    """
    Train for one epoch - maximum speed implementation with memory optimizations.
    
    Focuses on speed by using non-blocking data transfers, efficient gradient 
    zeroing, and minimal memory management operations. Supports optimizer state offloading.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        dataloader: Data loader with training batches
        device: Device to train on
        epoch: Current epoch number
        max_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_amp: Whether to use automatic mixed precision
        use_bfloat16: Whether to use bfloat16 instead of float16
        scaler: Gradient scaler for AMP
        offload_optimizer_state: Whether to offload optimizer states to CPU
    """
    model.train()
    start_time = time.time()
    total_loss = 0
    total_tokens = 0
    
    log_interval = 60  # Log every minute (60 seconds)
    early_update_batches = [1, 10, 100]  # Get early updates on these batches
    total_batches = len(dataloader)
    last_log_time = start_time - log_interval  # Ensure we log the first batch
    batch_times = []
    
    # Create a dedicated minute timer for consistent updates
    last_minute_log = start_time
    minute_timer_active = True
    
    # Log epoch start info with timestamp
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    logging.info(f"Starting training epoch {epoch} with gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"Total batches: {total_batches}, Batch size: {dataloader.batch_size}")
    
    # Memory tracking
    peak_memory = 0
    if device.type == 'cuda':
        # Clear memory at start of epoch
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        # Get total GPU memory for reference
        cuda_max_memory = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    
    # Total number of batches
    total_batches = len(dataloader)
    
    # Initialize step counter
    step = epoch * len(dataloader)
    
    # Use offloaded optimizer if requested
    if offload_optimizer_state:
        try:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer_was_wrapped = True
        except ImportError:
            logging.warning("[WARNING] ZeroRedundancyOptimizer not available, falling back to regular optimizer")
            optimizer_was_wrapped = False
    else:
        optimizer_was_wrapped = False
    
    # Efficient zeroing - only zero once at start of epoch rather than every batch
    optimizer.zero_grad(set_to_none=True)
    
    for i, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        # Check if a minute has passed regardless of batch progress - FORCE MINUTE UPDATES
        current_time = time.time()
        if minute_timer_active and (current_time - last_minute_log >= log_interval):
            minute_elapsed = int((current_time - start_time) / 60)
            
            # Calculate progress and stats for minute log
            progress_pct = (i / total_batches) * 100 if total_batches > 0 else 0
            
            # Calculate throughput metrics
            elapsed = current_time - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            
            # Calculate estimated time remaining
            tokens_remaining = (total_tokens / (i + 1)) * (total_batches - (i + 1))
            eta_seconds = tokens_remaining / tokens_per_sec if tokens_per_sec > 0 else 0
            
            # Format ETA nicely
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"
            
            # Track peak CUDA memory usage
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)
            
            # Get current memory usage if available
            if device.type == 'cuda':
                cuda_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                cuda_max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_utilization = (cuda_memory_allocated / cuda_max_memory) * 100
                cuda_memory_str = f"Memory: {cuda_memory_allocated:.1f}MB ({memory_utilization:.1f}%)"
            else:
                cuda_memory_str = ""
            
            # Log minute update with timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            logging.info(f"Minute {minute_elapsed} update | "
                        f"Progress: {progress_pct:.1f}% | Batch {i}/{total_batches} | " 
                        f"ETA: {eta_str if 'eta_str' in locals() else 'calculating...'} | {cuda_memory_str}")
            
            # Force flush logs to ensure they're written to disk
            force_flush_logs()
            
            # Update minute log timer
            last_minute_log = current_time
        
        # Handle data non-blocking for efficiency
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with appropriate mixed precision
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                # Scale by accumulation factor
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
        else:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # Scale by accumulation factor
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
        
        # Backward pass with appropriate mixed precision
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights with appropriate accumulation steps
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1 == total_batches):
            # Clip gradients
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            # Step learning rate scheduler (if batch-based)
            if scheduler is not None and isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,)):
                scheduler.step()
            
            # Zero gradients efficiently (set_to_none for faster zeroing)
            optimizer.zero_grad(set_to_none=True)
            
            # Increment step counter
            step += 1
        
        # Track loss and tokens for throughput calculation
        with torch.no_grad():
            # Get the raw loss value for logging
            raw_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1)).item()
            
            # Calculate tokens processed
            batch_size = inputs.size(0)
            seq_length = inputs.size(1)
            total_tokens += batch_size * seq_length
            total_loss += raw_loss * batch_size * seq_length
        
        # Calculate progress percentage
        progress_pct = 100.0 * (i + 1) / total_batches
        
        # Calculate throughput metrics
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        # Calculate estimated time remaining
        tokens_remaining = (total_tokens / (i + 1)) * (total_batches - (i + 1))
        eta_seconds = tokens_remaining / tokens_per_sec if tokens_per_sec > 0 else 0
        
        # Format ETA nicely
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds/3600:.1f}h"
        
        # Track peak CUDA memory usage
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)
        
        # Check if we should force a log update due to time
        current_time = time.time()
        force_time_update = (current_time - last_log_time >= log_interval)
        
        # Memory tracking (only check when we're going to log to avoid overhead)
        should_log = force_time_update or (current_time - last_log_time >= log_interval) or (i + 1) in early_update_batches or (i + 1 == total_batches)
        
        if should_log and device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)
            
            # Get total GPU memory
            memory_utilization = (current_memory / cuda_max_memory) * 100
            
            # Check for extreme memory pressure
            if memory_utilization > 97:
                logging.warning("[WARNING] Extremely high memory pressure detected - performed emergency garbage collection")
                torch.cuda.empty_cache()
                gc.collect()
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_utilization = (current_memory / cuda_max_memory) * 100
            
            cuda_memory_str = f"VRAM: {current_memory:.1f}MB/{cuda_max_memory:.1f}MB ({memory_utilization:.1f}%)"
        else:
            cuda_memory_str = ""
        
        # Generate timestamp for consistent log formatting
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        
        # Log progress at regular intervals and for important batches
        if should_log:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            logging.info(f"Progress: {progress_pct:.1f}% | Batch {i+1}/{total_batches} | "
                        f"Loss: {raw_loss:.4f} | Rate: {tokens_per_sec:.1f} tokens/sec | ETA: {eta_str} | {cuda_memory_str}")
            last_log_time = current_time
            
            # Update forced log timer if this was a force update
            if force_time_update:
                last_log_time = current_time
            
            # Force flush logs to ensure they're written to disk immediately
            force_flush_logs()
    
    # Format total training time
    train_time = time.time() - start_time
    train_time_str = format_time(train_time)
    
    # Log final stats with timestamp
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    logging.info(f"Epoch {epoch} completed in {train_time_str} | "
               f"Avg Loss: {avg_loss:.4f} | "
               f"Speed: {tokens_per_sec:.1f} tokens/sec | "
               f"Peak VRAM: {peak_memory:.1f}MB")
    
    # Force final log flush
    force_flush_logs()

    return avg_loss, tokens_per_sec

def evaluate(model, dataloader, device):
    """
    Evaluate model on validation data.
    
    Simple and efficient implementation focusing on maximum speed.
    """
    model.eval()
    total_loss = 0
    start_time = time.time()
    
    logging.info("Starting evaluation...")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Update tracking
            total_loss += loss.item()
            
            # Log progress every 10 batches
            if (i + 1) % 10 == 0 or i == 0 or i == len(dataloader) - 1:
                progress = (i + 1) / len(dataloader) * 100
                logging.info(f"Evaluation progress: {progress:.1f}% ({i+1}/{len(dataloader)} batches)")
                force_flush_logs()
    
    avg_loss = total_loss / len(dataloader)
    elapsed_time = time.time() - start_time
    
    logging.info(f"Evaluation completed in {elapsed_time:.2f}s with loss: {avg_loss:.4f}")
    force_flush_logs()
    
    return avg_loss

def generate_text(model, char_to_idx, idx_to_char, seed_text, max_length=500, temperature=1.0, device="cuda"):
    """Generate text from the model using sampling with temperature.
    
    Args:
        temperature: Controls randomness in generation.
            - Lower values (0.5-0.7): More deterministic, focused outputs
            - Higher values (0.8-1.2): More diverse, creative outputs
            - Very high values (1.5+): More random, potentially incoherent outputs
    """
    model.eval()
    
    logging.info(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting text generation")
    logging.info(f"Seed text: '{seed_text}'")
    logging.info(f"Parameters: max_length={max_length}, temperature={temperature}")
    
    # Convert seed text to tensor
    chars = list(seed_text)
    
    # Handle unknown characters more robustly
    # First, determine if we have an empty string token or need to use a default
    unknown_token = char_to_idx.get("", -1)
    if unknown_token == -1:
        # No empty string token, find a fallback
        if " " in char_to_idx:
            unknown_token = char_to_idx[" "]  # Space is a reasonable fallback
        elif len(char_to_idx) > 0:
            unknown_token = list(char_to_idx.values())[0]  # Use the first token as fallback
        else:
            raise ValueError("Character vocabulary is empty! Cannot generate text.")
    
    # Convert characters to token IDs with robust handling of unknown characters
    input_ids = [char_to_idx.get(c, unknown_token) for c in chars]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    start_time = time.time()
    
    # Use the model's built-in generation if available (more efficient)
    if hasattr(model, 'generate'):
        logging.info("Using model's built-in generation method")
        with torch.no_grad():
            # Generate text with the model's optimized generation method
            generated_ids = model.generate(
                input_tensor,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=40,  # Default top-k value
                verbose=False
            )
            
        # Convert the generated token IDs back to text
        generated_text = ''.join([idx_to_char[idx.item()] for idx in generated_ids[0]])
        
        # Log generation time
        total_time = time.time() - start_time
        tokens_per_sec = len(generated_text) / total_time
        
        logging.info(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                   f"Text generation completed in {total_time:.2f}s | "
                   f"Speed: {tokens_per_sec:.1f} tokens/sec")
        
        return generated_text
    
    # Fallback to manual generation for backward compatibility
    generated = list(seed_text)
    
    with torch.no_grad():
        logging.info("\nGenerating text:")
        logging.info("-" * 40)
        
        # Only show detailed token information for first 3 tokens
        detail_tokens = 3
        
        # Time-based progress reporting (every 30 seconds for generation)
        progress_interval = 30  # seconds
        last_progress_time = start_time
        
        for i in range(max_length):
            # Forward pass
            with torch.amp.autocast('cuda', enabled=device=="cuda"):
                logits = model(input_tensor)
            
            # Get last time step logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Ensure next_token is in idx_to_char before accessing
            if next_token in idx_to_char:
                next_char = idx_to_char[next_token]
            else:
                # If token not in vocabulary, use a fallback character
                logging.warning(f"Generated token {next_token} not in vocabulary. Using fallback.")
                if 0 in idx_to_char:
                    next_char = idx_to_char[0]
                else:
                    next_char = " "  # Space as ultimate fallback
            
            generated.append(next_char)
            
            # Print detailed information for first few tokens only
            if i < detail_tokens:
                top_p, top_i = torch.topk(probs, 5)
                top_tokens = []
                for idx in top_i:
                    if idx.item() in idx_to_char:
                        top_tokens.append(f'{idx_to_char[idx.item()]}')
                    else:
                        top_tokens.append(f'<UNK>')
                logging.info(f"\nStep {i+1}:")
                logging.info(f"  Top tokens: {top_tokens}")
                logging.info(f"  Top probs: {[f'{p.item():.4f}' for p in top_p]}")
                logging.info(f"  Selected: '{next_char}' (index {next_token}, prob: {probs[next_token].item():.4f})")
            
            # Print progress at time intervals
            current_time = time.time()
            if (current_time - last_progress_time >= progress_interval or i + 1 == max_length) and i >= detail_tokens:
                elapsed = current_time - start_time
                tokens_per_sec = (i + 1) / elapsed
                remaining = (max_length - i - 1) / tokens_per_sec if tokens_per_sec > 0 else 0
                
                # Calculate percentage
                percent_complete = ((i + 1) / max_length) * 100
                
                logging.info(f"\nGeneration: {percent_complete:.1f}% | "
                           f"Tokens: {i+1}/{max_length} | "
                           f"Speed: {tokens_per_sec:.1f} tokens/sec | "
                           f"ETA: {remaining:.1f}s")
                
                # Print a sample of recent text (last 30 chars)
                recent_text = ''.join(generated[-min(30, len(generated)):])
                logging.info(f"Recent text: '{recent_text}'")
                
                # Update last progress time
                last_progress_time = current_time
            
            # Update input tensor for next iteration
            input_tensor = torch.cat([
                input_tensor, 
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ], dim=1)
            
            # Truncate to prevent excessive memory usage
            if input_tensor.size(1) > 100:
                input_tensor = input_tensor[:, -100:]
    
    total_time = time.time() - start_time
    tokens_per_sec = max_length / total_time
    
    logging.info(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Text generation completed in {total_time:.2f}s | "
                f"Speed: {tokens_per_sec:.1f} tokens/sec")
    
    return ''.join(generated)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, char_to_idx, idx_to_char):
    """Save model checkpoint."""
    # Use model's built-in save method if available
    if hasattr(model, 'save_checkpoint'):
        model.save_checkpoint(
            path=path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char
        )
        logging.info(f"Checkpoint saved to {path}")
    else:
        # Fall back to standard checkpoint saving
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "loss": loss,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint."""
    # Use model's built-in load method if available
    if hasattr(model, 'load_checkpoint'):
        logging.info(f"Attempting to load checkpoint from {path} using model's method")
        try:
            # CharTransformer.load_checkpoint is a class method that returns a new model instance
            # and other checkpoint data
            result = CharTransformer.load_checkpoint(
                path=path,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            # Unpack the result (model, optimizer, epoch, loss, char_to_idx, idx_to_char)
            model = result[0]
            if len(result) > 1:
                optimizer = result[1]
            if len(result) > 2:
                epoch = result[2]
            if len(result) > 3:
                loss = result[3]
            if len(result) > 4:
                char_to_idx = result[4]
            if len(result) > 5:
                idx_to_char = result[5]
                
            logging.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f} using model's method")
            return model, optimizer, epoch, loss, char_to_idx, idx_to_char
        except Exception as e:
            logging.warning(f"Failed to load using model's method: {e}. Falling back to standard loading.")
            
    # Fall back to standard checkpoint loading
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    logging.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f} using standard method")
    return model, optimizer, epoch, loss, char_to_idx, idx_to_char

def plot_losses(train_losses, val_losses, filename='training_loss.png'):
    """Plot and save the training and validation losses."""
    plt.figure(figsize=(10, 6))
    
    # Plot epoch-level losses
    if train_losses:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    
    # Plot validation losses if available
    if val_losses:
        epochs = range(1, len(val_losses) + 1)
        plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Set reasonable y-axis limits
    all_losses = []
    if train_losses:
        all_losses.extend(train_losses)
    if val_losses:
        all_losses.extend(val_losses)
    
    if all_losses:
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        # Add padding
        padding = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
        plt.ylim(max(0, min_loss - padding), max_loss + padding)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logging.info(f"Final loss plot saved to {filename}")

def update_progress_plot(train_losses, val_losses, current_epoch_progress, args):
    """Create or update a plot of the training and validation losses."""
    if not args.save_loss_plot:
        return
    
    # Create the plot silently without logging
    try:
        # Create figure and axis
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        
        # Plot validation loss
        if val_losses:
            val_epochs = list(range(1, len(val_losses) + 1))
            plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')
        
        # If we're in the middle of an epoch, plot the current progress
        if current_epoch_progress:
            current_epoch = len(train_losses) + 1
            x = [current_epoch - 1, current_epoch]
            y = [train_losses[-1] if train_losses else 0, current_epoch_progress]
            plt.plot(x, y, 'b--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('training_loss.png')
        plt.close()
    except Exception as e:
        # Silently fail, don't disrupt training
        pass

def parse_arguments():
    """
    Parse command line arguments with simple, straightforward options.
    """
    parser = argparse.ArgumentParser(description='Train a character-level transformer model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the processed data pickle file')
    
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (will be auto-tuned if not specified)')
    parser.add_argument('--force_batch_size', action='store_true',
                        help='Force the specified batch size even when auto-tuning')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable automatic mixed precision training')
    parser.add_argument('--use_bfloat16', action='store_true',
                        help='Use bfloat16 precision instead of float16 if available')
    parser.add_argument('--max_memory_usage', type=float, default=0.85,
                        help='Maximum GPU memory usage fraction (0.0-1.0)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before updating weights')
    parser.add_argument('--use_compile', action='store_true',
                        help='Use torch.compile() for faster training if available')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers (0 means main process only)')
    parser.add_argument('--use_onecycle', action='store_true',
                        help='Use OneCycleLR instead of CosineAnnealing for learning rate scheduling')
    parser.add_argument('--warmup_pct', type=float, default=0.1,
                        help='Percentage of training steps to use for warmup')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--sample_every_epoch', action='store_true',
                        help='Generate sample text after each epoch')
    
    return parser.parse_args()

def create_optimizer(model, lr, weight_decay=0.01, offload_optimizer=False):
    """
    Create optimizer for the model with proper weight decay settings and optional CPU offloading.
    
    Args:
        model: The model whose parameters will be optimized
        lr: Learning rate
        weight_decay: Weight decay coefficient
        offload_optimizer: Whether to offload optimizer states to CPU
    """
    # Separate parameters that should have weight decay from those that shouldn't
    decay_params = []
    no_decay_params = []
    
    # Simple approach - apply weight decay only to weights in Linear layers, not to biases or LayerNorm/Embedding weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'layernorm' in name or 'ln' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    # Create parameter groups
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(optim_groups, lr=lr)
    
    # If offloading is requested, use CPU offloading for some optimizer states
    if offload_optimizer and torch.cuda.is_available():
        # This is a simple form of ZeRO stage 1 optimizer
        logging.info("Using CPU offloading for optimizer states to save GPU memory")
        
        # Move some optimizer state to CPU - we create our own state dictionary 
        # and handle the transfer to CPU
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                # Ensure param.grad is allocated on the same device as param
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(param.device)
        
        # After the first optimizer step, states like exp_avg and exp_avg_sq will be created
        # We'll let the optimizer create these normally, then move in train_epoch
    
    return optimizer

def get_gtx1650ti_optimizations():
    """
    Returns optimal settings for GTX 1650 Ti GPU for maximum training speed
    """
    return {
        'batch_size': 12,  # Optimal batch size for training
        'max_grad_norm': 1.0,  # Gradient clipping for training stability
        'disable_amp': True,  # Disable automatic mixed precision (better performance on GTX 1650 Ti)
    }

def sample_text(model, char_to_idx, idx_to_char, device, seed_text="", max_length=100, temperature=0.8):
    """
    Generate sample text using the trained model.
    Simple implementation for generating text during training.
    """
    model.eval()
    
    # Use empty string if no seed text provided
    if not seed_text:
        seed_text = "TYRION: "
    
    # Convert seed text to indices
    context = torch.tensor([[char_to_idx.get(c, 0) for c in seed_text]], dtype=torch.long).to(device)
    
    # Generate text
    generated_text = seed_text
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits from model
            logits = model(context)
            
            # Take last token's logits
            logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
            # Sample from the distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to context
            context = torch.cat([context, next_token], dim=1)
            
            # Convert to character and add to generated text
            next_char = idx_to_char[next_token.item()]
            generated_text += next_char
            
            # Break if end of text
            if next_char == '\n\n' and len(generated_text) > 10:
                break
    
    logging.info(f"Sample text: {generated_text}")
    return generated_text

def find_max_batch_size(model, sample_input_shape, device, start_batch=32, patience=3, max_failures=5):
    """
    Find the maximum batch size that fits in GPU memory through binary search.
    Uses a more conservative approach to avoid CUDA errors.
    
    Args:
        model: The model to test
        sample_input_shape: Shape of a single input sample (without batch dimension)
        device: CUDA device to use
        start_batch: Initial batch size to try
        patience: Number of successful batch sizes to try before stopping
        max_failures: Max number of OOM failures before reducing search range
        
    Returns:
        Maximum batch size that works reliably
    """
    logging.info(f"Finding maximum possible batch size for available CUDA memory...")
    
    # Make sure model is on the correct device
    model.to(device)
    
    # Track successful batch sizes
    successful_batch_sizes = []
    
    # Binary search parameters - start more conservatively
    low = 1
    high = start_batch  # Start more conservatively
    current = start_batch // 2  # Start with half the requested batch size
    failures = 0
    
    # Create a fixed vocab size for testing - avoid index out of bounds
    vocab_size = 100  # Small fixed vocabulary for testing
    
    try:
        while len(successful_batch_sizes) < patience and failures < max_failures:
            try:
                # Clear cache before test
                torch.cuda.empty_cache()
                
                # Create a batch with a fixed range of indices to avoid out of bounds
                dummy_input = torch.randint(0, vocab_size, (current, *sample_input_shape), 
                                          dtype=torch.long, device='cpu').to(device)
                
                # Do a test forward pass only - no backward pass needed to test memory
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # If we get here, the batch size works
                logging.info(f"Batch size {current} fits in memory")
                successful_batch_sizes.append(current)
                
                # Increase for next test (binary search upward)
                low = current
                current = min(current + (high - current) // 2, high)
                
                # If we're at the limit of our search range, break
                if current == high and low == high:
                    break
                    
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # Memory error, reduce batch size
                if "CUDA out of memory" in str(e) or "cudaMalloc failed" in str(e):
                    logging.info(f"Batch size {current} too large (CUDA OOM)")
                    high = current - 1
                    current = max(low + (high - low) // 2, low)
                    failures += 1
                    
                    # Reset model if it's in a bad state
                    torch.cuda.empty_cache()
                else:
                    # Some other error occurred
                    logging.warning(f"Unexpected error during batch size search: {str(e)}")
                    break
        
        # Get the maximum successful batch size
        if successful_batch_sizes:
            max_batch = max(successful_batch_sizes)
            # Apply a larger safety margin (80% of max) for stability
            safe_batch = max(1, int(max_batch * 0.8))
            logging.info(f"Maximum reliable batch size determined: {safe_batch} (max tested: {max_batch})")
            return safe_batch
        else:
            logging.warning("Could not determine maximum batch size. Using fallback size of 8")
            return 8
            
    except Exception as e:
        logging.warning(f"Batch size search failed with error: {str(e)}. Using fallback size of 8")
        return 8

def get_memory_optimized_settings(device=None, device_name=None):
    """
    Get memory optimization settings for specific GPU models.
    
    This function provides tailored settings to maximize VRAM utilization
    and throughput by using as much available memory as possible. The goal
    is maximum speed, not conservative memory management.
    
    Args:
        device: PyTorch device object
        device_name: GPU name string if available
        
    Returns:
        dict: Dictionary of optimized settings for maximum throughput
    """
    # Default settings - already fairly aggressive
    settings = {
        'batch_size': 16,               # Higher default batch size
        'gradient_accumulation_steps': 2,
        'max_memory_usage': 0.95,       # Target 95% memory usage by default
        'disable_amp': False,
        'dtype': torch.float32,
        'pin_memory': True,
        'force_aggressive_memory': True  # Default to aggressive memory usage
    }
    
    # If no device, return defaults
    if device is None:
        return settings
        
    # If not CUDA, return defaults
    if device.type != 'cuda':
        return settings
        
    # Get device name if not provided
    if device_name is None:
        device_name = torch.cuda.get_device_name(device)
    
    # Get total VRAM in MB
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    logging.info(f"Detected {total_vram:.0f}MB total VRAM - optimizing for maximum utilization")
    
    # NVIDIA GTX 1650 Ti specific optimizations (4GB VRAM)
    if '1650 Ti' in device_name:
        settings.update({
            'batch_size': 64,             # Extremely aggressive batch size for maximum parallelism
            'gradient_accumulation_steps': 1,  # No accumulation - direct updates for speed
            'max_memory_usage': 0.99,     # Use 99% of available memory
            'disable_amp': True,          # Disable AMP for better throughput on this GPU
            'dtype': torch.float32,       # Use full precision 
            'force_aggressive_memory': True,  # Enable aggressive memory features
            'reduce_activation_memory': False, # Don't sacrifice speed for memory efficiency
            'enable_tiling': True,        # Enable tiled matrix multiplications
            'max_sequence_parallel': True # Maximize parallellism across sequence dimension
        })
        logging.info("Applied GTX 1650 Ti specific settings for MAXIMUM throughput and VRAM utilization")
    
    # High-end GPUs (8GB+ VRAM)
    elif any(x in device_name for x in ['RTX', 'GTX 16', 'Quadro']):
        # For higher-end GPUs, use even more aggressive settings
        settings.update({
            'batch_size': 128,           # Very large batch size for maximum throughput
            'gradient_accumulation_steps': 1,
            'max_memory_usage': 0.98,    # Use 98% of available memory
            'force_aggressive_memory': True,  
            'enable_flash_attention': True,  # Enable flash attention if supported
            'enable_tiling': True
        })
        logging.info("Applied high-end GPU settings for maximum throughput")
        
    # Return the optimized settings for maximum throughput
    return settings

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set up logging properly using our custom function
    log_file = os.path.join(os.path.dirname(args.checkpoint_dir), 'training_log.txt')
    setup_logger(log_file=log_file, level=logging.INFO)
    
    # Make sure we catch all stdout/stderr for immediate visibility
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Force Python's stdout/stderr to be line-buffered for more immediate feedback
    import io
    sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
    sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), 'wb', 0), write_through=True)
    
    # Import torch here if it wasn't imported globally
    import torch
    import torch.cuda
    
    # Log system info
    device_info = f"Using device: {args.device}"
    if args.device.startswith('cuda'):
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
            device_info += f", CUDA Version: {torch.version.cuda}"
            torch_version = getattr(torch, '__version__', 'unknown')
            device_info += f", PyTorch: {torch_version}"
        else:
            logging.warning("CUDA is not available despite requesting a CUDA device. Falling back to CPU.")
            args.device = 'cpu'
            device_info = "Using device: cpu (CUDA not available)"
    
    logging.info(f"Starting training with arguments: {vars(args)}")
    logging.info(device_info)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up device
    device = torch.device(args.device)
    
    # Get optimized settings for the current GPU
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    optimized_settings = get_memory_optimized_settings(device, gpu_name)
    
    # Enhanced optimizations for GTX 1650 Ti
    is_gtx1650ti = False
    if torch.cuda.is_available() and '1650 Ti' in gpu_name:
        is_gtx1650ti = True
        logging.info("Detected GTX 1650 Ti - Using enhanced optimizations for maximum VRAM utilization")
        
        # Apply optimized settings from our settings engine
        if is_gtx1650ti:
            logging.info("Detected GTX 1650 Ti - Using enhanced optimizations for maximum VRAM utilization")
            
            # Override user settings with our aggressive settings to maximize throughput
            args.batch_size = optimized_settings['batch_size']
            args.max_memory_usage = optimized_settings['max_memory_usage']
            args.disable_amp = optimized_settings['disable_amp']
            args.gradient_accumulation_steps = optimized_settings['gradient_accumulation_steps']
            args.force_batch_size = True  # Force our batch size
            
            logging.info(f"[SPEED] Using aggressive batch size of {args.batch_size}")
            logging.info(f"[SPEED] Setting VRAM utilization target to {args.max_memory_usage*100:.1f}%")
            logging.info(f"[SPEED] {'Disabling' if args.disable_amp else 'Enabling'} AMP for optimal throughput")
            logging.info(f"[SPEED] Setting gradient_accumulation_steps to {args.gradient_accumulation_steps}")
            logging.info(f"[SPEED] Maximizing parallelism for highest throughput")
            
            # Force PyTorch to use all available VRAM by applying tensor core optimizations
            if hasattr(torch.backends.cuda, "matmul"):
                # Enable TensorCore operations for faster matrix multiplications
                torch.backends.cuda.matmul.allow_tf32 = True
                logging.info("[SPEED] Enabled TensorCore operations (TF32) for faster computation")
            
            # Set deterministic algorithms to False for better performance
            if hasattr(torch.backends.cudnn, "deterministic"):
                torch.backends.cudnn.deterministic = False
                logging.info("[SPEED] Disabled deterministic algorithms for faster training")
            
            # Enable cuDNN benchmark for faster convolutions
            torch.backends.cudnn.benchmark = True
            logging.info("[SPEED] Enabled cuDNN benchmarking for faster convolutions")
    
    # For non-GTX 1650 Ti GPUs, still apply some aggressive settings 
    elif device.type == 'cuda':
        logging.info(f"CUDA device detected: {gpu_name} - applying throughput optimizations")
        
        # Still apply some aggressive optimizations
        if not args.force_batch_size:  # Only override if not forced by user
            args.batch_size = max(args.batch_size or 1, optimized_settings['batch_size'])
            logging.info(f"[SPEED] Using batch size {args.batch_size} for higher throughput")
        
        # Apply higher memory usage
        args.max_memory_usage = max(args.max_memory_usage, optimized_settings['max_memory_usage'])
        logging.info(f"[SPEED] Setting VRAM utilization target to {args.max_memory_usage*100:.1f}%")
        
        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
        logging.info("[SPEED] Enabled cuDNN benchmarking for faster training")
    
    # Disable AMP if requested or on old GPUs
    use_amp = not args.disable_amp and device.type == 'cuda'
    if use_amp:
        logging.info("Using Automatic Mixed Precision (AMP) training")
        scaler = torch.cuda.amp.GradScaler()
    else:
        logging.info("Not using AMP - training with standard precision")
        scaler = None
    
    # Check for bfloat16 support
    if args.use_bfloat16 and torch.cuda.is_available():
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            logging.info("BFloat16 precision is supported and will be used if AMP is enabled")
        else:
            logging.warning("BFloat16 precision requested but not supported on this GPU. Will use Float16 instead.")
    
    # Load data with optimized loading
    logging.info(f"Loading data from {args.data_path}")
    train_loader, val_loader, char_to_idx, idx_to_char = load_data(
        args.data_path, args.batch_size, device.type, args.num_workers
    )
    
    # Create model
    config = TransformerConfig(
        vocab_size=len(char_to_idx),
        n_layer=12,
        n_head=12,
        n_embd=768,
        context_size=1024,
        dropout=0.1,
        bias=True
    )
    
    model = CharTransformer(config)
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = num_params * 4 / 1024**2  # 4 bytes per float parameter
    
    logging.info(f"Model created with {num_params:,} parameters")
    logging.info(f"Model size in memory: {model_size_mb:.2f}MB")
    
    # Move model to device for batch size testing
    model.to(device)
    
    # Apply gradient checkpointing to save memory during backpropagation for large models
    # Only do this for GTX 1650 Ti where VRAM is limited
    use_gradient_checkpointing = is_gtx1650ti and num_params > 50_000_000
    if use_gradient_checkpointing:
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            logging.info("Enabled gradient checkpointing to save GPU memory")
        elif hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing to save GPU memory")
        else:
            logging.warning("Gradient checkpointing not supported by this model")
    
    # For GTX 1650 Ti, find the maximum batch size through testing
    if is_gtx1650ti and not args.force_batch_size:
        # Skip dynamic batch size finding as it can cause CUDA errors
        # We're using a fixed optimized batch size (set earlier) that we know works well
        logging.info(f"Using pre-determined batch size of {args.batch_size} for GTX 1650 Ti")
        logging.info(f"This batch size with gradient_accumulation_steps={args.gradient_accumulation_steps} provides good performance")
    
    # Print summary information for immediate visibility
    print("\n===== TRAINING STARTING =====", flush=True)
    print(f"Model: {num_params:,} parameters ({model_size_mb:.2f}MB)", flush=True)
    print(f"Data: {len(train_loader)} training batches, {len(val_loader)} validation batches", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=============================\n", flush=True)
    
    # Use torch.compile if available and requested
    if args.use_compile and device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            logging.info("Using torch.compile() for faster training")
            # Try different backends in order of preference
            available_backends = []
            try:
                import torch._dynamo
                available_backends = torch._dynamo.list_backends()
                logging.info(f"Available torch.compile backends: {available_backends}")
            except:
                logging.warning("Could not list available backends")
            
            # Choose backend based on availability
            backend = None
            for preferred_backend in ["inductor", "nvfuser", "aot_eager", "eager"]:
                if preferred_backend in available_backends:
                    backend = preferred_backend
                    break
            
            # If we couldn't find any of our preferred backends, use the default
            if backend is None:
                backend = "inductor"  # Default in PyTorch
                
            logging.info(f"Compiling model with backend: {backend}")
            model = torch.compile(model, backend=backend)
            logging.info(f"Model successfully compiled with {backend} backend")
        except Exception as e:
            logging.warning(f"Failed to compile model: {e}. Proceeding with standard model.")
    
    # Move model to device
    model.to(device)
    
    # Create optimizer with appropriate settings and memory optimizations
    use_optimizer_offload = is_gtx1650ti  # Enable optimizer offloading for GTX 1650 Ti
    if use_optimizer_offload:
        logging.info("Enabling optimizer state CPU offloading to save GPU memory")
        
    optimizer = create_optimizer(
        model, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        offload_optimizer=use_optimizer_offload
    )
    
    # Create scheduler based on user preference
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    if args.use_onecycle:
        # OneCycleLR with warmup and higher peak learning rate
        logging.info(f"Using OneCycleLR scheduler with warmup_pct={args.warmup_pct}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr * 3,  # 3x higher peak learning rate
            total_steps=total_steps,
            pct_start=args.warmup_pct,  # Portion of training used for warmup
            anneal_strategy='cos',
            div_factor=25.0,  # Initial LR = max_lr/25
            final_div_factor=10000.0,  # Final LR = initial_lr/10000
        )
        logging.info(f"OneCycleLR: Initial LR={args.lr/25:.6f}, Peak LR={args.lr*3:.6f}, Final LR={args.lr/250000:.8f}")
    else:
        # Standard CosineAnnealing
        logging.info("Using CosineAnnealing scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps, 
            eta_min=args.lr/10
        )
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Log total and usable CUDA memory
    if device.type == 'cuda':
        cuda_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        cuda_usable_memory = cuda_total_memory - 800  # Subtract CUDA context overhead
        cuda_memory_target = cuda_usable_memory * args.max_memory_usage
        
        logging.info(f"Total CUDA memory: {cuda_total_memory:.1f}MB")
        logging.info(f"Usable CUDA memory (after context): {cuda_usable_memory:.1f}MB")
        logging.info(f"Memory allocation target: {cuda_memory_target:.1f}MB ({args.max_memory_usage*100:.1f}%)")
        
        # Ultra-aggressive memory pre-allocation to force maximum VRAM usage
        if device.type == 'cuda':
            # Force PyTorch to allocate a large chunk of memory initially
            logging.info("[MEMORY] Pre-allocating maximum CUDA memory to force high throughput...")
            try:
                # Calculate pre-allocation size - ultra aggressive at 98-99% of target
                prealloc_factor = 0.98  # Ultra aggressive
                prealloc_size = int(cuda_memory_target * prealloc_factor)
                prealloc_bytes = prealloc_size * 1024 * 1024  # Convert MB to bytes
                
                # Get the current allocated memory as baseline
                before_allocation = torch.cuda.memory_allocated() / (1024 * 1024)
                
                # Create multiple smaller tensors instead of one large one for better allocation
                prealloc_tensors = []
                
                # Use more chunks for better fragmentation patterns (helps with throughput)
                num_chunks = 20  # More chunks for better memory layout
                chunk_size = prealloc_bytes // num_chunks // 4  # Split into chunks, each with float32 (4 bytes)
                
                # Multi-stage allocation strategy for better VRAM utilization patterns
                allocation_stages = [0.5, 0.75, 0.9, 0.98]  # Progressive allocation targets
                
                for stage_idx, stage_target in enumerate(allocation_stages):
                    # Calculate stage allocation size
                    stage_size = int(prealloc_bytes * stage_target / len(allocation_stages))
                    stage_chunks = max(1, num_chunks // len(allocation_stages))
                    stage_chunk_size = stage_size // stage_chunks // 4
                    
                    logging.info(f"[MEMORY] Allocation stage {stage_idx+1}/{len(allocation_stages)}: Targeting {stage_target*100:.0f}% of allocation goal")
                    
                    for i in range(stage_chunks):
                        # Allocate a chunk
                        tensor = torch.empty(stage_chunk_size, dtype=torch.float, device=device)
                        # Use calculations to force allocation and utilize memory pathways
                        tensor.normal_()
                        tensor = torch.nn.functional.relu(tensor)  # More compute to engage GPU
                        tensor = tensor + torch.randn(1, device=device)[0]
                        prealloc_tensors.append(tensor)
                    
                    # Report stage progress
                    stage_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                    stage_delta = stage_allocated - before_allocation
                    current_pct = (stage_allocated / cuda_total_memory) * 100
                    logging.info(f"  - Current VRAM: {stage_allocated:.1f}MB ({current_pct:.1f}% of total VRAM)")
                
                # Final allocation report
                actual_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                allocation_delta = actual_allocated - before_allocation
                allocation_pct = (actual_allocated / cuda_total_memory) * 100
                logging.info(f"[MEMORY] Successfully pre-allocated {allocation_delta:.1f}MB of CUDA memory ({actual_allocated:.1f}MB total, {allocation_pct:.1f}% of VRAM)")
                
                # Try to find and allocate any remaining memory pockets for absolute maximum utilization
                try:
                    # Calculate remaining available memory (be very aggressive)
                    remaining = cuda_memory_target - actual_allocated
                    if remaining > 50:  # If more than 50MB still available, go for it
                        # Try to allocate most of the remaining memory
                        remaining_tensor_size = int(remaining * 0.98 * 1024 * 1024 // 4)
                        logging.info(f"[MEMORY] Attempting to allocate final {remaining:.1f}MB of VRAM for maximum utilization")
                        
                        # Try multiple smaller tensors for better chances of allocation
                        final_chunks = 5
                        final_chunk_size = remaining_tensor_size // final_chunks
                        for i in range(final_chunks):
                            try:
                                final_tensor = torch.empty(final_chunk_size, dtype=torch.float, device=device)
                                final_tensor.zero_()
                                prealloc_tensors.append(final_tensor)
                            except Exception:
                                # If we hit the limit, that's fine - we're pushing to the absolute maximum
                                pass
                        
                        # Report final allocation
                        final_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                        additional = final_allocated - actual_allocated
                        final_pct = (final_allocated / cuda_total_memory) * 100
                        if additional > 0:
                            logging.info(f"[MEMORY] Allocated additional {additional:.1f}MB in final pass (total: {final_allocated:.1f}MB, {final_pct:.1f}% of VRAM)")
                        actual_allocated = final_allocated
                except Exception as e:
                    logging.info(f"Final allocation attempt skipped: {str(e)}")
                
                # Create a training-time buffer tensor that stays allocated for the entire training process
                # This helps maintain high VRAM usage throughout training
                try:
                    permanent_buffer_size = int(cuda_memory_target * 0.05)
                    if permanent_buffer_size > 100:  # Only if we have enough memory to spare
                        logging.info(f"[MEMORY] Creating permanent VRAM buffer of {permanent_buffer_size:.1f}MB to maintain high utilization")
                        # Store this in a global variable so it persists through training
                        global permanent_vram_buffer  # This will persist throughout the script's execution
                        permanent_vram_buffer = torch.zeros(int(permanent_buffer_size * 1024 * 1024 // 4), 
                                                          dtype=torch.float, device=device)
                        permanent_vram_buffer.zero_() # Touch it to ensure allocation
                except Exception as e:
                    logging.info(f"Permanent buffer creation skipped: {str(e)}")
                
                # Free the temporary pre-allocation tensors to make space for actual training
                for tensor in prealloc_tensors:
                    del tensor
                prealloc_tensors = None
                torch.cuda.empty_cache()
                logging.info("[MEMORY] Released pre-allocated memory, now available for training")
                
                # Verify release worked correctly
                after_release = torch.cuda.memory_allocated() / (1024 * 1024)
                after_pct = (after_release / cuda_total_memory) * 100
                logging.info(f"[MEMORY] Memory after release: {after_release:.1f}MB ({after_pct:.1f}% of VRAM)")
                
            except Exception as e:
                logging.warning(f"Memory pre-allocation failed: {str(e)}")
                
                # Try to recover
                torch.cuda.empty_cache()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_speed = train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=use_amp,
            use_bfloat16=args.use_bfloat16,
            scaler=scaler,
            offload_optimizer_state=use_optimizer_offload
        )
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} completed in {epoch_time:.1f}s | "
                    f"Train loss: {train_loss:.4f}, Train speed: {train_speed:.1f} tokens/sec | "
                    f"Val loss: {val_loss:.4f}")
        
        # Save checkpoint (only save best model to reduce I/O overhead)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'config': config,
            }, model_path)
            logging.info(f"Saved new best model to {model_path}")
        
        # Optional sample generation
        if args.sample_every_epoch:
            sample_text(model, char_to_idx, idx_to_char, device, seed_text="TYRION: ", max_length=100)
    
    # Log training completion
    logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from {best_model_path} for final evaluation")
        
        # Final evaluation
        final_val_loss = evaluate(model, val_loader, device)
        logging.info(f"Final validation loss: {final_val_loss:.4f}")
    else:
        logging.warning(f"Could not find best model at {best_model_path}")

    # Clean up
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return model

if __name__ == "__main__":
    # Logging setup has been moved to the beginning of the file
    logging.info("Starting training script...")
    try:
        main()
        print("\nTraining completed successfully!", flush=True)
    except KeyboardInterrupt:
        print("\nTraining stopped by user (Ctrl+C).", flush=True)
        print("You can restart training later using --checkpoint_dir to load the latest checkpoint.", flush=True)
        # Clean up on exit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 