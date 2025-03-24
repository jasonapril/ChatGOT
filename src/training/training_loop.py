#!/usr/bin/env python
"""
Training Loop Module
==================

This module provides the core training loop functionality, including:

1. Efficient training epoch implementation
2. Memory-optimized backpropagation
3. Mixed precision support
4. Performance monitoring
5. Gradient accumulation

This is designed to maximize training throughput while providing detailed progress feedback.
"""

import torch
import torch.nn.functional as F
import logging
import time
import gc
import contextlib
from typing import Tuple, Dict, Any, Callable, List, Optional
import numpy as np
from tqdm import tqdm

from src.logger import force_flush_logs, format_time

def train_epoch(model, optimizer, scheduler, dataloader, device, epoch, 
               max_grad_norm=1.0, gradient_accumulation_steps=1, use_amp=False, 
               use_bfloat16=False, scaler=None, offload_optimizer_state=False):
    """
    Train for one epoch with maximum speed and memory optimizations.
    
    This function implements an advanced training loop with features such as:
    - Non-blocking data transfers for overlapped compute/transfer
    - Gradient accumulation for effective larger batch sizes
    - Mixed precision training with automatic detection for different precisions
    - Memory-efficient optimizer state handling
    - Aggressive memory managament for maximum throughput
    - Comprehensive progress logging with real-time metrics
    
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
        
    Returns:
        tuple: (avg_loss, tokens_per_sec)
    """
    # Check if using 8-bit optimizer
    is_8bit_optimizer = False
    try:
        import bitsandbytes as bnb
        if isinstance(optimizer, bnb.optim.AdamW8bit) or isinstance(optimizer, bnb.optim.Adam8bit) or isinstance(optimizer, bnb.optim.SGD8bit):
            is_8bit_optimizer = True
            logging.info("Using 8-bit optimizer for training")
    except (ImportError, AttributeError):
        pass
    
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
    
    # Log epoch start info
    logging.info(f"Starting training epoch {epoch} with gradient accumulation steps: {gradient_accumulation_steps}")
    logging.info(f"Total batches: {total_batches}, Batch size: {dataloader.batch_size}")
    
    # Memory tracking
    peak_memory = 0
    if device.type == 'cuda':
        # Clear memory at start of epoch
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Initialize step counter
    step = epoch * len(dataloader)
    
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
            tokens_remaining = (total_tokens / (i + 1)) * (total_batches - (i + 1)) if i > 0 else 0
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
            
            # Log minute update
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
        
        # Forward pass with mixed precision for efficiency
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Get model output
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # Scale by accumulation factor
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
        
        # Update total loss and tokens
        total_loss += loss.item() * gradient_accumulation_steps
        total_tokens += inputs.numel()
        
        # Backward pass with mixed precision
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if gradient accumulation step is reached
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
            # Clip gradients
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Apply optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Increment step counter
            step += 1
        
        # Track batch time for throughput calculation
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Log each 10% of the epoch or at early update batches
        if ((i + 1) % max(1, total_batches // 10) == 0 or 
            i == 0 or (i + 1) == total_batches or
            i + 1 in early_update_batches or
            time.time() - last_log_time >= log_interval):
            
            # Calculate progress and stats
            progress_pct = ((i + 1) / total_batches) * 100
            avg_loss = total_loss / (i + 1)
            
            # Calculate throughput metrics
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            avg_batch_time = sum(batch_times) / len(batch_times)
            
            # Calculate estimated time remaining
            batches_remaining = total_batches - (i + 1)
            eta_seconds = batches_remaining * avg_batch_time
            
            # Format ETA nicely
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"
            
            # Get memory usage if available
            memory_str = ""
            if device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                peak_memory = max(peak_memory, current_memory)
                cuda_max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                memory_utilization = (current_memory / cuda_max_memory) * 100
                memory_str = f"Memory: {current_memory:.1f}MB/{cuda_max_memory:.1f}MB ({memory_utilization:.1f}%)"
            
            # Log progress
            logging.info(f"Epoch {epoch} | "
                        f"Progress: {progress_pct:.1f}% | Batch {i+1}/{total_batches} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Throughput: {tokens_per_sec:.2f} tokens/sec | "
                        f"ETA: {eta_str} | {memory_str}")
            
            # Force flush logs
            force_flush_logs()
            
            # Update last log time
            last_log_time = time.time()
    
    # Epoch completed - calculate final stats
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    avg_loss = total_loss / len(dataloader)
    
    # Final memory usage report
    if device.type == 'cuda':
        logging.info(f"Peak GPU memory usage: {peak_memory:.1f}MB")
    
    logging.info(f"Epoch {epoch} completed in {format_time(elapsed)}")
    logging.info(f"Average loss: {avg_loss:.4f}")
    logging.info(f"Training throughput: {tokens_per_sec:.2f} tokens/sec")
    
    return avg_loss, tokens_per_sec 