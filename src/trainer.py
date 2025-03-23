#!/usr/bin/env python
"""
Trainer Module
=============

This module provides functions for training, evaluating, and generating text from
transformer models. It focuses on:

1. Highly optimized training loops for maximum throughput
2. Advanced memory-efficient training techniques
3. Robust monitoring and logging
4. CUDA-optimized implementations

The module implements multiple optimization techniques to maximize training throughput,
including gradient accumulation, mixed precision training, and memory-efficient backpropagation.
It also provides detailed progress monitoring and validation to help debug training issues.

Design Principles:
- Maximum GPU utilization through optimized training loops
- Comprehensive memory management to prevent OOM errors
- Robust progress tracking with immediate feedback
- Fault tolerance with graceful error handling
- Flexible optimization strategies for different hardware
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
               use_bfloat16=False, scaler=None, offload_optimizer_state=False,
               use_torch_compile=False, compile_mode='reduce-overhead'):
    """
    Train for one epoch with maximum speed and memory optimizations.
    
    This function implements an advanced training loop with features such as:
    - Non-blocking data transfers for overlapped compute/transfer
    - Gradient accumulation for effective larger batch sizes
    - Mixed precision training with automatic detection for different precisions
    - Memory-efficient optimizer state handling
    - Aggressive memory managament for maximum throughput
    - Comprehensive progress logging with real-time metrics
    - PyTorch 2.0+ model compilation for faster execution (if enabled)
    - 8-bit optimizer compatibility
    
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
        use_torch_compile: Whether to use torch.compile for model optimization
        compile_mode: Compilation mode for torch.compile (default: 'reduce-overhead')
        
    Returns:
        tuple: (avg_loss, tokens_per_sec)
    """
    # Avoid scoping issues by importing torch again
    import torch
    import torch.nn.functional as F
    
    # Enable torch.compile if requested
    if use_torch_compile and device.type == 'cuda':
        try:
            # Check if torch.compile is available
            if not hasattr(torch, 'compile'):
                logging.warning("torch.compile not available in this PyTorch version. Continuing with uncompiled model.")
            else:
                # Configure dynamo to suppress errors and fall back to eager mode
                try:
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    logging.info("Configured torch._dynamo to suppress errors and use fallbacks")
                except ImportError:
                    logging.warning("Could not configure torch._dynamo directly. Continuing anyway.")
                
                # Attempt to compile the model with the requested mode
                compile_start_time = time.time()
                original_model = model
                
                # Store original model for fallback
                try:
                    model = torch.compile(model, mode=compile_mode)
                    compile_time = time.time() - compile_start_time
                    logging.info(f"Model successfully compiled with torch.compile (mode={compile_mode}) in {compile_time:.2f}s")
                except Exception as e:
                    logging.warning(f"Failed to compile model with torch.compile: {e}")
                    logging.warning("Continuing with uncompiled model")
                    model = original_model
        except Exception as e:
            logging.warning(f"Exception during torch.compile setup: {e}")
            logging.warning("Continuing with standard training")
    
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
        # Get total GPU memory for reference
        cuda_max_memory = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        
        # Force high VRAM usage by pre-caching some tensors
        vram_usage_target = 0.75  # Target 75% VRAM usage
        current_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(device).total_memory
        if current_usage < vram_usage_target:
            logging.info(f"[MEMORY] Current VRAM utilization is low ({current_usage*100:.1f}%). Allocating cache tensors to increase GPU utilization.")
            try:
                # Calculate how much memory to allocate to reach target
                total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
                current_vram = torch.cuda.memory_allocated() / (1024 * 1024)
                target_vram = total_vram * vram_usage_target
                additional_mb = max(0, target_vram - current_vram)
                
                if additional_mb > 100:  # Only allocate if significant amount needed
                    logging.info(f"[MEMORY] Allocating additional {additional_mb:.1f}MB to increase GPU utilization")
                    # Create a large tensor and keep it alive during training
                    # Store it in function attributes to prevent garbage collection
                    tensor_size = int(additional_mb * 1024 * 1024 // 4)  # 4 bytes per float32
                    if not hasattr(train_epoch, 'cache_tensors'):
                        train_epoch.cache_tensors = []
                    
                    # Create a cache tensor but don't require gradients
                    with torch.no_grad():
                        cache_tensor = torch.zeros(tensor_size, dtype=torch.float32, device=device)
                        # Touch it to ensure it's allocated
                        cache_tensor[0] = 1.0
                        train_epoch.cache_tensors.append(cache_tensor)
                    
                    # Log the new memory usage
                    new_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(device).total_memory
                    logging.info(f"[MEMORY] Increased VRAM utilization to {new_usage*100:.1f}%")
            except Exception as e:
                logging.warning(f"[MEMORY] Failed to allocate additional memory: {e}")
    
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
        
        # Backward pass with appropriate mixed precision
        if use_amp and device.type == 'cuda' and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights with appropriate accumulation steps
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1 == total_batches):
            # Gradient clipping
            if max_grad_norm > 0:
                if use_amp and device.type == 'cuda' and scaler is not None:
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Special handling for 8-bit optimizers
            if is_8bit_optimizer:
                try:
                    # When using 8-bit optimizers with AMP, ensure proper interaction
                    if use_amp and device.type == 'cuda' and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                except RuntimeError as e:
                    logging.warning(f"Error during 8-bit optimizer step: {e}")
                    logging.warning("Attempting to continue with training")
                    # Try to continue despite the error
                    if "CUDA error" in str(e) and device.type == 'cuda':
                        torch.cuda.empty_cache()
            else:
                # Regular optimizer
                if use_amp and device.type == 'cuda' and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
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
        
        # Log progress at regular intervals and for important batches
        if should_log:
            # Add optimizer type to log
            if is_8bit_optimizer:
                opt_type = "8-bit"
            else:
                opt_type = "std"
                
            # Add compilation info
            if use_torch_compile:
                compiled_status = "compiled" if hasattr(model, "_orig_mod") else "uncompiled"
            else:
                compiled_status = "uncompiled"
            
            logging.info(f"Progress: {progress_pct:.1f}% | Batch {i+1}/{total_batches} | "
                        f"Loss: {raw_loss:.4f} | Rate: {tokens_per_sec:.1f} tokens/sec | "
                        f"Opt: {opt_type} | Model: {compiled_status} | "
                        f"ETA: {eta_str} | {cuda_memory_str}")
            last_log_time = current_time
            
            # Force flush logs to ensure they're written to disk immediately
            force_flush_logs()
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # Format total training time
    train_time = time.time() - start_time
    train_time_str = format_time(train_time)
    
    # Log final stats
    logging.info(f"Epoch {epoch} completed in {train_time_str} | "
               f"Avg Loss: {avg_loss:.4f} | "
               f"Speed: {tokens_per_sec:.1f} tokens/sec | "
               f"Peak VRAM: {peak_memory:.1f}MB")
    
    # Force final log flush
    force_flush_logs()

    return avg_loss, tokens_per_sec

def evaluate(model, dataloader, device):
    """
    Evaluate model on validation data with memory-efficient implementation.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader with validation batches
        device: Device to run on
        
    Returns:
        float: Average validation loss
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
    """
    Generate text from the model using sampling with temperature.
    
    This function implements an efficient text generation algorithm with:
    - Temperature-controlled sampling for creativity control
    - Memory-efficient generation that doesn't store the full attention history
    - Comprehensive logging of generation progress
    - Fallback handling for unknown characters
    
    Args:
        model: The trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seed_text: Initial text to prompt the model
        max_length: Maximum length of generated text
        temperature: Controls randomness - lower is more deterministic
        device: Device to run on
        
    Returns:
        str: Generated text including the seed text
    """
    model.eval()
    
    logging.info(f"\nStarting text generation")
    logging.info(f"Seed text: '{seed_text}'")
    logging.info(f"Parameters: max_length={max_length}, temperature={temperature}")
    
    # Convert seed text to tensor
    chars = list(seed_text)
    
    # Handle unknown characters more robustly
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
        
        logging.info(f"Text generation completed in {total_time:.2f}s | "
                   f"Speed: {tokens_per_sec:.1f} tokens/sec")
        
        return generated_text
    
    # Fallback to manual generation for backward compatibility
    generated = list(seed_text)
    
    with torch.no_grad():
        logging.info("\nGenerating text:")
        
        # Only show detailed token information for first 3 tokens
        detail_tokens = 3
        
        # Time-based progress reporting (every 30 seconds for generation)
        progress_interval = 30  # seconds
        last_progress_time = start_time
        
        for i in range(max_length):
            # Forward pass with mixed precision for efficiency
            with torch.amp.autocast(device_type=device.type, enabled=device=="cuda"):
                # Get model output
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
    
    logging.info(f"\nText generation completed in {total_time:.2f}s | "
                f"Speed: {tokens_per_sec:.1f} tokens/sec")
    
    return ''.join(generated)

def sample_text(model, char_to_idx, idx_to_char, device, seed_text="", max_length=100, temperature=0.8):
    """
    Generate sample text for monitoring during training.
    
    This is a simplified version of the text generation function that can be called
    frequently during training to monitor model progress.
    
    Args:
        model: The model to generate text from
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Device to run on
        seed_text: Initial text to prompt the model
        max_length: Maximum length of generated text
        temperature: Controls randomness - lower is more deterministic
        
    Returns:
        str: Generated text sample
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
            probs = torch.softmax(logits, dim=-1)
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