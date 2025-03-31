#!/usr/bin/env python
"""
Evaluation Module
===============

This module provides functionality for evaluating model performance, including:

1. Efficient validation loop implementation
2. Loss calculation and metrics reporting
3. Performance optimization for evaluation
4. Memory-efficient inference
5. Progress tracking for long evaluation runs

The evaluation functions are optimized for speed and memory efficiency.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import force_flush_logs, format_time

def evaluate(model, dataloader, device, use_amp=False, log_interval=None):
    """
    Evaluate the model on a validation dataset.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader with validation batches
        device: Device to evaluate on
        use_amp: Whether to use automatic mixed precision
        log_interval: How often to log progress (in seconds, None for no logs)
        
    Returns:
        Validation loss
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    last_log_time = start_time
    
    logging.info(f"Starting evaluation on {len(dataloader)} batches")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Transfer data to device
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Update metrics
            total_loss += loss.item()
            total_tokens += inputs.numel()
            
            # Log progress
            if log_interval and time.time() - last_log_time >= log_interval:
                progress_pct = ((i + 1) / len(dataloader)) * 100
                avg_loss = total_loss / (i + 1)
                elapsed = time.time() - start_time
                throughput = total_tokens / elapsed if elapsed > 0 else 0
                
                logging.info(f"Evaluation | "
                            f"Progress: {progress_pct:.1f}% | Batch {i+1}/{len(dataloader)} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Throughput: {throughput:.2f} tokens/sec")
                
                last_log_time = time.time()
                force_flush_logs()
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    
    logging.info(f"Evaluation completed in {format_time(elapsed)}")
    logging.info(f"Validation loss: {avg_loss:.4f}")
    logging.info(f"Evaluation throughput: {tokens_per_sec:.2f} tokens/sec")
    
    return avg_loss

def evaluate_with_metrics(model, dataloader, device, use_amp=False, metrics=None):
    """
    Evaluate the model with additional metrics beyond just loss.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader with validation batches
        device: Device to evaluate on
        use_amp: Whether to use automatic mixed precision
        metrics: Dictionary of metric functions to compute
        
    Returns:
        Dictionary of metrics including loss
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    # Initialize metric accumulators
    metric_values = {}
    if metrics:
        for name in metrics:
            metric_values[name] = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Transfer data to device
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Update loss and token count
            total_loss += loss.item()
            total_tokens += inputs.numel()
            
            # Compute additional metrics
            if metrics:
                for name, metric_fn in metrics.items():
                    metric_values[name] += metric_fn(outputs, targets).item()
    
    # Calculate averages
    results = {'loss': total_loss / len(dataloader)}
    
    if metrics:
        for name in metrics:
            results[name] = metric_values[name] / len(dataloader)
    
    # Calculate throughput
    elapsed = time.time() - start_time
    results['tokens_per_sec'] = total_tokens / elapsed if elapsed > 0 else 0
    results['elapsed_time'] = elapsed
    
    # Log results
    logging.info(f"Evaluation completed in {format_time(elapsed)}")
    logging.info(f"Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in results.items() if k != 'elapsed_time'])}")
    
    return results

def evaluate_perplexity(model, dataloader, device, use_amp=False):
    """
    Calculate model perplexity on a validation dataset.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader with validation batches
        device: Device to evaluate on
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Tuple of (loss, perplexity)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Transfer data to device
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), reduction='sum')
            
            # Update loss and token count
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logging.info(f"Evaluation perplexity: {perplexity:.2f}")
    
    return avg_loss, perplexity 