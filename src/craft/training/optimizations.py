#!/usr/bin/env python
"""
Training Optimizations Module
=========================

This module provides optimization functions for the training process, including:

1. Batch size optimization
2. CUDA and memory optimizations
3. Hardware-specific configurations
4. Performance tuning utilities

These functions are designed to maximize training throughput on the available hardware.
"""

import argparse
import logging
import torch
from typing import Dict, Any, Optional, Tuple

def setup_mixed_precision(args: argparse.Namespace) -> Tuple[bool, Optional[torch.cuda.amp.GradScaler]]:
    """
    Set up mixed precision training if requested.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (use_amp, scaler)
    """
    scaler = None
    
    if args.use_amp:
        if not torch.cuda.is_available():
            logging.warning("AMP requested but CUDA not available. Disabling AMP.")
            args.use_amp = False
        else:
            logging.info("Setting up automatic mixed precision training")
            dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16
            
            # Check if bfloat16 is supported
            if args.use_bfloat16 and not torch.cuda.is_bf16_supported():
                logging.warning("bfloat16 requested but not supported by this GPU. Using float16 instead.")
                dtype = torch.float16
            
            # Create gradient scaler for AMP
            scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
            logging.info(f"AMP enabled with dtype: {dtype}")
    
    return args.use_amp, scaler

def optimize_memory_usage(args: argparse.Namespace, device: torch.device) -> None:
    """
    Optimize memory usage for training.
    
    Args:
        args: Command-line arguments
        device: PyTorch device
    """
    if not torch.cuda.is_available() or device.type != 'cuda':
        return
        
    from src.memory_management import preallocate_gpu_memory, get_memory_optimized_settings
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    logging.info(f"GPU: {gpu_name} with {total_memory:.2f}GB total memory")
    
    # Get memory optimized settings
    mem_settings = get_memory_optimized_settings(
        gpu_name=gpu_name, 
        force_aggressive=args.force_aggressive_memory
    )
    
    # Apply memory settings
    if mem_settings.get('preallocate', False):
        target_usage = mem_settings.get('target_usage', 0.7)
        preallocate_gpu_memory(target_usage=target_usage)
        
    # Log memory stats
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logging.info(f"GPU memory after optimization: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def setup_torch_compile(args: argparse.Namespace, model: torch.nn.Module) -> torch.nn.Module:
    """
    Set up torch.compile for the model if available and enabled.

    Args:
        args: Training arguments
        model: Model to compile

    Returns:
        Compiled model if torch.compile is available and enabled, otherwise original model
    """
    if not args.torch_compile:
        return model

    if not hasattr(torch, 'compile'):
        logging.warning("torch.compile not available - using uncompiled model")
        return model

    try:
        logging.info(f"Compiling model with mode: {args.compile_mode}")
        return torch.compile(model, mode=args.compile_mode)
    except Exception as e:
        logging.warning(f"Failed to compile model: {e}")
        return model

def configure_activation_checkpointing(args: argparse.Namespace, model: torch.nn.Module) -> None:
    """
    Configure activation checkpointing for memory savings if requested.
    
    Args:
        args: Command-line arguments
        model: The model to configure
    """
    if not args.use_activation_checkpointing:
        return
        
    try:
        # This will need to be adapted based on the actual model architecture
        # For a transformer model, typically need to apply checkpointing to transformer blocks/layers
        
        if hasattr(model, 'transformer_layers'):
            import torch.utils.checkpoint as checkpoint
            
            logging.info("Applying activation checkpointing to transformer layers")
            
            # Example - actual implementation depends on specific model architecture
            for layer in model.transformer_layers:
                layer.use_checkpoint = True
                
        else:
            logging.warning("Model doesn't have expected structure for activation checkpointing")
            
    except Exception as e:
        logging.warning(f"Failed to apply activation checkpointing: {e}") 