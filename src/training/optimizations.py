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

def optimize_training_parameters(args: argparse.Namespace, device: torch.device,
                                is_cuda: bool, char_to_idx: Dict[str, int]) -> argparse.Namespace:
    """
    Optimize training parameters for maximum throughput.
    
    Args:
        args: Command-line arguments
        device: PyTorch device
        is_cuda: Whether CUDA is available
        char_to_idx: Character to index mapping
        
    Returns:
        Updated arguments with optimized settings
    """
    logging.info("OPTIMIZING TRAINING PARAMETERS")
    
    # Apply CUDA optimizations if requested
    if args.optimize_cuda and is_cuda:
        logging.info("Applying CUDA optimizations...")
        from src.utils.device import apply_all_cuda_optimizations
        cuda_results = apply_all_cuda_optimizations()
        
        # Log optimization results
        for name, success in cuda_results.items():
            logging.info(f"CUDA Optimization '{name}': {'Enabled' if success else 'Disabled'}")
    
    # Find optimal batch size if requested
    if args.optimize_batch_size and is_cuda:
        from src.batch_size_finder import find_optimal_batch_size
        
        logging.info("Finding optimal batch size...")
        
        # Create args object for batch size finder
        bsf_args = argparse.Namespace()
        bsf_args.data_path = args.data_path
        bsf_args.sequence_length = args.sequence_length
        bsf_args.min_batch = 1
        bsf_args.max_batch = 256 if not hasattr(args, 'max_batch') else args.max_batch
        bsf_args.test_batches = args.test_batches
        bsf_args.force_cpu = args.force_cpu
        
        # Find optimal batch size
        optimal_batch = find_optimal_batch_size(bsf_args)
        logging.info(f"Optimal batch size determined: {optimal_batch}")
        
        # Update batch size
        args.batch_size = optimal_batch
    
    # Calculate gradient accumulation steps if effective batch size is specified
    if args.effective_batch_size is not None and args.batch_size is not None:
        args.gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)
        logging.info(f"Using gradient accumulation steps: {args.gradient_accumulation_steps}")
    elif args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 1
    
    return args

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

def setup_8bit_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> Optional[torch.optim.Optimizer]:
    """
    Set up 8-bit optimizer if requested and available.
    
    Args:
        args: Command-line arguments
        model: The model to optimize
        
    Returns:
        Optimizer or None if 8-bit optimizer not requested/available
    """
    if not (args.use_8bit_optimizer or args.use_8bit_adam or args.use_8bit_adamw):
        return None
        
    try:
        import bitsandbytes as bnb
        logging.info("Setting up 8-bit optimizer")
        
        if args.use_8bit_adam:
            logging.info("Using 8-bit Adam optimizer")
            return bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
        elif args.use_8bit_adamw:
            logging.info("Using 8-bit AdamW optimizer")
            return bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
        else:
            logging.info("Using generic 8-bit optimizer")
            return bnb.optim.optimizer.Optimizer8bit(
                model.parameters(),
                lr=args.lr,
                optim_bits=8
            )
            
    except ImportError:
        logging.warning("bitsandbytes not available. Using standard optimizer instead.")
        return None

def setup_torch_compile(args: argparse.Namespace, model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply torch.compile to the model if requested and available.
    
    Args:
        args: Command-line arguments
        model: The model to compile
        
    Returns:
        Original or compiled model
    """
    if not args.use_torch_compile:
        return model
        
    if not torch.cuda.is_available():
        logging.warning("torch.compile requested but CUDA not available. Skipping compilation.")
        return model
    
    try:
        if not hasattr(torch, 'compile'):
            logging.warning("torch.compile not available in this PyTorch version. Continuing with uncompiled model.")
            return model
        
        # Configure dynamo to suppress errors and fall back to eager mode
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            logging.info("Configured torch._dynamo to suppress errors and use fallbacks")
        except ImportError:
            logging.warning("Could not configure torch._dynamo directly. Continuing anyway.")
        
        # Attempt to compile the model with the requested mode
        import time
        compile_start_time = time.time()
        original_model = model
        
        try:
            model = torch.compile(model, mode=args.compile_mode)
            compile_time = time.time() - compile_start_time
            logging.info(f"Model successfully compiled with torch.compile (mode={args.compile_mode}) in {compile_time:.2f}s")
        except Exception as e:
            logging.warning(f"Failed to compile model with torch.compile: {e}")
            logging.warning("Continuing with uncompiled model")
            model = original_model
            
        return model
        
    except Exception as e:
        logging.warning(f"Exception during torch.compile setup: {e}")
        logging.warning("Continuing with standard model")
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