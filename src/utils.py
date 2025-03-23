"""
Utility Module
=============

This module contains various utility functions for the training process,
including:

1. Configuration management
2. File and directory handling
3. Device setup and CUDA utilities
4. Profiling and benchmarking tools
5. Progress visualization

Design Principles:
- Robust error handling with graceful fallbacks
- Maximum compatibility across different environments
- Minimal dependencies, mostly using standard library
- Comprehensive logging for debugging
"""

import torch
import os
import time
import json
import logging
import argparse
import random
import numpy as np
import multiprocessing
import psutil
import io
import math
import re
from typing import Dict, Any, Optional, Tuple, List, Union

# Global variable to track if we've set PyTorch interop threads
_INTEROP_THREADS_SET = False

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure CuDNN uses deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed} for reproducibility")

def diagnose_cuda_status():
    """Print diagnostic information about CUDA availability and status."""
    import sys
    import subprocess
    import torch

    print("\n=== CUDA DIAGNOSTIC INFORMATION ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Capability: {torch.cuda.get_device_capability(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    - Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is not available. Checking system:")
        
        try:
            # For Windows
            if sys.platform == 'win32':
                result = subprocess.run(['nvidia-smi'], shell=True, check=False, 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    print("NVIDIA driver seems installed (nvidia-smi works):")
                    print(result.stdout.decode('utf-8', errors='replace'))
                else:
                    print("nvidia-smi failed, driver may not be installed")
                    print(f"Error: {result.stderr.decode('utf-8', errors='replace')}")
            # For Linux/Mac
            else:
                result = subprocess.run(['nvidia-smi'], check=False, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    print("NVIDIA driver seems installed (nvidia-smi works):")
                    print(result.stdout.decode('utf-8', errors='replace'))
                else:
                    print("nvidia-smi failed, driver may not be installed")
                    print(f"Error: {result.stderr.decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"Error checking GPU drivers: {e}")
    
    print("=== DIAGNOSTIC COMPLETE ===\n")

def setup_device(force_cpu: bool = False) -> Tuple[torch.device, bool, int]:
    """
    Set up the device for training, with appropriate optimizations.
    
    Args:
        force_cpu: Whether to force CPU usage even if CUDA is available
        
    Returns:
        Tuple containing (device, is_cuda_available, num_gpus)
    """
    global _INTEROP_THREADS_SET
    
    # Check for CUDA availability
    is_cuda = torch.cuda.is_available() and not force_cpu
    num_gpus = torch.cuda.device_count() if is_cuda else 0
    device = torch.device("cuda" if is_cuda else "cpu")
    
    if is_cuda:
        # Log GPU information
        logging.info(f"Using CUDA with {num_gpus} GPU(s)")
        logging.info(f"PyTorch version: {torch.__version__}")
        
        # Print and log available memory for each GPU
        gpu_info = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem_free_gb = mem_free / (1024**3)
            mem_total_gb = mem_total / (1024**3)
            gpu_info.append(f"GPU {i}: {props.name}, {mem_total_gb:.2f} GB total, {mem_free_gb:.2f} GB free")
            
        logging.info("GPU Memory Info:\n" + "\n".join(gpu_info))
        
        # Enable TF32 precision on Ampere or newer GPUs (much faster with minimal precision loss)
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("Enabled TF32 precision for faster training")
        
        # Optimize cuDNN
        torch.backends.cudnn.benchmark = True
        logging.info("Enabled cuDNN benchmark mode for faster training")
        
        return device, is_cuda, num_gpus
    else:
        if force_cpu:
            logging.info("Forced CPU usage as requested")
        else:
            logging.warning("*** CUDA NOT DETECTED - FALLING BACK TO CPU ***")
            
        logging.info(f"PyTorch version: {torch.__version__}")
        
        # Try to enable CPU optimizations
        if hasattr(torch, 'set_num_threads'):
            # Use all available CPU cores
            num_cores = multiprocessing.cpu_count()
            torch.set_num_threads(num_cores)
            logging.info(f"Set PyTorch to use all {num_cores} CPU cores")
        
        if hasattr(torch, 'set_num_interop_threads') and not _INTEROP_THREADS_SET:
            # Set inter-op parallelism to number of physical cores
            torch.set_num_interop_threads(max(2, multiprocessing.cpu_count() // 2))
            logging.info(f"Set PyTorch interop threads to {max(2, multiprocessing.cpu_count() // 2)}")
            _INTEROP_THREADS_SET = True
            
        # Enable MKL optimizations if available
        try:
            import mkl
            mkl.set_num_threads(multiprocessing.cpu_count())
            logging.info(f"Enabled MKL optimizations with {multiprocessing.cpu_count()} threads")
        except ImportError:
            logging.info("MKL not available, using default CPU optimizations")
            
        logging.warning("Training on CPU will be significantly slower than on GPU")
        
        # Try to get more diagnostic information
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logging.info(f"nvidia-smi output:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"nvidia-smi errors:\n{result.stderr}")
        except Exception as e:
            logging.warning(f"Could not run nvidia-smi: {e}")
        
        return device, is_cuda, num_gpus

def get_time_estimate(
    current_step: int, 
    total_steps: int, 
    time_elapsed: float
) -> Dict[str, Union[float, str]]:
    """
    Calculate time estimates for training progress.
    
    Args:
        current_step: Current training step
        total_steps: Total number of steps
        time_elapsed: Time elapsed so far (in seconds)
        
    Returns:
        Dictionary with time estimates
    """
    if current_step == 0:
        return {
            "avg_time_per_step": 0.0,
            "estimated_time_remaining": "unknown",
            "estimated_total_time": "unknown",
            "completion_percentage": 0.0
        }
    
    # Calculate average time per step (in seconds)
    avg_time_per_step = time_elapsed / current_step
    
    # Calculate estimated remaining time (in seconds)
    remaining_steps = total_steps - current_step
    estimated_time_remaining = remaining_steps * avg_time_per_step
    
    # Calculate estimated total time (in seconds)
    estimated_total_time = total_steps * avg_time_per_step
    
    # Calculate completion percentage
    completion_percentage = (current_step / total_steps) * 100
    
    # Format time values for human readability
    def format_time(seconds: float) -> str:
        """Format seconds into hours:minutes:seconds string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    return {
        "avg_time_per_step": avg_time_per_step,
        "estimated_time_remaining": format_time(estimated_time_remaining),
        "estimated_total_time": format_time(estimated_total_time),
        "completion_percentage": completion_percentage
    }

def format_number(number: Union[int, float]) -> str:
    """
    Format a number for human readability.
    
    Args:
        number: The number to format
        
    Returns:
        Formatted string representation
    """
    if isinstance(number, int):
        # Format large integers with commas
        return f"{number:,}"
    else:
        # Format floats with appropriate precision
        if abs(number) >= 1000:
            return f"{number:,.2f}"  # Large numbers with comma and 2 decimal places
        elif abs(number) >= 10:
            return f"{number:.2f}"   # Medium numbers with 2 decimal places
        elif abs(number) >= 0.01:
            return f"{number:.4f}"   # Small numbers with 4 decimal places
        else:
            return f"{number:.6e}"   # Very small numbers in scientific notation

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    tokenizer_data: Dict[str, Any],
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None
) -> str:
    """
    Save a training checkpoint with model and optimizer state.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        epoch: Current epoch number
        loss: Current loss value
        args: Command line arguments
        tokenizer_data: Dictionary with character mappings
        checkpoint_dir: Directory to save checkpoints
        filename: Specific filename to use (default: None for automatic naming)
        
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate checkpoint filename if not provided
    if filename is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_loss_{loss:.4f}_{timestamp}.pt"
    
    # Full path to checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'args': vars(args),  # Convert Namespace to dict
        'char_to_idx': tokenizer_data.get('char_to_idx', {}),
        'idx_to_char': tokenizer_data.get('idx_to_char', {})
    }
    
    # Save checkpoint to file
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a training checkpoint with model and optimizer state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The PyTorch model (optional)
        optimizer: The optimizer (optional)
        scheduler: The learning rate scheduler (optional)
        device: Device to load the model onto (optional)
        
    Returns:
        Dictionary containing the loaded checkpoint data
    """
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint with appropriate map_location
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Restore model state if provided
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model state loaded from {checkpoint_path}")
    
    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Optimizer state loaded from {checkpoint_path}")
    
    # Restore scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"Scheduler state loaded from {checkpoint_path}")
    
    # Log checkpoint information
    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    return checkpoint

def get_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    """
    Find the most recent checkpoint file in the given directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        logging.warning(f"No checkpoint files found in {checkpoint_dir}.")
        return None
    
    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # Return the most recent checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    logging.info(f"Latest checkpoint found: {latest_checkpoint}")
    
    return latest_checkpoint

def create_output_dir(base_dir: str = "outputs") -> str:
    """
    Create timestamped output directory for model outputs.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Path to the created output directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory created: {output_dir}")
    
    return output_dir

def save_args(args: argparse.Namespace, output_dir: str) -> str:
    """
    Save command line arguments to a JSON file.
    
    Args:
        args: Command line arguments
        output_dir: Directory to save arguments to
        
    Returns:
        Path to the saved arguments file
    """
    # Convert Namespace to dictionary
    args_dict = vars(args)
    
    # Create output file path
    args_file = os.path.join(output_dir, "args.json")
    
    # Save arguments as JSON
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
    
    logging.info(f"Arguments saved to {args_file}")
    
    return args_file

def speed_benchmark(
    model: torch.nn.Module, 
    inputs: torch.Tensor, 
    device: torch.device,
    warmup: int = 5, 
    iterations: int = 20,
    use_amp: bool = False,
    use_bfloat16: bool = False
) -> Dict[str, Any]:
    """
    Benchmark model inference speed.
    
    Args:
        model: The PyTorch model to benchmark
        inputs: Input tensor to use for benchmarking
        device: Device to run on
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        use_amp: Whether to use automatic mixed precision
        use_bfloat16: Whether to use bfloat16 instead of float16
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    
    # Ensure inputs are on the correct device
    inputs = inputs.to(device)
    
    # Run warmup iterations to initialize CUDA for accurate timing
    logging.info(f"Running {warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                    _ = model(inputs)
            else:
                _ = model(inputs)
    
    # Synchronize CUDA for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Run benchmark iterations and measure time
    logging.info(f"Running {iterations} benchmark iterations...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                    _ = model(inputs)
            else:
                _ = model(inputs)
    
    # Synchronize CUDA for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate results
    total_time = end_time - start_time
    avg_time = total_time / iterations
    batch_size = inputs.size(0)
    seq_length = inputs.size(1)
    
    # Calculate tokens per second
    tokens_per_batch = batch_size * seq_length
    tokens_per_second = tokens_per_batch / avg_time
    
    # Get memory usage if using CUDA
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)    # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    # Compile results
    results = {
        "device": str(device),
        "batch_size": batch_size,
        "sequence_length": seq_length,
        "tokens_per_batch": tokens_per_batch,
        "total_time": total_time,
        "iterations": iterations,
        "avg_time_per_batch": avg_time,
        "tokens_per_second": tokens_per_second,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "precision": "bfloat16" if use_amp and use_bfloat16 else "float16" if use_amp else "float32"
    }
    
    # Log results
    logging.info(f"Benchmark results:")
    logging.info(f"- Average time per batch: {avg_time*1000:.2f} ms")
    logging.info(f"- Throughput: {tokens_per_second:.2f} tokens/second")
    logging.info(f"- Memory allocated: {memory_allocated:.2f} MB")
    
    return results

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: The PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """
    Estimate memory usage for a model in different precisions.
    
    Args:
        model: The PyTorch model
        
    Returns:
        Dictionary with memory estimates in MB for different precisions
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Memory for different precisions (in bytes)
    float32_size = total_params * 4  # 4 bytes per parameter
    float16_size = total_params * 2  # 2 bytes per parameter
    int8_size = total_params * 1     # 1 byte per parameter
    
    # Convert to MB
    def to_mb(bytes_value: float) -> float:
        return bytes_value / (1024 * 1024)
    
    # Add optimizer state (approximately 2x model size for Adam)
    adam_state_size = float32_size * 2
    
    # Add activations (rough estimate based on model size)
    # For a transformer, activations can be ~4x the model size during training
    activations_size = float32_size * 4
    
    # Total memory during training and inference
    training_memory = float32_size + adam_state_size + activations_size
    inference_memory_float32 = float32_size + (activations_size / 4)  # Less activations during inference
    inference_memory_float16 = float16_size + (activations_size / 8)  # Half precision
    inference_memory_int8 = int8_size + (activations_size / 16)       # Quantized
    
    return {
        "parameters": total_params,
        "model_size_float32_mb": to_mb(float32_size),
        "model_size_float16_mb": to_mb(float16_size),
        "model_size_int8_mb": to_mb(int8_size),
        "training_memory_mb": to_mb(training_memory),
        "inference_memory_float32_mb": to_mb(inference_memory_float32),
        "inference_memory_float16_mb": to_mb(inference_memory_float16),
        "inference_memory_int8_mb": to_mb(inference_memory_int8)
    } 