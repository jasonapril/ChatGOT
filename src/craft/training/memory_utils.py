"""
Memory optimization utilities for training large models.
"""
import logging
import gc
import torch
from typing import Optional


def get_memory_optimized_settings(available_memory_gb: Optional[float] = None) -> dict:
    """
    Get memory-optimized settings based on available GPU memory.

    Args:
        available_memory_gb: Available GPU memory in GB. If None, will be detected.

    Returns:
        Dictionary with optimized settings.
    """
    if available_memory_gb is None:
        # Try to detect available GPU memory
        if torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Get free memory instead of total
            free_memory = torch.cuda.memory_reserved(0) / 1e9
            available_memory_gb = min(available_memory_gb, free_memory)
        else:
            available_memory_gb = 2.0  # Default conservative value

    # Define settings based on available memory
    settings = {}

    if available_memory_gb < 4:
        # Very limited memory
        settings.update({
            'batch_size': 1,
            'gradient_accumulation_steps': 16,
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'attention_implementation': 'memory_efficient',
            'model_size': 'tiny',
            'max_seq_length': 256
        })
    elif available_memory_gb < 8:
        # Limited memory
        settings.update({
            'batch_size': 4,
            'gradient_accumulation_steps': 8,
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'attention_implementation': 'memory_efficient',
            'model_size': 'small',
            'max_seq_length': 512
        })
    elif available_memory_gb < 16:
        # Moderate memory
        settings.update({
            'batch_size': 16,
            'gradient_accumulation_steps': 2,
            'mixed_precision': True,
            'gradient_checkpointing': False,
            'attention_implementation': 'flash_attention',
            'model_size': 'medium',
            'max_seq_length': 1024
        })
    else:
        # Abundant memory
        settings.update({
            'batch_size': 32,
            'gradient_accumulation_steps': 1,
            'mixed_precision': True,
            'gradient_checkpointing': False,
            'attention_implementation': 'standard',
            'model_size': 'large',
            'max_seq_length': 2048
        })

    logging.info(f"Memory-optimized settings for {available_memory_gb:.1f}GB: {settings}")
    return settings


def preallocate_gpu_memory(fraction: float = 0.9) -> None:
    """
    Preallocate GPU memory to avoid fragmentation.

    Args:
        fraction: Fraction of available memory to allocate.
    """
    if not torch.cuda.is_available():
        logging.warning("No GPU available, skipping memory preallocation.")
        return

    # Clear any existing tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Get current device
    device = torch.cuda.current_device()

    # Get available memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = total_memory - allocated_memory

    # Calculate how much to allocate
    memory_to_allocate = int(available_memory * fraction)

    # Allocate memory
    logging.info(f"Preallocating {memory_to_allocate / 1e9:.2f}GB of GPU memory")
    prealloc = torch.empty(memory_to_allocate // 4, dtype=torch.float32, device=device)
    del prealloc
    torch.cuda.empty_cache()

    logging.info("GPU memory preallocation completed")


def reduce_gpu_memory_usage(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply techniques to reduce GPU memory usage during training.

    Args:
        model: The PyTorch model
    """
    # Enable gradient checkpointing if supported
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable() # type: ignore[operator]
        logging.info("Enabled gradient checkpointing")

    # Use CPU offloading for optimizer states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Cleared CUDA cache and collected garbage")

    return model


def log_memory_usage() -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logging.info(f"GPU memory: {allocated:.2f}GB allocated, {max_allocated:.2f}GB max, {cached:.2f}GB reserved")

# Removed MemoryMonitor class as it seemed unused/incomplete 