"""
Metrics calculation utilities for model training and evaluation.
"""
import math
import time
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_tokens_per_second(
    num_tokens: int,
    elapsed_time: float,
    window_size: int = 10,
    history: Optional[List[float]] = None
) -> float:
    """
    Calculate tokens processed per second with smoothing.

    Args:
        num_tokens: Number of tokens processed
        elapsed_time: Elapsed time in seconds
        window_size: Size of the moving average window
        history: Previous measurements

    Returns:
        Smoothed tokens per second
    """
    if elapsed_time == 0:
        return 0.0

    current_tps = num_tokens / elapsed_time

    # Apply smoothing if history provided
    if history is not None:
        history.append(current_tps)
        if len(history) > window_size:
            history.pop(0)
        return sum(history) / len(history)

    return current_tps


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.

    Args:
        loss: Cross-entropy loss

    Returns:
        Perplexity
    """
    return math.exp(loss) if loss < 100 else float('inf')


def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model size statistics
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    # Calculate activation memory (approximate)
    batch_size = 1  # Just for estimation
    seq_len = 1024  # Typical sequence length
    d_model: int
    if hasattr(model, 'd_model') and isinstance(getattr(model, 'd_model'), int):
        d_model = getattr(model, 'd_model')
    else:
        d_model = 768  # Default for standard models
        if hasattr(model, 'd_model'):
            logger.warning(f"model.d_model found but is not an integer ({type(getattr(model, 'd_model', None))}). Using default d_model=768.")
        else:
            logger.warning(f"model.d_model not found. Using default d_model=768.")

    # Estimated memory for activations in a forward pass
    # This is a rough estimate and will vary by architecture
    activation_size = batch_size * seq_len * d_model * 4  # float32 is 4 bytes

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_memory_mb': param_size / (1024 ** 2),
        'buffer_memory_mb': buffer_size / (1024 ** 2),
        'activation_memory_mb': activation_size / (1024 ** 2),
        'total_memory_mb': (param_size + buffer_size + activation_size) / (1024 ** 2)
    }


def calculate_throughput_stats(
    tokens_per_second_history: List[float]
) -> Dict[str, float]:
    """
    Calculate throughput statistics.

    Args:
        tokens_per_second_history: History of tokens per second measurements

    Returns:
        Dictionary with throughput statistics
    """
    if not tokens_per_second_history:
        return {
            'mean_tps': 0.0,
            'median_tps': 0.0,
            'min_tps': 0.0,
            'max_tps': 0.0,
            'std_tps': 0.0
        }

    tps_array = np.array(tokens_per_second_history)

    return {
        'mean_tps': float(np.mean(tps_array)),
        'median_tps': float(np.median(tps_array)),
        'min_tps': float(np.min(tps_array)),
        'max_tps': float(np.max(tps_array)),
        'std_tps': float(np.std(tps_array))
    }


def calculate_training_eta(
    current_step: int,
    total_steps: int,
    tokens_per_second: float,
    tokens_per_step: int
) -> float:
    """
    Calculate estimated time remaining for training.

    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        tokens_per_second: Current tokens per second
        tokens_per_step: Tokens processed per step

    Returns:
        Estimated seconds remaining
    """
    if tokens_per_second == 0 or current_step >= total_steps:
        return 0.0

    steps_remaining = total_steps - current_step
    tokens_remaining = steps_remaining * tokens_per_step
    seconds_remaining = tokens_remaining / tokens_per_second

    return seconds_remaining 