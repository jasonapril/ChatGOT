"""
Common utilities for Craft.

This module provides utility functions used across the framework.
"""
import logging
import random
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed} for reproducibility")


def setup_device(device_name: str = "auto") -> torch.device:
    """
    Set up and return the device for computation.
    
    Args:
        device_name: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        Configured PyTorch device
    """
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    # Log device information
    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device)
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        logging.info(f"  - Total memory: {device_properties.total_memory / 1024**3:.2f} GB")
        logging.info(f"  - CUDA capability: {device_properties.major}.{device_properties.minor}")
    else:
        logging.info("Using CPU for computation")
    
    return device


def format_number(number: Union[int, float]) -> str:
    """
    Format a number with commas for easier reading.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if isinstance(number, int):
        return f"{number:,}"
    elif isinstance(number, float):
        return f"{number:,.2f}"
    else:
        return str(number) 