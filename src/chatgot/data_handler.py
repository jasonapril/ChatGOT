"""
Data Handler Module
==================

This module provides functions for loading and processing training data efficiently.
It focuses on:

1. Optimized data loading with proper parallelization
2. Memory-efficient data handling
3. Automatic batch size optimization
4. Comprehensive data validation and error handling

The design prioritizes throughput and efficiency, ensuring that data loading doesn't
become a bottleneck in the training process. It implements advanced techniques like
prefetching, pinned memory, and parallel processing to maximize GPU utilization.

Design Principles:
- Minimize CPU-GPU transfer bottlenecks
- Optimize worker management for multi-core utilization
- Balance memory usage with loading speed
- Provide robust error handling for data loading issues
"""

import pickle
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Any, Optional

def load_data(data_path: str, batch_size: Optional[int], device_type: str, 
             num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Load processed data from pickle file with optimized data loading settings.
    
    This function implements an advanced data loading strategy with features such as:
    - Multi-worker data loading for CPU utilization
    - Prefetching for reduced latency
    - Pinned memory for faster CPU->GPU transfers
    - Automatic batch size determination
    
    Args:
        data_path: Path to the processed data pickle file
        batch_size: Batch size for training (None for auto-detection)
        device_type: Device type (cuda or cpu)
        num_workers: Number of data loading worker processes
        
    Returns:
        tuple: (train_loader, val_loader, char_to_idx, idx_to_char)
    """
    logging.info(f"Loading data from {data_path}")
    
    # Set a default batch size if None is provided
    if batch_size is None:
        batch_size = 8  # Default batch size
        logging.info(f"No batch size specified, using default: {batch_size}")
    
    # Load the data from pickle file
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}: {str(e)}")
        raise RuntimeError(f"Data loading failed: {str(e)}")
    
    # Get vocabulary and data splits
    char_to_idx = data["char_to_idx"]
    idx_to_char = data["idx_to_char"]
    train_inputs = data["train_inputs"]
    train_targets = data["train_targets"]
    val_inputs = data["val_inputs"]
    val_targets = data["val_targets"]
    
    # Validate data
    if not _validate_data(train_inputs, train_targets, val_inputs, val_targets, char_to_idx):
        logging.warning("Data validation detected potential issues")
    
    logging.info(f"Data loaded: {len(train_inputs)} training sequences, {len(val_inputs)} validation sequences")
    logging.info(f"Vocabulary size: {len(char_to_idx)} characters")
    
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

def _validate_data(train_inputs, train_targets, val_inputs, val_targets, char_to_idx) -> bool:
    """
    Validate data integrity to prevent training issues.
    
    Performs checks for:
    - Empty datasets
    - Mismatched input/target lengths
    - Out-of-vocabulary indices
    - Data type consistency
    
    Args:
        train_inputs: Training input sequences
        train_targets: Training target sequences
        val_inputs: Validation input sequences
        val_targets: Validation target sequences
        char_to_idx: Character to index mapping
        
    Returns:
        bool: True if data passes all validation checks
    """
    valid = True
    
    # Check for empty datasets
    if len(train_inputs) == 0:
        logging.error("Training dataset is empty")
        valid = False
    
    if len(val_inputs) == 0:
        logging.warning("Validation dataset is empty")
    
    # Check for mismatched lengths
    if len(train_inputs) != len(train_targets):
        logging.error(f"Training inputs ({len(train_inputs)}) and targets ({len(train_targets)}) have different lengths")
        valid = False
    
    if len(val_inputs) != len(val_targets):
        logging.error(f"Validation inputs ({len(val_inputs)}) and targets ({len(val_targets)}) have different lengths")
        valid = False
    
    # Check for vocabulary coverage (sample a few examples)
    vocab_size = len(char_to_idx)
    sample_size = min(100, len(train_inputs))
    
    for i in range(sample_size):
        # Check if any indices are out of vocabulary range
        if max(train_inputs[i]) >= vocab_size or max(train_targets[i]) >= vocab_size:
            logging.warning(f"Sample {i} contains out-of-vocabulary indices. Max allowed: {vocab_size-1}")
            valid = False
            break
    
    return valid

def recreate_data_loaders(dataset, batch_size, device_type, num_workers=0):
    """
    Recreate data loaders with a new batch size.
    
    This is useful when dynamically adjusting batch size during training.
    
    Args:
        dataset: Tuple of (train_dataset, val_dataset)
        batch_size: New batch size
        device_type: Device type (cuda or cpu)
        num_workers: Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset, val_dataset = dataset
    
    # Optimize settings
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    # Create train loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=(device_type == 'cuda'),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    
    # Use larger batch size for validation
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
    
    logging.info(f"Recreated data loaders with batch size {batch_size} (eval: {eval_batch_size})")
    
    return train_loader, val_loader

def get_sample_input_shape(dataloader):
    """
    Get the shape of a single input sample from the dataloader.
    
    Args:
        dataloader: DataLoader instance
        
    Returns:
        tuple: Shape of a single input sample
    """
    # Extract a single batch and get item shape
    for batch in dataloader:
        inputs, _ = batch
        # Return shape without batch dimension
        return tuple(inputs.shape[1:])
        
    # If dataloader is empty
    raise ValueError("Cannot determine sample shape: dataloader is empty") 