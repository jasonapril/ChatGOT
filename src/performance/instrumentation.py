#!/usr/bin/env python
"""
Monitoring Instrumentation Module
===============================

This module provides instrumentation utilities for PyTorch models and dataloaders:

1. Context managers for measuring component performance
2. Model hooks for tracking forward/backward pass metrics
3. Dataloader wrappers for measuring data loading performance
4. Integration with throughput monitoring

These tools help identify bottlenecks in the training pipeline.
"""

import time
import logging
import os
import json
import torch
import torch.nn as nn
import contextlib
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import threading

from src.performance.throughput_core import ThroughputMonitor

class InstrumentedModel(nn.Module):
    """Wrapper for a PyTorch model that instruments forward and backward passes."""
    
    def __init__(self, model: nn.Module, monitor: ThroughputMonitor):
        """
        Initialize the instrumented model.
        
        Args:
            model: PyTorch model to instrument
            monitor: ThroughputMonitor instance
        """
        super().__init__()
        self.model = model
        self.monitor = monitor
        self._batch_size = 0
        self._seq_length = 0
        
        # Register backward hook
        self._backward_hooks = []
        
    def extract_batch_info(self, x: torch.Tensor) -> Tuple[int, int]:
        """
        Extract batch size and sequence length from input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (batch_size, seq_length)
        """
        # Default implementation assumes shape [batch_size, seq_length, ...]
        if isinstance(x, torch.Tensor):
            if len(x.shape) >= 2:
                return x.shape[0], x.shape[1]
            elif len(x.shape) == 1:
                return 1, x.shape[0]
        elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return self.extract_batch_info(x[0])
            
        # Default fallback
        return 1, 1
        
    def forward(self, *args, **kwargs):
        """
        Instrumented forward pass.
        
        Args:
            *args: Arguments for the model's forward method
            **kwargs: Keyword arguments for the model's forward method
            
        Returns:
            Model outputs
        """
        # Extract batch information from input
        if args and isinstance(args[0], torch.Tensor):
            self._batch_size, self._seq_length = self.extract_batch_info(args[0])
        elif kwargs and 'input_ids' in kwargs and isinstance(kwargs['input_ids'], torch.Tensor):
            self._batch_size, self._seq_length = self.extract_batch_info(kwargs['input_ids'])
        
        # Start timing the forward pass
        self.monitor.start_forward(self._batch_size, self._seq_length)
        
        # Run forward pass
        outputs = self.model(*args, **kwargs)
        
        # End timing the forward pass
        self.monitor.end_forward()
        
        # Register backward hook on output tensor
        if isinstance(outputs, torch.Tensor) and outputs.requires_grad:
            outputs.register_hook(self._backward_hook)
        elif isinstance(outputs, tuple) and all(isinstance(o, torch.Tensor) for o in outputs):
            for o in outputs:
                if o.requires_grad:
                    o.register_hook(self._backward_hook)
        
        return outputs
    
    def _backward_hook(self, grad: torch.Tensor) -> None:
        """
        Hook for backward pass timing.
        
        Args:
            grad: Gradient tensor
        """
        # Only track once per backward pass by checking if we're already tracking
        if not hasattr(self, '_backward_started') or not self._backward_started:
            self._backward_started = True
            self.monitor.start_backward()
            
            # Schedule end_backward to run at the end of the backward pass
            def on_backward_end(*args, **kwargs):
                self.monitor.end_backward()
                self._backward_started = False
            
            # Register this to run after the backward pass completes
            torch._C._FunctionBase.register_hook(torch.autograd.function._SingleLevelFunction, on_backward_end)

@contextlib.contextmanager
def measure_batch(monitor: ThroughputMonitor, batch_size: int, seq_length: int):
    """
    Context manager for measuring batch processing time.
    
    Args:
        monitor: ThroughputMonitor instance
        batch_size: Number of samples in the batch
        seq_length: Length of each sequence in the batch
    """
    try:
        monitor.start_batch(batch_size, seq_length)
        yield
    finally:
        monitor.end_batch()

@contextlib.contextmanager
def measure_optimizer_step(monitor: ThroughputMonitor):
    """
    Context manager for measuring optimizer step time.
    
    Args:
        monitor: ThroughputMonitor instance
    """
    try:
        monitor.start_optimizer()
        yield
    finally:
        monitor.end_optimizer()

@contextlib.contextmanager
def measure_data_loading(monitor: ThroughputMonitor):
    """
    Context manager for measuring data loading time.
    
    Args:
        monitor: ThroughputMonitor instance
    """
    try:
        monitor.start_data_loading()
        yield
    finally:
        monitor.end_data_loading()

class InstrumentedDataLoader:
    """Wrapper for a PyTorch DataLoader that measures data loading time."""
    
    def __init__(self, dataloader: torch.utils.data.DataLoader, monitor: ThroughputMonitor):
        """
        Initialize the instrumented dataloader.
        
        Args:
            dataloader: DataLoader to instrument
            monitor: ThroughputMonitor instance
        """
        self.dataloader = dataloader
        self.monitor = monitor
        
        # Copy attributes from the original dataloader
        self.batch_size = dataloader.batch_size
        self.dataset = dataloader.dataset
        
    def __iter__(self):
        """Instrumented iterator."""
        iterator = iter(self.dataloader)
        
        # Create a wrapper iterator that measures time
        def instrumented_iterator():
            while True:
                try:
                    # Measure data loading time
                    with measure_data_loading(self.monitor):
                        batch = next(iterator)
                    yield batch
                except StopIteration:
                    break
        
        return instrumented_iterator()
    
    def __len__(self):
        """Return the length of the dataloader."""
        return len(self.dataloader)

def create_instrumented_model(model: nn.Module, monitor: Optional[ThroughputMonitor] = None) -> InstrumentedModel:
    """
    Create an instrumented model.
    
    Args:
        model: PyTorch model to instrument
        monitor: Optional ThroughputMonitor instance (will create if None)
        
    Returns:
        Instrumented model
    """
    if monitor is None:
        monitor = ThroughputMonitor()
    
    return InstrumentedModel(model, monitor)

def create_instrumented_dataloader(dataloader: torch.utils.data.DataLoader, 
                                  monitor: Optional[ThroughputMonitor] = None) -> InstrumentedDataLoader:
    """
    Create an instrumented dataloader.
    
    Args:
        dataloader: PyTorch DataLoader to instrument
        monitor: Optional ThroughputMonitor instance (will create if None)
        
    Returns:
        Instrumented dataloader
    """
    if monitor is None:
        monitor = ThroughputMonitor()
    
    return InstrumentedDataLoader(dataloader, monitor)

def save_monitor_stats(monitor: ThroughputMonitor, filepath: str, timeout: float = 5.0, 
                      use_threading: bool = True) -> bool:
    """
    Save monitor statistics to a JSON file with timeout.
    
    Args:
        monitor: ThroughputMonitor instance
        filepath: Path to save the statistics
        timeout: Maximum time in seconds to wait for file operations
        use_threading: Whether to use threading (for testing, set to False)
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Function to actually save the stats
    def _save_stats():
        try:
            # Create a summary dictionary that can be serialized to JSON
            summary = monitor.get_summary()
            
            # Remove non-serializable objects
            if 'throughput_history' in summary:
                summary['throughput_history'] = [float(t) for t in summary['throughput_history']]
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filepath))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save to file with explicit close to avoid hanging
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
                
            logging.info(f"Monitor statistics saved to {filepath}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving monitor statistics: {e}"
            logging.warning(error_msg)
            return False
    
    # For testing, allow direct execution without threading
    if not use_threading:
        return _save_stats()
    
    # Normal threaded execution path
    import threading
    
    # Use a dictionary to store thread results (needed to pass by reference)
    result = {"success": False}
    
    def _save_with_timeout():
        result["success"] = _save_stats()
    
    # Run the save operation in a separate thread with a timeout
    save_thread = threading.Thread(target=_save_with_timeout)
    save_thread.daemon = True
    save_thread.start()
    
    # Wait for the thread to complete or timeout
    save_thread.join(timeout)
    
    # Check if the thread is still running (timeout occurred)
    if save_thread.is_alive():
        logging.error(f"Saving monitor statistics timed out after {timeout} seconds")
        return False
    
    # Return the success status from the thread
    return result["success"] 