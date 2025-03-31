#!/usr/bin/env python
"""
Throughput Core Monitoring Module
=================================

This module provides core functionality for monitoring training throughput including:

1. Batch and iteration timing
2. Memory usage tracking
3. Component-level timing breakdown
4. Thread-safe metric collection

This is designed to be used for analyzing and optimizing training performance.
"""

import time
import logging
import threading
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

class ThroughputMonitor:
    """Monitors training throughput and identifies bottlenecks."""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize the throughput monitor.
        
        Args:
            window_size: Number of batches to average over for metrics
        """
        self.window_size = window_size
        self.reset()
        
        # Set up threading lock for thread-safe updates
        self._lock = threading.Lock()
        
        # Set up CUDA event tracking if available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def reset(self) -> None:
        """Reset all metrics and counters."""
        self.batch_times = []
        self.tokens_per_batch = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.data_loading_times = []
        self.current_batch_start = None
        self.current_forward_start = None
        self.current_backward_start = None
        self.current_optimizer_start = None
        self.current_data_loading_start = None
        self.total_samples = 0
        self.total_tokens = 0
        self.peak_memory = 0.0
        self.recent_throughputs = []
    
    def start_batch(self, batch_size: int, seq_length: int) -> None:
        """
        Mark the start of a new batch.
        
        Args:
            batch_size: Number of samples in the batch
            seq_length: Length of each sequence in the batch
        """
        with self._lock:
            self.current_batch_start = time.time()
            self.total_samples += batch_size
            self.total_tokens += batch_size * seq_length
            
            # Record CUDA event if available
            if self.cuda_available:
                self.start_event.record()
    
    def end_batch(self) -> None:
        """Mark the end of the current batch and record metrics."""
        with self._lock:
            if self.current_batch_start is None:
                return
                
            # Record CUDA event and synchronize if available
            if self.cuda_available:
                self.end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = self.start_event.elapsed_time(self.end_event)
                batch_time = elapsed_ms / 1000.0  # Convert to seconds
            else:
                batch_time = time.time() - self.current_batch_start
            
            self.batch_times.append(batch_time)
            
            # Keep only the most recent window_size batches
            if len(self.batch_times) > self.window_size:
                self.batch_times = self.batch_times[-self.window_size:]
            
            # Calculate tokens processed in this batch
            if self.tokens_per_batch:
                tokens_in_batch = self.tokens_per_batch[-1]
                self.recent_throughputs.append(tokens_in_batch / batch_time)
                
                # Keep only the most recent window_size throughputs
                if len(self.recent_throughputs) > self.window_size:
                    self.recent_throughputs = self.recent_throughputs[-self.window_size:]
            
            # Reset batch start time
            self.current_batch_start = None
            
            # Update peak memory usage if CUDA is available
            if self.cuda_available:
                current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                self.peak_memory = max(self.peak_memory, current_memory)
    
    def start_data_loading(self) -> None:
        """Mark the start of data loading."""
        with self._lock:
            self.current_data_loading_start = time.time()
    
    def end_data_loading(self) -> None:
        """Mark the end of data loading and record time."""
        with self._lock:
            if self.current_data_loading_start is None:
                return
                
            load_time = time.time() - self.current_data_loading_start
            self.data_loading_times.append(load_time)
            
            # Keep only the most recent window_size times
            if len(self.data_loading_times) > self.window_size:
                self.data_loading_times = self.data_loading_times[-self.window_size:]
            
            self.current_data_loading_start = None
    
    def start_forward(self, batch_size: int, seq_length: int) -> None:
        """
        Mark the start of forward pass.
        
        Args:
            batch_size: Number of samples in the batch
            seq_length: Length of each sequence in the batch
        """
        with self._lock:
            self.current_forward_start = time.time()
            self.tokens_per_batch.append(batch_size * seq_length)
            
            # Keep only the most recent window_size batch sizes
            if len(self.tokens_per_batch) > self.window_size:
                self.tokens_per_batch = self.tokens_per_batch[-self.window_size:]
    
    def end_forward(self) -> None:
        """Mark the end of forward pass and record time."""
        with self._lock:
            if self.current_forward_start is None:
                return
                
            forward_time = time.time() - self.current_forward_start
            self.forward_times.append(forward_time)
            
            # Keep only the most recent window_size times
            if len(self.forward_times) > self.window_size:
                self.forward_times = self.forward_times[-self.window_size:]
            
            self.current_forward_start = None
    
    def start_backward(self) -> None:
        """Mark the start of backward pass."""
        with self._lock:
            self.current_backward_start = time.time()
    
    def end_backward(self) -> None:
        """Mark the end of backward pass and record time."""
        with self._lock:
            if self.current_backward_start is None:
                return
                
            backward_time = time.time() - self.current_backward_start
            self.backward_times.append(backward_time)
            
            # Keep only the most recent window_size times
            if len(self.backward_times) > self.window_size:
                self.backward_times = self.backward_times[-self.window_size:]
            
            self.current_backward_start = None
    
    def start_optimizer(self) -> None:
        """Mark the start of optimizer step."""
        with self._lock:
            self.current_optimizer_start = time.time()
    
    def end_optimizer(self) -> None:
        """Mark the end of optimizer step and record time."""
        with self._lock:
            if self.current_optimizer_start is None:
                return
                
            optimizer_time = time.time() - self.current_optimizer_start
            self.optimizer_times.append(optimizer_time)
            
            # Keep only the most recent window_size times
            if len(self.optimizer_times) > self.window_size:
                self.optimizer_times = self.optimizer_times[-self.window_size:]
            
            self.current_optimizer_start = None
    
    def get_throughput(self) -> float:
        """
        Calculate the average throughput in tokens per second.
        
        Returns:
            Average throughput (tokens/sec)
        """
        with self._lock:
            if not self.batch_times or not self.tokens_per_batch:
                return 0.0
                
            # Calculate average tokens per second over the window
            avg_tokens = np.mean(self.tokens_per_batch[-len(self.batch_times):])
            avg_time = np.mean(self.batch_times)
            
            if avg_time > 0:
                return avg_tokens / avg_time
            return 0.0
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """
        Get the breakdown of time spent in different components as percentages.
        
        Returns:
            Dictionary with component names and their percentage of total time
        """
        with self._lock:
            if not self.batch_times:
                return {}
                
            # Calculate average times for each component
            avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
            avg_forward_time = np.mean(self.forward_times) if self.forward_times else 0
            avg_backward_time = np.mean(self.backward_times) if self.backward_times else 0
            avg_optimizer_time = np.mean(self.optimizer_times) if self.optimizer_times else 0
            avg_data_loading_time = np.mean(self.data_loading_times) if self.data_loading_times else 0
            
            # Calculate other time (e.g., overhead, CPU-GPU sync)
            other_time = max(0, avg_batch_time - (avg_forward_time + avg_backward_time + avg_optimizer_time))
            
            # Calculate total time for percentage calculation
            total_time = avg_batch_time
            
            if total_time <= 0:
                return {}
                
            # Calculate percentages
            return {
                'forward': (avg_forward_time / total_time) * 100,
                'backward': (avg_backward_time / total_time) * 100,
                'optimizer': (avg_optimizer_time / total_time) * 100,
                'data_loading': (avg_data_loading_time / total_time) * 100,
                'other': (other_time / total_time) * 100
            }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        with self._lock:
            # Check CUDA availability FIRST
            if not self.cuda_available:
                return {
                    'allocated': 0.0,
                    'reserved': 0.0,
                    'peak': self.peak_memory # Keep peak even if CUDA is off now
                }
            
            # If CUDA is available, get stats
            try:
                allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                # Peak memory is updated in end_batch, just retrieve it here
                return {
                    'allocated': allocated_mb,
                    'reserved': reserved_mb,
                    'peak': self.peak_memory
                }
            except RuntimeError as e:
                 # Handle potential errors during CUDA calls (e.g., during shutdown)
                 logging.warning(f"Error getting CUDA memory stats: {e}")
            return {
                    'allocated': 0.0,
                    'reserved': 0.0,
                'peak': self.peak_memory
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a complete summary of all monitored metrics.
        
        Returns:
            Dictionary with all metrics
        """
        # Avoid using lock to prevent potential deadlocks
        # Copy the data we need to avoid race conditions
        batch_times = self.batch_times.copy() if self.batch_times else []
        recent_throughputs = self.recent_throughputs.copy() if self.recent_throughputs else []
        total_samples = self.total_samples
        total_tokens = self.total_tokens
        
        # Calculate metrics outside of lock
        try:
            # Handle throughput calculation
            throughput = 0.0
            if batch_times and recent_throughputs:
                throughput = float(sum(recent_throughputs) / len(recent_throughputs)) if recent_throughputs else 0.0
            
            # Handle component breakdown
            component_breakdown = {}
            if batch_times:
                # Calculate average times for each component
                avg_forward_time = np.mean(self.forward_times) if self.forward_times else 0
                avg_backward_time = np.mean(self.backward_times) if self.backward_times else 0
                avg_optimizer_time = np.mean(self.optimizer_times) if self.optimizer_times else 0
                avg_data_loading_time = np.mean(self.data_loading_times) if self.data_loading_times else 0
                avg_batch_time = np.mean(batch_times)
                
                # Calculate other time
                other_time = max(0, avg_batch_time - (avg_forward_time + avg_backward_time + avg_optimizer_time))
                
                # Calculate percentages
                if avg_batch_time > 0:
                    component_breakdown = {
                        'forward': float((avg_forward_time / avg_batch_time) * 100),
                        'backward': float((avg_backward_time / avg_batch_time) * 100),
                        'optimizer': float((avg_optimizer_time / avg_batch_time) * 100),
                        'data_loading': float((avg_data_loading_time / avg_batch_time) * 100),
                        'other': float((other_time / avg_batch_time) * 100)
                    }
            
            # Handle memory stats
            if self.cuda_available:
                memory_stats = {
                    'allocated': float(torch.cuda.memory_allocated() / (1024 ** 2)),
                    'reserved': float(torch.cuda.memory_reserved() / (1024 ** 2)),
                    'peak': float(self.peak_memory)
                }
            else:
                memory_stats = {
                    'allocated': 0.0,
                    'reserved': 0.0,
                    'peak': float(self.peak_memory)
                }
            
            # Calculate averages and stdevs
            avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0
            std_batch_time = float(np.std(batch_times)) if len(batch_times) > 1 else 0.0
            
            # Ensure all values are serializable (not numpy types)
            return {
                'throughput': float(throughput),
                'component_breakdown': component_breakdown,
                'memory': memory_stats,
                'total_samples': int(total_samples),
                'total_tokens': int(total_tokens),
                'avg_batch_time': float(avg_batch_time),
                'std_batch_time': float(std_batch_time),
                'throughput_history': [float(t) for t in recent_throughputs]
            }
        except Exception as e:
            # If anything fails, return a minimal summary to prevent complete failure
            import logging
            logging.error(f"Error creating summary: {e}")
            return {
                'throughput': 0.0,
                'component_breakdown': {},
                'memory': {'allocated': 0.0, 'reserved': 0.0, 'peak': 0.0},
                'total_samples': int(total_samples),
                'total_tokens': int(total_tokens),
                'avg_batch_time': 0.0,
                'std_batch_time': 0.0,
                'throughput_history': []
            } 