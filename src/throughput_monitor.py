#!/usr/bin/env python
"""
Throughput Monitoring Module
===========================

This module provides utilities for monitoring and optimizing training throughput 
including:

1. Real-time metrics collection and visualization
2. Performance bottleneck identification
3. Time profiling of training components
4. CUDA kernel analysis
"""

import time
import logging
import threading
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
            
            # Calculate percentages
            if avg_batch_time > 0:
                return {
                    "forward": (avg_forward_time / avg_batch_time) * 100,
                    "backward": (avg_backward_time / avg_batch_time) * 100,
                    "optimizer": (avg_optimizer_time / avg_batch_time) * 100,
                    "data_loading": (avg_data_loading_time / avg_batch_time) * 100,
                    "other": (other_time / avg_batch_time) * 100
                }
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            throughput = self.get_throughput()
            component_breakdown = self.get_component_breakdown()
            
            metrics = {
                "throughput": throughput,
                "throughput_history": self.recent_throughputs.copy() if self.recent_throughputs else [0],
                "peak_memory_mb": self.peak_memory,
                "total_samples": self.total_samples,
                "total_tokens": self.total_tokens,
                "component_breakdown": component_breakdown,
                "avg_batch_time": np.mean(self.batch_times) if self.batch_times else 0,
                "forward_ratio": component_breakdown.get("forward", 0),
                "backward_ratio": component_breakdown.get("backward", 0),
                "optimizer_ratio": component_breakdown.get("optimizer", 0),
                "data_loading_ratio": component_breakdown.get("data_loading", 0),
            }
            
            return metrics
    
    def plot_throughput_history(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot the history of throughput.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure if save_path is None, otherwise None
        """
        with self._lock:
            if not self.recent_throughputs:
                return None
                
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.recent_throughputs, marker='o', alpha=0.7)
            ax.set_title('Training Throughput History')
            ax.set_xlabel('Batch')
            ax.set_ylabel('Tokens per Second')
            ax.grid(True, alpha=0.3)
            
            # Add average line
            avg_throughput = np.mean(self.recent_throughputs)
            ax.axhline(y=avg_throughput, color='r', linestyle='--', 
                      label=f'Average: {avg_throughput:.1f} tokens/sec')
            ax.legend()
            
            if save_path:
                plt.savefig(save_path)
                plt.close(fig)
                return None
            return fig
    
    def plot_component_breakdown(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot the breakdown of time spent in different components.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            
        Returns:
            Matplotlib figure if save_path is None, otherwise None
        """
        with self._lock:
            breakdown = self.get_component_breakdown()
            if not breakdown:
                return None
                
            # Extract components and percentages
            components = list(breakdown.keys())
            percentages = list(breakdown.values())
            
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                percentages, 
                labels=components,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Style the chart
            ax.set_title('Time Breakdown by Component')
            plt.setp(autotexts, size=10, weight="bold")
            
            if save_path:
                plt.savefig(save_path)
                plt.close(fig)
                return None
            return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a text report of all metrics.
        
        Args:
            save_path: Path to save the report (if None, report is not saved)
            
        Returns:
            Report as a string
        """
        with self._lock:
            metrics = self.get_metrics()
            breakdown = metrics["component_breakdown"]
            
            report = (
                "==== THROUGHPUT MONITORING REPORT ====\n\n"
                f"Average Throughput: {metrics['throughput']:.2f} tokens/second\n"
                f"Peak Memory Usage: {metrics['peak_memory_mb']:.2f} MB\n"
                f"Total Samples Processed: {metrics['total_samples']:,}\n"
                f"Total Tokens Processed: {metrics['total_tokens']:,}\n\n"
                "Component Time Breakdown:\n"
            )
            
            # Add component breakdown
            for component, percentage in breakdown.items():
                report += f"  - {component.capitalize()}: {percentage:.2f}%\n"
            
            # Add bottleneck identification
            bottleneck = max(breakdown.items(), key=lambda x: x[1]) if breakdown else (None, 0)
            if bottleneck[0]:
                report += f"\nPrimary Bottleneck: {bottleneck[0].capitalize()} ({bottleneck[1]:.2f}%)\n"
                
                # Add specific recommendations based on bottleneck
                if bottleneck[0] == "data_loading":
                    report += (
                        "\nRecommendations for Data Loading Bottleneck:\n"
                        "  - Increase num_workers in DataLoader\n"
                        "  - Use GPU pinned memory for faster CPU-GPU transfer\n"
                        "  - Pre-fetch data to overlap with computation\n"
                    )
                elif bottleneck[0] == "forward" or bottleneck[0] == "backward":
                    report += (
                        "\nRecommendations for Computation Bottleneck:\n"
                        "  - Try using Automatic Mixed Precision (AMP)\n"
                        "  - Optimize batch size for your GPU\n"
                        "  - Enable CUDA graphs for static computation\n"
                    )
                elif bottleneck[0] == "optimizer":
                    report += (
                        "\nRecommendations for Optimizer Bottleneck:\n"
                        "  - Try using gradient accumulation\n"
                        "  - Consider using a different optimizer\n"
                        "  - Reduce weight decay or learning rate\n"
                    )
            
            # Save report if requested
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report)
            
            return report

def attach_to_trainer(trainer: Any, monitor: ThroughputMonitor) -> None:
    """
    Attach throughput monitor to a trainer by wrapping the necessary methods.
    
    Args:
        trainer: The trainer object
        monitor: ThroughputMonitor instance
    """
    # Store original methods
    original_forward = trainer._forward_step
    original_backward = trainer._backward_step
    original_optimizer = trainer._optimizer_step
    original_train_epoch = trainer.train_epoch
    
    # Wrap methods to include monitoring
    def wrapped_forward(self, *args, **kwargs):
        batch_size = args[0].size(0) if args else kwargs.get('input_ids', kwargs.get('inputs', None)).size(0)
        seq_length = args[0].size(1) if args else kwargs.get('input_ids', kwargs.get('inputs', None)).size(1)
        
        monitor.start_forward(batch_size, seq_length)
        result = original_forward(self, *args, **kwargs)
        monitor.end_forward()
        return result
    
    def wrapped_backward(self, *args, **kwargs):
        monitor.start_backward()
        result = original_backward(self, *args, **kwargs)
        monitor.end_backward()
        return result
    
    def wrapped_optimizer(self, *args, **kwargs):
        monitor.start_optimizer()
        result = original_optimizer(self, *args, **kwargs)
        monitor.end_optimizer()
        return result
    
    def wrapped_train_epoch(self, *args, **kwargs):
        # Reset monitor for new epoch
        monitor.reset()
        result = original_train_epoch(self, *args, **kwargs)
        return result
    
    # Replace methods with wrapped versions
    trainer._forward_step = wrapped_forward.__get__(trainer, type(trainer))
    trainer._backward_step = wrapped_backward.__get__(trainer, type(trainer))
    trainer._optimizer_step = wrapped_optimizer.__get__(trainer, type(trainer))
    trainer.train_epoch = wrapped_train_epoch.__get__(trainer, type(trainer))

def attach_to_dataloader(dataloader: torch.utils.data.DataLoader, monitor: ThroughputMonitor) -> torch.utils.data.DataLoader:
    """
    Wrap a dataloader to monitor data loading time.
    
    Args:
        dataloader: PyTorch DataLoader
        monitor: ThroughputMonitor instance
        
    Returns:
        Wrapped dataloader
    """
    # Create a wrapper iterator class
    class MonitoredDataLoaderIter:
        def __init__(self, dataloader_iter, monitor):
            self.dataloader_iter = dataloader_iter
            self.monitor = monitor
        
        def __iter__(self):
            return self
        
        def __next__(self):
            monitor.start_data_loading()
            try:
                batch = next(self.dataloader_iter)
                monitor.end_data_loading()
                return batch
            except StopIteration:
                monitor.end_data_loading()
                raise
    
    # Create a wrapper dataloader class
    class MonitoredDataLoader(torch.utils.data.DataLoader):
        def __init__(self, dataloader, monitor):
            self.dataloader = dataloader
            self.monitor = monitor
            # Copy all attributes from original dataloader
            for attr_name in dir(dataloader):
                if not attr_name.startswith('__') and not callable(getattr(dataloader, attr_name)):
                    setattr(self, attr_name, getattr(dataloader, attr_name))
        
        def __iter__(self):
            return MonitoredDataLoaderIter(iter(self.dataloader), self.monitor)
        
        def __len__(self):
            return len(self.dataloader)
    
    return MonitoredDataLoader(dataloader, monitor)

def profile_model(
    model: torch.nn.Module, 
    input_shape: Tuple[int, ...], 
    device: torch.device,
    n_runs: int = 10
) -> Dict[str, float]:
    """
    Profile a model's forward and backward pass performance.
    
    Args:
        model: PyTorch model to profile
        input_shape: Shape of input tensor
        device: Device to run on
        n_runs: Number of runs for averaging
        
    Returns:
        Dictionary with profiling results
    """
    model.to(device)
    model.train()
    
    # Create random input - use Long type for transformer models with embeddings
    # For character-level models, generate random indices within vocab size range
    vocab_size = 89  # Default vocab size, should be updated in real applications
    x = torch.randint(0, vocab_size, input_shape, device=device, dtype=torch.long)
    
    # Warmup
    for _ in range(3):
        y = model(x)
        loss = y.sum()
        loss.backward()
    
    # Synchronize before measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure forward pass
    forward_times = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        y = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        forward_times.append(time.time() - start)
    
    # Measure backward pass
    backward_times = []
    for _ in range(n_runs):
        y = model(x)
        loss = y.sum()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        loss.backward()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        backward_times.append(time.time() - start)
        
        # Clear gradients
        for param in model.parameters():
            param.grad = None
    
    # Calculate memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_allocated = 0
    
    # Calculate average times
    avg_forward = sum(forward_times) / n_runs
    avg_backward = sum(backward_times) / n_runs
    
    # Calculate throughput (assuming batch_size is the first dimension)
    batch_size = input_shape[0]
    seq_length = input_shape[1] if len(input_shape) > 1 else 1
    tokens_per_run = batch_size * seq_length
    throughput = tokens_per_run / (avg_forward + avg_backward)
    
    return {
        "avg_forward_time": avg_forward,
        "avg_backward_time": avg_backward,
        "total_time_per_batch": avg_forward + avg_backward,
        "memory_allocated_mb": memory_allocated,
        "estimated_throughput": throughput,
        "forward_backward_ratio": avg_forward / avg_backward if avg_backward > 0 else float('inf'),
    }

def print_throughput_report(monitor: ThroughputMonitor) -> None:
    """
    Print a nicely formatted throughput report to console.
    
    Args:
        monitor: ThroughputMonitor instance
    """
    metrics = monitor.get_metrics()
    breakdown = metrics["component_breakdown"]
    
    print("\n" + "="*60)
    print(" "*20 + "THROUGHPUT REPORT")
    print("="*60)
    
    print(f"\nAverage Throughput: {metrics['throughput']:.2f} tokens/second")
    print(f"Peak Memory Usage: {metrics['peak_memory_mb']:.2f} MB")
    
    # Print component breakdown
    print("\nTime Breakdown:")
    for component, percentage in breakdown.items():
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        print(f"  {component.capitalize():10s}: {percentage:5.1f}% {bar}")
    
    # Find bottleneck
    bottleneck = max(breakdown.items(), key=lambda x: x[1]) if breakdown else (None, 0)
    if bottleneck[0]:
        print(f"\nPrimary Bottleneck: {bottleneck[0].capitalize()} ({bottleneck[1]:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile model throughput")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for profiling")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length for profiling")
    parser.add_argument("--model_type", type=str, default="transformer", help="Model type to profile")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs for profiling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run profiling on")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a model for profiling
    if args.model_type == "transformer":
        from src.model import create_transformer_model
        
        model = create_transformer_model(
            vocab_size=89,  # Example vocab size
            max_seq_length=args.seq_length,
            d_model=768,
            n_head=12,
            d_hid=3072,
            n_layers=12,
            dropout=0.1
        )
    else:
        logging.error(f"Unknown model type: {args.model_type}")
        exit(1)
    
    # Set device
    device = torch.device(args.device)
    
    # Profile the model
    input_shape = (args.batch_size, args.seq_length)
    logging.info(f"Profiling model with input shape: {input_shape} on {device}")
    
    results = profile_model(model, input_shape, device, n_runs=args.n_runs)
    
    # Print results
    print("\n" + "="*60)
    print(" "*20 + "MODEL PROFILING RESULTS")
    print("="*60)
    
    print(f"\nModel Type: {args.model_type}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Device: {device}")
    
    print(f"\nAverage Forward Time: {results['avg_forward_time']*1000:.2f} ms")
    print(f"Average Backward Time: {results['avg_backward_time']*1000:.2f} ms")
    print(f"Total Time per Batch: {results['total_time_per_batch']*1000:.2f} ms")
    print(f"Memory Usage: {results['memory_allocated_mb']:.2f} MB")
    print(f"Estimated Throughput: {results['estimated_throughput']:.2f} tokens/second")
    print(f"Forward/Backward Ratio: {results['forward_backward_ratio']:.2f}")
    
    print("\nTo improve throughput:")
    print("1. Try different batch sizes")
    print("2. Enable mixed precision training")
    print("3. Optimize model architecture")
    print("4. Adjust gradient accumulation steps")
    
    print("="*60) 