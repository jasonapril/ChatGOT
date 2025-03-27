"""Performance monitoring utilities for tracking resource usage and performance."""

from typing import Dict, Optional

from ..performance.instrumentation import setup_instrumentation
from ..performance.throughput_core import (
    ThroughputMonitor,
    measure_throughput,
    calculate_performance_stats
)

def setup_monitoring(config: Dict) -> None:
    """Set up performance monitoring based on configuration."""
    return setup_instrumentation(config)

def get_resource_metrics(include_gpu: bool = True) -> Dict:
    """Get current resource usage metrics."""
    import psutil
    import torch
    
    metrics = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
    }
    
    if include_gpu and torch.cuda.is_available():
        metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
        metrics['gpu_utilization'] = torch.cuda.utilization()
    
    return metrics 