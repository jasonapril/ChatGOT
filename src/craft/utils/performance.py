"""Performance monitoring utilities for tracking resource usage and performance."""

from typing import Dict, Optional

# from ..performance.instrumentation import setup_instrumentation # No longer needed
# Only import ThroughputMonitor, as the others don't exist as standalone functions
from ..performance.throughput_core import ThroughputMonitor
# from ..performance.throughput_core import (
#     ThroughputMonitor,
#     measure_throughput,
#     calculate_performance_stats
# )

# Remove the setup_monitoring function as it calls a non-existent function
# def setup_monitoring(config: Dict) -> None:
#     """Set up performance monitoring based on configuration."""
#     # Import inside the function
#     from ..performance.instrumentation import setup_instrumentation
#     return setup_instrumentation(config)

def get_resource_metrics(include_gpu: bool = True) -> Dict:
    """Get current resource usage metrics."""
    import psutil
    import torch
    
    # Call virtual_memory once and store the result
    vmem = psutil.virtual_memory()

    metrics = {
        'cpu_percent': psutil.cpu_percent(),
        # Use the stored result
        'memory_percent': vmem.percent,
        'memory_used_gb': vmem.used / (1024**3),
    }
    
    if include_gpu and torch.cuda.is_available():
        metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / (1024**3)
        # metrics['gpu_utilization'] = torch.cuda.utilization() # May need specific library like pynvml
    
    return metrics 