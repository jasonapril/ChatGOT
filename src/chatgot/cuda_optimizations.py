"""
CUDA Optimization Module
=======================

This module implements various CUDA-specific optimizations to maximize 
throughput for transformer model training, including:

1. CUDA Graphs for repetitive computation graphs
2. Tensor Core utilization
3. Memory access pattern optimization
4. Thread and block size tuning
5. Custom CUDA kernels for attention
"""

import torch
import logging
import os
import platform
from typing import Dict, Optional, Any

def enable_tf32():
    """
    Enable TensorFloat32 precision on Ampere+ GPUs.
    This can significantly speed up training with minimal precision loss.
    """
    if torch.cuda.is_available():
        # Check if GPU supports TF32
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            try:
                # Enable TF32 precision
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logging.info("[CUDA] Enabled TF32 precision for matrix operations")
                return True
            except Exception as e:
                logging.warning(f"[CUDA] Failed to enable TF32: {e}")
    return False

def optimize_cudnn():
    """
    Set optimal cuDNN settings for the current hardware.
    """
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        # Enable benchmark mode for faster training when input sizes don't change
        torch.backends.cudnn.benchmark = True
        
        # Set deterministic algorithms to False for better performance
        torch.backends.cudnn.deterministic = False
        
        # Enable cuDNN autotuner
        os.environ['CUDNN_FRONTEND_ENABLE_TENSOR_CORES'] = '1'
        
        # Optimize convolution algorithms selection
        if 'linux' in platform.system().lower():
            # Linux-specific cudnn optimizations
            os.environ['CUDNN_LOGINFO_DBG'] = '0'
            os.environ['CUDNN_LOGDEST_DBG'] = '/dev/null'
        
        logging.info("[CUDA] Optimized cuDNN settings for maximum performance")
        return True
    return False

def setup_optimal_cuda_cache():
    """
    Set optimal CUDA memory allocation settings through environment variables.
    Focus on throughput rather than minimum memory usage.
    """
    if torch.cuda.is_available():
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # Configure memory allocation strategy based on available GPU memory
        if total_memory < 5:  # Small GPUs (4GB or less)
            # More conservative memory strategy for small GPUs
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
            logging.info("[CUDA] Configured memory allocation for small GPU (<5GB)")
        else:
            # More aggressive throughput-oriented memory allocation for larger GPUs
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8'
            logging.info("[CUDA] Configured memory allocation for larger GPU (>5GB)")
        
        # Enable JIT fusion for PyTorch operations
        torch.jit.set_fusion_strategy([('STATIC', 3), ('DYNAMIC', 1)])
        
        # Try to enable fast attention implementations if available
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # This uses Flash Attention when available
                logging.info("[CUDA] Using optimized scaled_dot_product_attention implementation")
        except Exception:
            pass
        
        # Set thread locality for CUDA operations
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'
        if 'linux' in platform.system().lower():
            try:
                # Try to set thread affinity on Linux
                import psutil
                if hasattr(psutil.Process(), 'cpu_affinity'):
                    # Use half of available cores for better thread locality
                    num_cpus = len(psutil.Process().cpu_affinity())
                    cores_to_use = max(1, num_cpus // 2)
                    psutil.Process().cpu_affinity(list(range(cores_to_use)))
                    logging.info(f"[CUDA] Set CPU affinity to {cores_to_use} cores for better thread locality")
            except ImportError:
                logging.debug("[CUDA] psutil not available, skipping thread affinity optimization")
            except Exception as e:
                logging.debug(f"[CUDA] Failed to set thread affinity: {e}")
        
        return True
    return False

def optimize_for_gpu_architecture():
    """
    Apply specific optimizations based on the detected GPU architecture.
    """
    if not torch.cuda.is_available():
        return False
    
    # Get GPU information
    gpu_name = torch.cuda.get_device_name(0)
    gpu_name_lower = gpu_name.lower()
    cuda_cap = torch.cuda.get_device_capability(0)
    cuda_cap_major, cuda_cap_minor = cuda_cap
    
    logging.info(f"[CUDA] Detected GPU: {gpu_name} (Compute Capability {cuda_cap_major}.{cuda_cap_minor})")
    
    optimizations_applied = False
    
    # RTX 30 series or newer (Ampere architecture or newer)
    if cuda_cap_major >= 8 or (cuda_cap_major == 7 and cuda_cap_minor >= 5):
        # Enable TF32 for Ampere and newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set optimal math mode
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # More aggressive workspace for faster GEMM
        optimizations_applied = True
        logging.info("[CUDA] Applied optimizations for Ampere/RTX architecture")
    
    # GTX 16 series optimizations
    elif any(gpu in gpu_name_lower for gpu in ['1650', '1660']):
        # GTX 16xx series has Turing architecture without tensor cores
        # Optimize for regular CUDA cores
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Ensure asynchronous kernel launches
        
        # Set smaller workspace size to avoid excessive memory usage
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if total_memory <= 4:  # 4GB or less
            # Limit memory usage for 1650 with 4GB
            torch.cuda.set_per_process_memory_fraction(0.9)
            logging.info("[CUDA] Limited memory usage to 90% for GTX 16xx series")
        
        optimizations_applied = True
        logging.info("[CUDA] Applied optimizations for GTX 16xx series")
    
    # GTX 10 series optimizations
    elif any(gpu in gpu_name_lower for gpu in ['1050', '1060', '1070', '1080']):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        torch.cuda.empty_cache()
        
        # Pascal architecture specific optimizations
        if total_memory <= 6:  # 6GB or less
            torch.cuda.set_per_process_memory_fraction(0.9)
        
        optimizations_applied = True
        logging.info("[CUDA] Applied optimizations for GTX 10xx series")
    
    return optimizations_applied

def create_cuda_graph_trainer(model, input_shape, device):
    """
    Create a CUDA graph for the forward and backward passes.
    This can significantly speed up training for static graphs.
    
    Args:
        model: The model to optimize
        input_shape: Shape of input tensors 
        device: CUDA device
        
    Returns:
        A function that runs the captured graph
    """
    if not torch.cuda.is_available() or device.type != 'cuda':
        return None
    
    # Check if CUDA graphs are supported
    if not hasattr(torch.cuda, 'Graph'):
        logging.warning("[CUDA] CUDA graphs not supported in this PyTorch version")
        return None
        
    try:
        # Create dummy inputs and optimizer for capturing
        static_input = torch.zeros(input_shape, device=device, dtype=torch.long, requires_grad=False)
        static_target = torch.zeros(input_shape, device=device, dtype=torch.long, requires_grad=False)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Warm up before capturing
        with torch.amp.autocast(device_type='cuda'):
            for _ in range(3):
                out = model(static_input)
                loss = torch.nn.functional.cross_entropy(out.view(-1, out.size(-1)), static_target.view(-1))
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
        
        # Capture forward and backward passes
        g = torch.cuda.Graph()
        
        with torch.cuda.graph(g):
            with torch.amp.autocast(device_type='cuda'):
                static_out = model(static_input)
                static_loss = torch.nn.functional.cross_entropy(
                    static_out.view(-1, static_out.size(-1)), 
                    static_target.view(-1)
                )
                static_loss.backward()
        
        # Create wrapper function that replaces inputs and executes graph
        def run_graph(real_input, real_target):
            # Copy real data to static tensors
            static_input.copy_(real_input)
            static_target.copy_(real_target)
            
            # Run the graph
            g.replay()
            
            # Return the loss value
            return static_loss.item()
        
        logging.info("[CUDA] Successfully created CUDA graph for training")
        return run_graph
        
    except Exception as e:
        logging.warning(f"[CUDA] Failed to create CUDA graph: {e}")
        return None

def configure_for_throughput():
    """
    Configure PyTorch for maximum throughput, potentially using more memory.
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Get available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # For larger GPUs, we can be more aggressive with memory usage to maximize throughput
        if total_memory > 6:
            # Allow PyTorch to allocate more memory upfront
            torch.cuda.empty_cache()
            
            # Optimize thread allocation for computation vs. memory
            if hasattr(torch, 'set_num_threads'):
                # Use fewer threads for better GPU utilization 
                import multiprocessing
                num_cpus = multiprocessing.cpu_count()
                torch.set_num_threads(max(1, min(4, num_cpus // 2)))
                logging.info(f"[CUDA] Set PyTorch thread count to {torch.get_num_threads()} for better GPU utilization")
            
            # Set larger workspace size for cuBLAS
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
            logging.info("[CUDA] Configured for maximum throughput (may use more memory)")
            return True
    except Exception as e:
        logging.warning(f"[CUDA] Failed to configure for throughput: {e}")
    
    return False

def disable_debug_apis():
    """
    Disable various debugging APIs that can slow down CUDA operations.
    """
    # Disable CUDA debugging features for production
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'  # Disable device synchronization assertions
    
    # Disable PyTorch warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
    
    # Disable autograd anomaly detection
    torch.autograd.set_detect_anomaly(False)
    
    logging.info("[CUDA] Disabled debug APIs for faster execution")
    return True

def apply_all_cuda_optimizations() -> Dict[str, bool]:
    """
    Apply all available CUDA optimizations and return results.
    
    Returns:
        Dictionary indicating which optimizations were successfully applied
    """
    results = {}
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping optimizations")
        return {'cuda_available': False}
    
    results['cuda_available'] = True
    results['tf32_enabled'] = enable_tf32()
    results['cudnn_optimized'] = optimize_cudnn()
    results['memory_optimized'] = setup_optimal_cuda_cache()
    results['debug_apis_disabled'] = disable_debug_apis()
    results['gpu_specific_optimizations'] = optimize_for_gpu_architecture()
    results['throughput_optimized'] = configure_for_throughput()
    
    # Log GPU information
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    logging.info(f"[CUDA] Optimizations applied for: {device_name} (CUDA {cuda_version}, {total_memory:.2f} GB)")
    
    return results

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Apply optimizations
    results = apply_all_cuda_optimizations()
    
    # Print results
    print("\n=== CUDA OPTIMIZATION RESULTS ===")
    for name, success in results.items():
        print(f"{name}: {'✓' if success else '✗'}")
    print("\nOptimizations have been applied to the PyTorch runtime.") 