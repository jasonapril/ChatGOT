"""Tests for performance utility functions."""

import pytest
from unittest.mock import patch, MagicMock

# Function to test
from craft.utils.performance import get_resource_metrics

# Mock psutil globally for these tests
@pytest.fixture(autouse=True)
def mock_psutil():
    mock_psutil_module = MagicMock(name="psutil")
    mock_psutil_module.cpu_percent.return_value = 55.5
    mock_vm = MagicMock()
    mock_vm.percent = 66.6
    mock_vm.used = 4 * 1024**3 # 4 GB
    mock_psutil_module.virtual_memory.return_value = mock_vm
    with patch.dict('sys.modules', {'psutil': mock_psutil_module}):
        yield mock_psutil_module

# Mock torch.cuda globally
@pytest.fixture(autouse=True)
def mock_torch_cuda():
    mock_torch_module = MagicMock(name="torch")
    mock_torch_module.cuda.is_available.return_value = True
    mock_torch_module.cuda.memory_allocated.return_value = 2 * 1024**3 # 2 GB
    with patch.dict('sys.modules', {'torch': mock_torch_module}):
        yield mock_torch_module.cuda


def test_get_resource_metrics_basic(mock_psutil, mock_torch_cuda):
    """Test basic CPU and memory metrics collection."""
    # Ensure GPU check doesn't interfere if include_gpu is True by default
    mock_torch_cuda.is_available.return_value = False 
    metrics = get_resource_metrics(include_gpu=False) # Explicitly exclude GPU for this test
    
    assert 'cpu_percent' in metrics
    assert metrics['cpu_percent'] == 55.5
    assert 'memory_percent' in metrics
    assert metrics['memory_percent'] == 66.6
    assert 'memory_used_gb' in metrics
    assert metrics['memory_used_gb'] == 4.0
    assert 'gpu_memory_used_gb' not in metrics # Should not be present
    
    mock_psutil.cpu_percent.assert_called_once()
    mock_psutil.virtual_memory.assert_called_once()
    mock_torch_cuda.memory_allocated.assert_not_called() # GPU part skipped

def test_get_resource_metrics_with_gpu(mock_psutil, mock_torch_cuda):
    """Test GPU metrics are included when CUDA is available and include_gpu is True."""
    mock_torch_cuda.is_available.return_value = True
    metrics = get_resource_metrics(include_gpu=True) # Default or explicitly True
    
    assert 'cpu_percent' in metrics
    assert metrics['cpu_percent'] == 55.5
    assert 'memory_percent' in metrics
    assert metrics['memory_percent'] == 66.6
    assert 'memory_used_gb' in metrics
    assert metrics['memory_used_gb'] == 4.0
    
    assert 'gpu_memory_used_gb' in metrics # Should be present
    assert metrics['gpu_memory_used_gb'] == 2.0
    
    mock_psutil.cpu_percent.assert_called_once()
    mock_psutil.virtual_memory.assert_called_once()
    mock_torch_cuda.is_available.assert_called_once()
    mock_torch_cuda.memory_allocated.assert_called_once()

def test_get_resource_metrics_no_cuda(mock_psutil, mock_torch_cuda):
    """Test GPU metrics are excluded when CUDA is not available."""
    mock_torch_cuda.is_available.return_value = False
    metrics = get_resource_metrics(include_gpu=True) # Try to include GPU
    
    assert 'cpu_percent' in metrics
    assert metrics['cpu_percent'] == 55.5
    assert 'memory_percent' in metrics
    assert metrics['memory_percent'] == 66.6
    assert 'memory_used_gb' in metrics
    assert metrics['memory_used_gb'] == 4.0
    
    assert 'gpu_memory_used_gb' not in metrics # Should be absent
    
    mock_psutil.cpu_percent.assert_called_once()
    mock_psutil.virtual_memory.assert_called_once()
    mock_torch_cuda.is_available.assert_called_once()
    mock_torch_cuda.memory_allocated.assert_not_called()

def test_get_resource_metrics_gpu_excluded(mock_psutil, mock_torch_cuda):
    """Test GPU metrics are excluded when include_gpu is False, even if CUDA is available."""
    mock_torch_cuda.is_available.return_value = True # CUDA *is* available
    metrics = get_resource_metrics(include_gpu=False) # But explicitly exclude
    
    assert 'cpu_percent' in metrics
    assert metrics['cpu_percent'] == 55.5
    assert 'memory_percent' in metrics
    assert metrics['memory_percent'] == 66.6
    assert 'memory_used_gb' in metrics
    assert metrics['memory_used_gb'] == 4.0
    
    assert 'gpu_memory_used_gb' not in metrics # Should be absent
    
    mock_psutil.cpu_percent.assert_called_once()
    mock_psutil.virtual_memory.assert_called_once()
    # The include_gpu check happens before the torch.cuda.is_available check
    mock_torch_cuda.is_available.assert_not_called() 
    mock_torch_cuda.memory_allocated.assert_not_called() 