import pytest
import torch
import numpy as np
import random
from unittest.mock import patch, MagicMock

from craft.utils.common import set_seed, setup_device, format_number

# --- Tests for set_seed ---

def test_set_seed():
    """Test that setting the seed ensures reproducibility."""
    seed = 42
    set_seed(seed)
    torch_val1 = torch.rand(10)
    numpy_val1 = np.random.rand(10)
    # Simulate CUDA operations if available
    if torch.cuda.is_available():
        # Use torch.tensor with device instead of deprecated constructor
        torch_cuda_val1 = torch.tensor(1.0, device='cuda').normal_()
    
    set_seed(seed)
    torch_val2 = torch.rand(10)
    numpy_val2 = np.random.rand(10)
    if torch.cuda.is_available():
        torch_cuda_val2 = torch.tensor(1.0, device='cuda').normal_()
    
    assert torch.equal(torch_val1, torch_val2)
    np.testing.assert_array_equal(numpy_val1, numpy_val2)
    if torch.cuda.is_available():
        assert torch.equal(torch_cuda_val1, torch_cuda_val2)

@patch('torch.backends.cudnn')
def test_set_seed_cudnn_settings(mock_cudnn):
    """Test that set_seed configures cudnn correctly."""
    set_seed(42)
    assert mock_cudnn.deterministic is True
    assert mock_cudnn.benchmark is False

# --- Tests for setup_device ---

@patch('torch.cuda.is_available', return_value=False)
@patch('logging.info') # Mock logging to avoid console output during tests
def test_setup_device_cpu(mock_log, mock_cuda_available):
    """Test setup_device returns cpu when cuda is not available."""
    device = setup_device("auto")
    assert device.type == 'cpu'
    mock_log.assert_any_call("Using CPU for computation")

@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.get_device_properties')
@patch('torch.cuda.get_device_name', return_value="Test GPU")
@patch('logging.info')
def test_setup_device_cuda_auto(mock_log, mock_get_name, mock_get_props, mock_cuda_available):
    """Test setup_device returns cuda when cuda is available (auto)."""
    mock_props = MagicMock()
    mock_props.total_memory = 8 * (1024**3) # 8 GB
    mock_props.major = 8
    mock_props.minor = 6
    mock_get_props.return_value = mock_props

    device = setup_device("auto")
    assert device.type == 'cuda'
    mock_get_props.assert_called_once_with(device)
    mock_log.assert_any_call("Using GPU: Test GPU")
    mock_log.assert_any_call("  - Total memory: 8.00 GB")
    mock_log.assert_any_call("  - CUDA capability: 8.6")

@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.get_device_properties')
@patch('torch.cuda.get_device_name', return_value="Test GPU")
@patch('logging.info')
def test_setup_device_cuda_explicit(mock_log, mock_get_name, mock_get_props, mock_cuda_available):
    """Test setup_device returns cuda when explicitly requested."""
    mock_props = MagicMock()
    mock_props.total_memory = 8 * (1024**3) # 8 GB
    mock_props.major = 8
    mock_props.minor = 6
    mock_get_props.return_value = mock_props
    
    device = setup_device("cuda")
    assert device.type == 'cuda'
    # Check specific index
    device = setup_device("cuda:0")
    assert device.type == 'cuda'
    assert device.index == 0

@patch('logging.info')
def test_setup_device_cpu_explicit(mock_log):
    """Test setup_device returns cpu when explicitly requested."""
    device = setup_device("cpu")
    assert device.type == 'cpu'

# --- Tests for format_number ---

@pytest.mark.parametrize("input_num, expected_str", [
    (123, "123"),
    (1234, "1,234"),
    (1234567, "1,234,567"),
    (-5678, "-5,678"),
    (0, "0"),
    (1234.56, "1,234.56"),
    (12345.678, "12,345.68"), # Rounds to 2 decimal places
    (0.12, "0.12"),
    (-9876.543, "-9,876.54"),
    (1e6, "1,000,000.00"),
    (None, "None"), # Test non-numeric case
    ("abc", "abc"), # Test non-numeric case
])
def test_format_number(input_num, expected_str):
    """Test format_number with various inputs."""
    assert format_number(input_num) == expected_str 