"""
Tests for training optimization functions.
"""
import pytest
import argparse
import logging # Import logging for caplog
from unittest.mock import patch, MagicMock, PropertyMock
import torch # Add this import

# Functions to test
from craft.training.optimizations import (
    setup_mixed_precision,
    setup_torch_compile,
    configure_activation_checkpointing
)

# Mock torch globally for these tests
@pytest.fixture(autouse=True)
def mock_torch():
    """Mocks the torch module and its relevant sub-components."""
    mock_torch_module = MagicMock(name="torch")
    mock_torch_module.device = MagicMock(return_value=MagicMock(type='cuda')) # Mock device object
    mock_torch_module.nn.Module = MagicMock(spec=type)
    
    # CUDA related mocks
    mock_torch_module.cuda.is_available.return_value = True
    mock_torch_module.cuda.is_bf16_supported.return_value = True
    mock_torch_module.cuda.amp.GradScaler = MagicMock(name="GradScaler")
    mock_torch_module.bfloat16 = "torch.bfloat16" # Use strings for dtype checks
    mock_torch_module.float16 = "torch.float16"
    
    # torch.compile related mocks
    mock_torch_module.compile = MagicMock(name="torch_compile")
    mock_torch_module.__version__ = "2.0.0" # Add dummy version
    
    with patch.dict('sys.modules', {'torch': mock_torch_module}):
        yield mock_torch_module

# --- Tests for setup_mixed_precision ---

def test_setup_mixed_precision_disabled(mock_torch):
    """Test AMP setup when args.use_amp is False."""
    args = argparse.Namespace(use_amp=False)
    use_amp, scaler = setup_mixed_precision(args)
    assert not use_amp
    assert scaler is None
    mock_torch.cuda.amp.GradScaler.assert_not_called()

@patch('craft.training.optimizations.torch.cuda.amp.GradScaler') # Patch the actual class used
@patch('torch.cuda.is_available', return_value=True)
def test_setup_mixed_precision_enabled_cuda(mock_is_available, mock_grad_scaler_class, mock_torch):
    """Test AMP setup when enabled and CUDA is available (default mocks)."""
    args = argparse.Namespace(use_amp=True, use_bfloat16=False)
    mock_scaler_instance = MagicMock(name="ScalerInstance")
    mock_grad_scaler_class.return_value = mock_scaler_instance # Mock the instance creation

    use_amp, scaler = setup_mixed_precision(args)

    assert use_amp
    assert scaler is mock_scaler_instance # Check it's the instance we returned
    mock_grad_scaler_class.assert_called_once_with(enabled=True)
    # Check dtype used was float16 (default when use_bfloat16=False)
    # This requires inspecting logger calls or modifying function slightly.
    # For now, we mainly check the scaler setup.

@patch('craft.training.optimizations.torch.cuda.amp.GradScaler') # Patch the actual class used
@patch('craft.training.optimizations.torch.cuda._lazy_init') # Prevent real CUDA init
@patch('craft.training.optimizations.torch.cuda.is_bf16_supported', return_value=True)
@patch('torch.cuda.is_available', return_value=True)
def test_setup_mixed_precision_enabled_bfloat16_supported(mock_is_available, mock_is_bf16_supported, mock_lazy_init, mock_grad_scaler_class, mock_torch):
    """Test AMP setup with bfloat16 when supported."""
    args = argparse.Namespace(use_amp=True, use_bfloat16=True)
    mock_scaler_instance = MagicMock(name="ScalerInstance_bf16")
    mock_grad_scaler_class.return_value = mock_scaler_instance

    use_amp, scaler = setup_mixed_precision(args)

    assert use_amp
    assert scaler is mock_scaler_instance # Check it's the instance we returned
    mock_grad_scaler_class.assert_called_once_with(enabled=True)
    # Should ideally check logs for warning and fallback to float16

@patch('craft.training.optimizations.torch.cuda.amp.GradScaler') # Patch the actual class used
@patch('craft.training.optimizations.torch.cuda._lazy_init') # Prevent real CUDA init
@patch('craft.training.optimizations.torch.cuda.is_bf16_supported', return_value=False)
@patch('torch.cuda.is_available', return_value=True)
def test_setup_mixed_precision_enabled_bfloat16_not_supported(mock_is_available, mock_is_bf16_supported, mock_lazy_init, mock_grad_scaler_class, mock_torch):
    """Test AMP setup falls back to float16 if bfloat16 not supported."""
    args = argparse.Namespace(use_amp=True, use_bfloat16=True)
    mock_scaler_instance = MagicMock(name="ScalerInstance_fp16_fallback")
    mock_grad_scaler_class.return_value = mock_scaler_instance

    use_amp, scaler = setup_mixed_precision(args)

    assert use_amp
    assert scaler is mock_scaler_instance # Check it's the instance we returned
    mock_grad_scaler_class.assert_called_once_with(enabled=True)
    # Should ideally check logs for warning and fallback to float16

@patch('torch.cuda.is_available', return_value=False) # Explicitly mock for this test
def test_setup_mixed_precision_enabled_no_cuda(mock_is_available, mock_torch):
    """Test AMP is disabled if requested but CUDA not available."""
    args = argparse.Namespace(use_amp=True, use_bfloat16=False)
    
    use_amp, scaler = setup_mixed_precision(args)
    
    assert not use_amp
    assert scaler is None
    mock_torch.cuda.amp.GradScaler.assert_not_called()
    # Should ideally check logs for warning and fallback to float16

# --- Tests for setup_torch_compile ---

def test_setup_torch_compile_disabled(mock_torch):
    """Test torch.compile is not called when args.torch_compile is False."""
    args = argparse.Namespace(torch_compile=False)
    mock_model = MagicMock(spec=torch.nn.Module)
    
    compiled_model = setup_torch_compile(args, mock_model)
    
    assert compiled_model == mock_model # Should return original model
    mock_torch.compile.assert_not_called()

@patch('craft.training.optimizations.torch.compile') # Patch where it's used
def test_setup_torch_compile_enabled_available(mock_compile, mock_torch):
    """Test torch.compile is called when enabled and available."""
    args = argparse.Namespace(torch_compile=True, compile_mode='default')
    mock_model = MagicMock(spec=torch.nn.Module, name="OriginalModel")
    mock_compiled_model = MagicMock(spec=torch.nn.Module, name="CompiledModel")
    mock_compile.return_value = mock_compiled_model
    
    compiled_model = setup_torch_compile(args, mock_model)
    
    mock_compile.assert_called_once_with(mock_model, mode='default')
    assert compiled_model == mock_compiled_model

@patch('builtins.hasattr', return_value=False) # Correct target
def test_setup_torch_compile_enabled_not_available(mock_hasattr, mock_torch):
    """Test returns original model if compile requested but not available."""
    # Simulate torch.compile not existing via hasattr returning False
    args = argparse.Namespace(torch_compile=True)
    mock_model = MagicMock(spec=torch.nn.Module)

    compiled_model = setup_torch_compile(args, mock_model)

    assert compiled_model == mock_model
    mock_torch.compile.assert_not_called()

@patch('craft.training.optimizations.torch.compile') # Patch where it's used
def test_setup_torch_compile_compilation_fails(mock_compile, mock_torch):
    """Test returns original model if torch.compile raises an exception."""
    args = argparse.Namespace(torch_compile=True, compile_mode='reduce-overhead')
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_compile.side_effect = RuntimeError("Compilation failed")
    
    with patch('craft.training.optimizations.logging.warning') as mock_warning:
        compiled_model = setup_torch_compile(args, mock_model)

    assert compiled_model == mock_model # Should fall back to original model
    mock_compile.assert_called_once_with(mock_model, mode='reduce-overhead')
    mock_warning.assert_called_once()

# --- Tests for configure_activation_checkpointing ---

def test_configure_activation_checkpointing_disabled(mock_torch):
    """Test no changes are made if use_activation_checkpointing is False."""
    args = argparse.Namespace(use_activation_checkpointing=False)
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.modules.return_value = [MagicMock()]

    configure_activation_checkpointing(args, mock_model)
    
    # Check that use_checkpoint was not set on any module
    for module in mock_model.modules():
        assert not hasattr(module, 'use_checkpoint') or not module.use_checkpoint.called

@patch('craft.training.optimizations.checkpoint', create=True) # Mock the potential import
def test_configure_activation_checkpointing_enabled_success(mock_checkpoint_util, mock_torch):
    """Test sets use_checkpoint=True on layers when enabled and model has layers."""
    args = argparse.Namespace(use_activation_checkpointing=True)
    mock_layer1 = MagicMock(name="Layer1")
    mock_layer2 = MagicMock(name="Layer2")
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.transformer_layers = [mock_layer1, mock_layer2]

    configure_activation_checkpointing(args, mock_model)
    
    assert mock_layer1.use_checkpoint == True
    assert mock_layer2.use_checkpoint == True

def test_configure_activation_checkpointing_enabled_no_layers(mock_torch):
    """Test it warns but doesn't fail if enabled but model lacks expected structure."""
    args = argparse.Namespace(use_activation_checkpointing=True)
    mock_model = MagicMock(spec=torch.nn.Module)
    del mock_model.transformer_layers

    with patch('craft.training.optimizations.logging.warning') as mock_warning:
        configure_activation_checkpointing(args, mock_model)

    mock_warning.assert_called_once()
    assert "Model doesn't have expected structure for activation checkpointing" in mock_warning.call_args[0][0]

@patch('craft.training.optimizations.checkpoint', create=True)
@patch('craft.training.optimizations.logging.warning')
def test_configure_activation_checkpointing_exception(mock_warning, mock_checkpoint_util):
    """Test it warns but doesn't fail if setting use_checkpoint raises an error."""
    args = argparse.Namespace(use_activation_checkpointing=True)
    mock_layer = MagicMock(name="Layer1", spec=True)
    mock_torch_module = MagicMock(name="torch")
    mock_torch_module.nn.Module = MagicMock(spec=type)

    # Configure the mock's 'use_checkpoint' property to raise TypeError when set
    mock_prop = PropertyMock(side_effect=TypeError("Simulated attribute setting error"))
    type(mock_layer).use_checkpoint = mock_prop # Patch the property on the type

    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.transformer_layers = [mock_layer]

    configure_activation_checkpointing(args, mock_model)

    # Check that the patched logging.warning (mock_warning) was called
    mock_warning.assert_called_once_with(
        "Failed to apply activation checkpointing: Simulated attribute setting error"
    )
    # Cleanup the property patch
    del type(mock_layer).use_checkpoint

# --- Tests for optimize_memory_usage ---
# ... existing tests ...