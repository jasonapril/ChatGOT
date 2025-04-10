#!/usr/bin/env python
"""
Tests for the Evaluator class in src/craft/training/evaluation.py
"""

import pytest
import torch
import logging
from unittest.mock import MagicMock, patch, ANY
from torch.utils.data import DataLoader, TensorDataset

from craft.training.evaluation import Evaluator
from craft.training.callbacks import CallbackList, Callback

@pytest.fixture
def mock_model_eval():
    """Fixture for a mock model suitable for evaluation."""
    model = MagicMock(spec=torch.nn.Module)
    # Mock the forward pass to return dummy logits
    # Shape: [batch_size, seq_len, vocab_size]
    output_tensor = torch.randn(2, 5, 10)
    model.return_value = output_tensor
    model.eval = MagicMock()
    return model

@pytest.fixture
def mock_val_dataloader():
    """Fixture for a mock validation dataloader yielding dict batches."""
    batches = [
        {
            'input_ids': torch.randint(0, 10, (2, 5)), # batch=2, seq=5
            'labels': torch.randint(0, 10, (2, 5))
        },
        {
            'input_ids': torch.randint(0, 10, (2, 5)),
            'labels': torch.randint(0, 10, (2, 5))
        }
    ]
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter(batches)
    dataloader.__len__.return_value = len(batches)
    return dataloader

@pytest.fixture
def mock_empty_dataloader():
    """Fixture for an empty dataloader."""
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([])
    dataloader.__len__.return_value = 0
    return dataloader

@pytest.fixture
def mock_device_eval():
    """Fixture for a device (CPU)."""
    return torch.device("cpu")

@pytest.fixture
def mock_logger_eval():
    """Fixture for patching the logger used by Evaluator."""
    with patch('craft.training.evaluation.logging.getLogger') as mock_get_logger:
        logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = logger_instance
        yield logger_instance

@pytest.fixture
def mock_tqdm_eval():
    """Fixture for patching tqdm used in evaluation."""
    with patch('craft.training.evaluation.tqdm') as mock_tqdm_cls:
        # Make tqdm return the iterable directly for simplicity
        mock_tqdm_cls.side_effect = lambda x, **kwargs: x
        yield mock_tqdm_cls

@pytest.fixture
def mock_autocast_eval():
    """Fixture for patching torch.amp.autocast."""
    with patch('torch.amp.autocast') as mock_autocast:
        # Simulate context manager behavior
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        yield mock_autocast

@pytest.fixture
def mock_cross_entropy_eval():
    """Fixture for patching F.cross_entropy."""
    # Return a mock tensor with an item() method
    mock_loss_tensor = MagicMock(spec=torch.Tensor)
    mock_loss_tensor.item.return_value = 1.5 # Default mock loss value
    # Mock isnan/isinf to return False by default
    mock_loss_tensor.isnan.return_value.any.return_value = False
    mock_loss_tensor.isinf.return_value.any.return_value = False

    with patch('torch.nn.functional.cross_entropy', return_value=mock_loss_tensor) as mock_ce:
        yield mock_ce, mock_loss_tensor # Yield both the patch and the returned tensor

# --- Test Class ---

class TestEvaluator:

    def test_init(self, mock_model_eval, mock_val_dataloader, mock_device_eval, mock_logger_eval):
        """Test Evaluator initialization."""
        config = {'eval_key': 'eval_value'}
        callbacks = [MagicMock()]
        evaluator = Evaluator(
            model=mock_model_eval,
            val_dataloader=mock_val_dataloader,
            device=mock_device_eval,
            config=config,
            use_amp=True,
            callbacks=callbacks
        )
        assert evaluator.model == mock_model_eval
        assert evaluator.val_dataloader == mock_val_dataloader
        assert evaluator.device == mock_device_eval
        assert evaluator.config == config
        assert evaluator.use_amp is True
        assert evaluator.callbacks == callbacks
        assert evaluator.logger == mock_logger_eval

    # --- evaluate() Tests --- #

    def test_evaluate_no_dataloader(self, mock_model_eval, mock_device_eval, mock_logger_eval):
        """Test evaluate returns empty dict and logs info when dataloader is None."""
        evaluator = Evaluator(
            model=mock_model_eval,
            val_dataloader=None, # No dataloader
            device=mock_device_eval,
            config={}
        )
        results = evaluator.evaluate()

        assert results == {}
        mock_logger_eval.info.assert_called_with("No validation dataloader provided, skipping evaluation.")
        mock_model_eval.eval.assert_not_called() # Should not even start evaluation

    def test_evaluate_successful_run(
        self,
        mock_model_eval,
        mock_val_dataloader,
        mock_device_eval,
        mock_logger_eval,
        mock_tqdm_eval,
        mock_autocast_eval,
        mock_cross_entropy_eval,
    ):
        """Test a basic successful evaluation run."""
        mock_ce_func, mock_loss_tensor = mock_cross_entropy_eval
        config = {}
        evaluator = Evaluator(
            model=mock_model_eval,
            val_dataloader=mock_val_dataloader,
            device=mock_device_eval,
            config=config,
            use_amp=False # AMP disabled for basic test
        )

        # Patch time and also isnan/isinf to handle mock loss tensor
        false_check = MagicMock(any=MagicMock(return_value=False))
        with patch('time.time', side_effect=[100.0, 102.0]) as mock_time, \
             patch('torch.isnan', return_value=false_check) as mock_isnan, \
             patch('torch.isinf', return_value=false_check) as mock_isinf:
            results = evaluator.evaluate()

        # Assertions
        mock_model_eval.eval.assert_called_once()
        assert mock_model_eval.call_count == len(mock_val_dataloader) # Called for each batch
        mock_autocast_eval.assert_called_with(device_type=mock_device_eval.type, enabled=False)
        assert mock_ce_func.call_count == len(mock_val_dataloader)

        # Check loss calculation
        num_batches = len(mock_val_dataloader)
        expected_avg_loss = mock_loss_tensor.item() # Since loss is constant 1.5 per batch
        assert 'loss' in results
        assert results['loss'] == pytest.approx(expected_avg_loss)

        # Check timing and logging
        assert 'tokens_per_sec' in results
        assert mock_time.call_count == 2
        mock_logger_eval.info.assert_any_call("Starting evaluation...")
        mock_logger_eval.info.assert_any_call(
            f"Evaluation finished in {2.0:.2f}s. Avg Loss: {expected_avg_loss:.4f}, Tokens/sec: {results['tokens_per_sec']:.2f}"
        )

    def test_evaluate_amp_enabled(
        self,
        mock_model_eval,
        mock_val_dataloader,
        mock_device_eval,
        mock_logger_eval,
        mock_tqdm_eval,
        mock_autocast_eval,
        mock_cross_entropy_eval,
    ):
        """Test evaluation run with AMP enabled."""
        mock_ce_func, mock_loss_tensor = mock_cross_entropy_eval
        evaluator = Evaluator(
            model=mock_model_eval,
            val_dataloader=mock_val_dataloader,
            device=mock_device_eval,
            config={},
            use_amp=True # AMP enabled
        )

        # Patch time and also isnan/isinf to handle mock loss tensor
        false_check = MagicMock(any=MagicMock(return_value=False))
        with patch('time.time', side_effect=[100.0, 102.0]) as mock_time, \
             patch('torch.isnan', return_value=false_check) as mock_isnan, \
             patch('torch.isinf', return_value=false_check) as mock_isinf:
            results = evaluator.evaluate()

        # Assertions
        mock_model_eval.eval.assert_called_once()
        assert mock_model_eval.call_count == len(mock_val_dataloader)
        mock_autocast_eval.assert_called_with(device_type=mock_device_eval.type, enabled=True)
        assert mock_ce_func.call_count == len(mock_val_dataloader)
        assert 'loss' in results
        assert results['loss'] == pytest.approx(mock_loss_tensor.item())

    def test_evaluate_nan_inf_loss(
        self,
        mock_model_eval,
        mock_val_dataloader,
        mock_device_eval,
        mock_logger_eval,
        mock_tqdm_eval,
        mock_autocast_eval,
        mock_cross_entropy_eval,
    ):
        """Test that NaN/Inf loss values are skipped and logged."""
        mock_ce_func, mock_loss_tensor = mock_cross_entropy_eval
        # Configure mock loss to return NaN/Inf sometimes
        mock_nan_loss = MagicMock(spec=torch.Tensor)
        mock_nan_loss.item.return_value = float('nan')
        mock_nan_loss.isnan.return_value.any.return_value = True
        mock_nan_loss.isinf.return_value.any.return_value = False

        mock_inf_loss = MagicMock(spec=torch.Tensor)
        mock_inf_loss.item.return_value = float('inf')
        mock_inf_loss.isnan.return_value.any.return_value = False
        mock_inf_loss.isinf.return_value.any.return_value = True

        # Simulate: [Valid, NaN, Inf]
        mock_ce_func.side_effect = [mock_loss_tensor, mock_nan_loss, mock_inf_loss]
        # Adjust dataloader to have 3 batches to match side_effect
        batches = [
            {'input_ids': torch.tensor([[1]]), 'labels': torch.tensor([[1]])},
            {'input_ids': torch.tensor([[2]]), 'labels': torch.tensor([[2]])},
            {'input_ids': torch.tensor([[3]]), 'labels': torch.tensor([[3]])}
        ]
        mock_val_dataloader.__iter__.return_value = iter(batches)
        mock_val_dataloader.__len__.return_value = len(batches)

        evaluator = Evaluator(
            model=mock_model_eval,
            val_dataloader=mock_val_dataloader,
            device=mock_device_eval,
            config={},
            use_amp=False
        )

        # Define side effect for isnan/isinf checks
        def isnan_side_effect(tensor):
            if tensor is mock_nan_loss: return MagicMock(any=MagicMock(return_value=True))
            return MagicMock(any=MagicMock(return_value=False))

        def isinf_side_effect(tensor):
            if tensor is mock_inf_loss: return MagicMock(any=MagicMock(return_value=True))
            return MagicMock(any=MagicMock(return_value=False))

        with patch('time.time', side_effect=[300.0, 300.5]) as mock_time, \
             patch('torch.isnan', side_effect=isnan_side_effect) as mock_isnan, \
             patch('torch.isinf', side_effect=isinf_side_effect) as mock_isinf:
            results = evaluator.evaluate()

        # Assertions
        mock_model_eval.eval.assert_called_once()
        assert mock_model_eval.call_count == len(batches)
        assert mock_ce_func.call_count == len(batches)

        # Check that NaN/Inf logs occurred
        assert mock_logger_eval.warning.call_count == 2 # One for NaN, one for Inf
        mock_logger_eval.warning.assert_any_call("NaN/Inf detected during evaluation. Skipping batch.")

        # Check that loss is calculated based only on the valid batch
        # Only the first batch (loss 1.5) was valid
        expected_avg_loss = mock_loss_tensor.item() / len(batches) # Loss is averaged over total batches
        assert 'loss' in results
        assert results['loss'] == pytest.approx(expected_avg_loss)

    def test_evaluate_empty_dataloader(
        self,
        mock_model_eval,
        mock_empty_dataloader,
        mock_device_eval,
        mock_logger_eval,
        mock_tqdm_eval,
        mock_autocast_eval,
        mock_cross_entropy_eval,
    ):
        """Test evaluation with an empty dataloader."""
        mock_ce_func, mock_loss_tensor = mock_cross_entropy_eval
        evaluator = Evaluator(
            model=mock_model_eval,
            val_dataloader=mock_empty_dataloader,
            device=mock_device_eval,
            config={},
            use_amp=False
        )

        with patch('time.time', side_effect=[200.0, 200.1]) as mock_time:
            results = evaluator.evaluate()

        # Assertions
        mock_model_eval.eval.assert_called_once()
        mock_model_eval.call_count == 0 # No batches, no forward passes
        mock_autocast_eval.assert_not_called()
        mock_ce_func.assert_not_called()

        # Check loss calculation
        assert 'loss' in results
        assert results['loss'] == 0.0 # Avg loss is 0

        # Check timing and logging
        assert 'tokens_per_sec' in results
        assert results['tokens_per_sec'] == 0.0
        assert mock_time.call_count == 2
        mock_logger_eval.info.assert_any_call("Starting evaluation...")
        mock_logger_eval.info.assert_any_call(
            f"Evaluation finished in {0.1:.2f}s. Avg Loss: {0.0:.4f}, Tokens/sec: {0.0:.2f}"
        )

    # ... more tests ... 

class MockModel(torch.nn.Module):
    """A simple mock model that returns a predefined loss."""
    def __init__(self, output_loss=1.5):
        super().__init__()
        # Use Parameter to ensure it's recognized by .to(device)
        self.param = torch.nn.Parameter(torch.tensor(1.0))
        self.output_loss = torch.tensor(float(output_loss))

    def forward(self, *args, **kwargs):
        # Simulate model output dictionary including loss
        # Ensure loss requires grad if needed, but detach for evaluator
        # Make loss device-aware
        loss_val = self.output_loss.to(self.param.device).detach().requires_grad_(False)
        return {'loss': loss_val}

@pytest.fixture
def mock_model():
    """Provides a MockModel instance."""
    return MockModel(output_loss=2.0)

@pytest.fixture
def mock_dataloader():
    """Provides a simple mock DataLoader."""
    # Create dummy data: 10 batches of size 4, with 2 features (can be anything)
    data = torch.randn(10 * 4, 2)
    targets = torch.randn(10 * 4, 1)
    dataset = TensorDataset(data, targets)
    # Use batch_size=4 for testing batch iteration
    dataloader = DataLoader(dataset, batch_size=4)
    return dataloader

@pytest.fixture
def mock_callbacks():
    """Provides a MagicMock instance for CallbackList."""
    # Mock the specific methods called by Evaluator
    mock_cb_list = MagicMock(spec=CallbackList)
    mock_cb_list.on_validation_begin = MagicMock()
    mock_cb_list.on_validation_batch_end = MagicMock() # Though not currently used by Evaluator
    mock_cb_list.on_validation_end = MagicMock()
    return mock_cb_list

@pytest.fixture
def mock_logger():
    """Provides a MagicMock for the logger."""
    # Patch the logger used within the Evaluator module
    with patch('craft.training.evaluation.logger', new_callable=MagicMock) as mock_log:
        yield mock_log

@pytest.fixture(params=[torch.device("cpu"), torch.device("cuda" if torch.cuda.is_available() else "cpu")])
def device(request):
    """Provides CPU and CUDA devices if available."""
    if request.param.type == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param

def test_evaluator_initialization(
    mock_model,
    mock_dataloader,
    device,
    mock_callbacks
):
    """Test that the Evaluator initializes correctly."""
    config = {'eval_batch_log_interval': 5}
    evaluator = Evaluator(
        model=mock_model,
        val_dataloader=mock_dataloader,
        device=device,
        config=config,
        use_amp=False,
        callbacks=mock_callbacks
    )
    assert evaluator.model == mock_model
    assert evaluator.val_dataloader == mock_dataloader
    assert evaluator.device == device
    assert evaluator.config == config
    assert not evaluator.use_amp
    assert evaluator.callbacks == mock_callbacks

# More tests to come... 