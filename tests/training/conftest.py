import pytest
import torch
from unittest.mock import MagicMock, patch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import TensorDataset, DataLoader
import logging
import time

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    # If ProgressTracker can't be imported, create a dummy class for spec=True
    class ProgressTracker: pass

# Import Callback for spec
from craft.training.callbacks import Callback


# --- Fixtures ---

@pytest.fixture
def mock_model():
    """Fixture for a mock model."""
    model = MagicMock(spec=torch.nn.Module)
    # Simulate returning logits with shape [batch, seq_len, vocab_size]
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    # Create a dummy output tensor with requires_grad=True
    output_tensor = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    model.return_value = output_tensor
    # Mock parameters() for gradient clipping tests
    mock_param = MagicMock(spec=torch.Tensor)
    model.parameters.return_value = [mock_param]
    # Mock train/eval modes
    model.train = MagicMock()
    model.eval = MagicMock()
    return model

@pytest.fixture
def mock_optimizer():
    """Fixture for a mock optimizer."""
    optimizer = MagicMock(spec=torch.optim.AdamW)
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    # Mock param_groups for scheduler/logging if needed
    optimizer.param_groups = [{'lr': 0.001}]
    return optimizer

@pytest.fixture
def mock_scheduler():
    """Fixture for a mock learning rate scheduler."""
    scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
    scheduler.step = MagicMock()
    return scheduler

@pytest.fixture
def mock_dataloader():
    """Fixture for a mock dataloader yielding one batch.
    Yields batches of (inputs, targets) with appropriate shapes.
    """
    # Consistent shapes with mock_model output: [batch=2, seq=5, vocab=10]
    batch_size = 2
    seq_len = 5
    vocab_size = 10
    # Inputs: [batch_size, seq_len] - values don't matter as much
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Targets: [batch_size, seq_len] - values should be valid indices
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)) 
    # Create a dummy dataset and loader yielding one batch
    dataset = TensorDataset(inputs, targets)
    # Use a real DataLoader to ensure correct batch format is yielded
    # Use batch_size=batch_size to yield the whole dummy dataset as one batch
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # We need a mock wrapper to control __len__ if needed by tqdm
    mock_loader = MagicMock(spec=DataLoader)
    # Make iteration return the real dataloader's iterator
    mock_loader.__iter__.return_value = iter(dataloader) 
    # Set length for progress bar
    mock_loader.__len__.return_value = len(dataloader) 
    return mock_loader

@pytest.fixture
def mock_device():
    """Fixture for a mock device."""
    # Use a real CPU device for simplicity in type checking etc.
    return torch.device("cpu")

@pytest.fixture
def mock_scaler():
    """Fixture for a mock GradScaler."""
    scaler = MagicMock()
    # Simulate the scaled loss tensor returned when scale is called with AMP enabled
    scaler.mock_scaled_loss = MagicMock(spec=torch.Tensor, backward=MagicMock(), name="mock_scaled_loss_for_scaler")
    # Default behavior: scale returns the mock scaled loss (as if AMP is enabled)
    # Tests where AMP is disabled will override this return value.
    scaler.scale = MagicMock(return_value=scaler.mock_scaled_loss, name="scaler.scale")
    scaler.step = MagicMock()
    scaler.update = MagicMock()
    scaler.is_enabled = MagicMock(return_value=True) # Assume enabled by default
    scaler.unscale_ = MagicMock()
    return scaler

@pytest.fixture
def mock_progress_tracker_instance():
    """Fixture for a ProgressTracker mock instance."""
    instance_mock = MagicMock(spec=ProgressTracker)
    instance_mock.increment_step = MagicMock(name="instance_mock.increment_step")
    current_step_counter = 0
    def increment_step_side_effect(*args, **kwargs):
        nonlocal current_step_counter
        current_step_counter += 1
        instance_mock.current_step = current_step_counter 
        return current_step_counter
    instance_mock.increment_step.side_effect = increment_step_side_effect
    instance_mock.add_metrics = MagicMock(name="instance_mock.add_metrics") 
    instance_mock.get_average_metrics = MagicMock(return_value={}, name="instance_mock.get_average_metrics") 
    instance_mock.log = MagicMock(name="instance_mock.log")
    instance_mock.update = MagicMock(name="instance_mock.update")
    instance_mock.current_step = 0 # Initialize attributes
    instance_mock.current_epoch = 0
    return instance_mock

@pytest.fixture
def mock_logger_fixture():
    """Fixture to patch the logger used within TrainingLoop."""
    with patch('craft.training.training_loop.logging.getLogger') as mock_get_logger:
        mock_logger = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger
        yield mock_logger # Yield the mocked logger instance

@pytest.fixture
def mock_callback_fixture():
    """Fixture for a single mock callback."""
    callback = MagicMock(spec=Callback)
    callback.on_train_begin = MagicMock()
    callback.on_train_end = MagicMock()
    callback.on_epoch_begin = MagicMock()
    callback.on_epoch_end = MagicMock()
    callback.on_step_begin = MagicMock()
    callback.on_step_end = MagicMock()
    return callback

@pytest.fixture
def start_global_step():
    """Provides a default starting global step."""
    return 0 