import pytest
import torch
from unittest.mock import MagicMock, patch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import TensorDataset, DataLoader
import logging
import time
from craft.config.schemas import TrainingConfig
from craft.data.tokenizers.base import BaseTokenizer

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    # If ProgressTracker can't be imported, create a dummy class for spec=True
    class ProgressTracker: pass

# Import Callback for spec
from craft.training.callbacks import Callback


# --- Fixtures specific to tests/training --- 
# These augment or specialize fixtures from the root conftest.py

@pytest.fixture(scope="session") # Scope session as device doesn't change
def mock_device():
    """Provides a mock device."""
    return torch.device("cpu")

@pytest.fixture
def mock_model():
    """Mocks a torch.nn.Module, simulating a forward pass."""
    m = MagicMock(spec=torch.nn.Module)
    m.to.return_value = m
    # Mock the forward call to return a mock tensor
    # The shape should be compatible with cross_entropy: [batch, vocab_size, seq_len]
    mock_output_tensor = MagicMock(spec=torch.Tensor)
    mock_output_tensor.view.return_value = mock_output_tensor # For loss calculation view
    m.return_value = mock_output_tensor 
    # Make sure parameters returns something iterable for grad clipping
    m.parameters = MagicMock(return_value=[torch.nn.Parameter(torch.tensor(1.0))]) 
    m.train = MagicMock()
    m.eval = MagicMock()
    return m

@pytest.fixture
def mock_model_with_generate(mock_model): # Can build upon the basic mock_model
    """Provides a mock model with a callable generate method."""
    # Simulate generate output (needs to include prompt tokens based on callback logic)
    # Prompt: [101, 5141, 2747, 1037, 1994, 102] -> 6 tokens
    # Generate 5 new tokens: [1, 2, 3, 4, 5]
    generated_ids = torch.tensor([[101, 5141, 2747, 1037, 1994, 102, 1, 2, 3, 4, 5]])
    mock_model.generate = MagicMock(return_value=generated_ids)
    # Ensure eval/train still exist from the base mock_model fixture
    return mock_model

@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer with encode/decode methods."""
    tokenizer = MagicMock()
    # Simulate HF tokenizer __call__ behavior
    def encode_side_effect(text, return_tensors=None):
        # Dummy encoding, return simple tensor-like structure
        if text == "Once upon a time":
            input_ids = torch.tensor([[101, 5141, 2747, 1037, 1994, 102]])
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        return {'input_ids': torch.tensor([[0]]), 'attention_mask': torch.tensor([[1]])}
    # Use side_effect to make the mock callable like a function that returns the dict
    tokenizer.side_effect = encode_side_effect
    tokenizer.decode.return_value = " generated sample text"
    tokenizer.eos_token_id = 2 # Dummy EOS token id
    tokenizer.pad_token_id = 0 # Add pad token id if needed
    return tokenizer

# mock_trainer fixture is defined in root conftest.py

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
    Uses a real DataLoader for realistic iteration.
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
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # Wrap in a mock ONLY to control __len__ if needed (e.g., by tqdm)
    # but iterate over the *real* dataloader.
    mock_loader = MagicMock(spec=DataLoader)
    mock_loader.__iter__.return_value = iter(dataloader) 
    mock_loader.__len__.return_value = len(dataloader) 
    return mock_loader

# Restore the previous mock_scaler fixture which had more detail
@pytest.fixture
def mock_scaler():
    """Fixture for a mock GradScaler."""
    scaler = MagicMock(spec=GradScaler)
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

# --- Common Fixtures used within tests/training (might override root fixtures) ---

@pytest.fixture
def mock_dataloader():
    """Mocks a torch.utils.data.DataLoader."""
    return MagicMock(spec=torch.utils.data.DataLoader)

# NOTE: mock_optimizer is now used from root conftest.py

@pytest.fixture
def mock_tokenizer():
    """Mocks a craft.data.tokenizers.base.BaseTokenizer."""
    return MagicMock(spec=BaseTokenizer)

@pytest.fixture
def mock_callback():
    """Provides a simple MagicMock for use as a callback."""
    return MagicMock() # Simple mock for callbacks

@pytest.fixture
def default_training_config() -> TrainingConfig:
    """Provides a default, valid TrainingConfig instance."""
    # Minimal valid config according to Pydantic model defaults
    # Add required fields that don't have defaults
    return TrainingConfig(batch_size=8) # Add a default batch_size

@pytest.fixture
def full_training_config() -> TrainingConfig:
    """Provides a TrainingConfig with more fields set."""
    return TrainingConfig(
        epochs=5,
        use_amp=False,
        gradient_accumulation_steps=4,
        max_grad_norm=0.5,
        log_interval=50,
        save_steps_interval=1000,
        time_save_interval_seconds=3600,
        eval_interval=500,
        compile_model=False,
        activation_checkpointing=False,
        torch_compile=False,
        batch_size=16, # Included for completeness
        learning_rate=1e-4, # Included for completeness
        max_steps=10000, # Included for completeness
    ) 