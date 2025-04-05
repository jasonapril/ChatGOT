import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
import logging

from craft.training.callbacks import Callback, CallbackList
from craft.training.progress import ProgressTracker
from craft.training.training_loop import TrainingLoop

# --- Mocks and Fixtures --- #

# Mock Model (Copied from original test_training.py)
class MockTrainModel(nn.Module):
    def __init__(self, vocab_size=10, d_model=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self._config = MagicMock() # Simple mock for config attribute if needed
        self._config.vocab_size = vocab_size

    def forward(self, x, targets=None):
        # Simulate a forward pass returning logits and optionally loss
        batch_size, seq_len = x.shape
        emb = self.embedding(x)
        logits = self.linear(emb) # (batch, seq_len, vocab_size)
        loss = None
        if targets is not None:
            # Dummy loss calculation
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return (logits, loss) if loss is not None else logits

# Mock Dataloader Fixture (Copied from original test_training.py)
@pytest.fixture
def mock_dataloader():
    # Simple dataloader yielding tuples of tensors (input_ids, target_ids)
    vocab_size = 10
    seq_len = 10
    batch_size = 4
    num_batches = 20 # Increased number of batches

    data = []
    # Generate sequences where target is often input + 1 (simple pattern)
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size -1, (batch_size, seq_len))
        target_ids = (input_ids + 1) % vocab_size # Target is next token
        # Introduce some noise/variation
        noise_mask = torch.rand(batch_size, seq_len) < 0.1
        target_ids[noise_mask] = torch.randint(0, vocab_size, (int(noise_mask.sum()),))

        data.append((input_ids, target_ids))
    return data

# Basic Config Fixture (Copied and trimmed from original test_training.py)
@pytest.fixture
def base_training_config():
    """Provides a base OmegaConf training config relevant for TrainingLoop."""
    conf = OmegaConf.create({
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "log_interval": 10,
        "use_amp": False,
        "torch_compile": False, # Added
        "activation_checkpointing": False # Added
        # Removed fields not directly used by TrainingLoop itself 
        # (e.g., num_epochs, eval_interval, batch_size, lr, max_steps)
    })
    return conf

# --- Tests for TrainingLoop (Moved from original test_training.py) --- #

def test_training_loop_init(base_training_config, mock_dataloader):
    """Test TrainingLoop initialization."""
    model = MockTrainModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader, # Pass the list of batches
        device=torch.device("cpu"),
        config=base_training_config # Use the trimmed config
    )
    assert training_loop.model is model
    assert training_loop.optimizer is optimizer
    assert training_loop.train_dataloader is mock_dataloader
    assert training_loop.device == torch.device("cpu")
    assert training_loop.config == base_training_config
    assert training_loop.log_interval == base_training_config.log_interval
    assert training_loop.use_amp == base_training_config.use_amp

def test_training_loop_train_epoch_runs(base_training_config, mock_dataloader):
    """Test that train_epoch runs without throwing errors."""
    model = MockTrainModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader, # Expects list of batches
        device=torch.device("cpu"),
        config=base_training_config,
        callbacks=CallbackList([]) # Explicitly pass CallbackList
    )
    total_steps = len(mock_dataloader)
    mock_progress_tracker = ProgressTracker(total_steps=total_steps)
    mock_progress_tracker.start() # Start the tracker
    try:
        # Pass global_step=0 as it's expected by the loop for callback context
        training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker) 
    except Exception as e:
        pytest.fail(f"TrainingLoop.train_epoch raised an exception: {e}")

def test_training_loop_gradient_accumulation(base_training_config, mock_dataloader):
    """Test that optimizer.step() is called correctly with gradient accumulation."""
    model = MockTrainModel()
    optimizer = MagicMock(spec=AdamW)
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    accumulation_steps = 2
    # Update the config object directly
    base_training_config.gradient_accumulation_steps = accumulation_steps
    total_steps = len(mock_dataloader)
    
    # Pass accumulation steps directly during init as config might not be the only source
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader, # Expects list of batches
        device=torch.device("cpu"),
        config=base_training_config, 
        log_interval=total_steps + 1, # Ensure logging doesn't interfere
        callbacks=CallbackList([]), 
        gradient_accumulation_steps=accumulation_steps # Pass explicitly
    )
    mock_progress_tracker = ProgressTracker(total_steps=total_steps)
    mock_progress_tracker.start() # Start the tracker
    # Pass global_step=0
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker)
    
    total_batches = len(mock_dataloader)
    expected_optimizer_steps = total_batches // accumulation_steps 
    assert optimizer.step.call_count == expected_optimizer_steps
    # zero_grad is called once before the loop and once per effective step
    assert optimizer.zero_grad.call_count == expected_optimizer_steps + 1 

def test_training_loop_callbacks_called(base_training_config, mock_dataloader):
    """Test that step_begin and step_end callbacks are called."""
    model = MockTrainModel()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    mock_callback = MagicMock(spec=Callback)
    mock_callback.on_step_begin = MagicMock()
    mock_callback.on_step_end = MagicMock()
    callbacks = CallbackList([mock_callback])
    total_steps = len(mock_dataloader)
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader, # Expects list of batches
        device=torch.device("cpu"),
        config=base_training_config,
        callbacks=callbacks
    )
    mock_progress_tracker = ProgressTracker(total_steps=total_steps)
    mock_progress_tracker.start() # Start the tracker
    # Pass global_step=0
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker) 
    total_batches = len(mock_dataloader)
    assert mock_callback.on_step_begin.call_count == total_batches
    assert mock_callback.on_step_end.call_count == total_batches

def test_training_loop_loss_decreases(base_training_config, mock_dataloader):
    """Very basic check if loss generally trends downward over a few steps."""
    model = MockTrainModel(vocab_size=10, d_model=8)
    optimizer = AdamW(model.parameters(), lr=1e-2) # Use a reasonable LR
    total_steps = len(mock_dataloader)
    # Update config directly
    base_training_config.log_interval = 1
    base_training_config.gradient_accumulation_steps = 1
    
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader, # Expects list of batches
        device=torch.device("cpu"),
        config=base_training_config,
        gradient_accumulation_steps=1, # Pass explicitly
        log_interval=1, # Pass explicitly
        callbacks=CallbackList([]) 
    )
    mock_progress_tracker = ProgressTracker(total_steps=total_steps)
    mock_progress_tracker.start() # Start the tracker
    losses = []
    original_update = mock_progress_tracker.update
    def capture_loss_update(*args, **kwargs):
        # Capture loss passed to progress tracker update
        if 'loss' in kwargs and isinstance(kwargs['loss'], (float, int)): 
            losses.append(kwargs['loss'])
        original_update(*args, **kwargs)
    mock_progress_tracker.update = capture_loss_update

    # Pass global_step=0
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker) 

    assert len(losses) > 5, "Need several loss values to check trend" 
    midpoint = len(losses) // 2
    if midpoint > 1: 
        first_few_avg = sum(losses[:3]) / 3
        last_few_avg = sum(losses[-3:]) / 3
        tolerance = 0.1 # Allow for noise
        assert last_few_avg < first_few_avg + tolerance, (
            f"Loss did not decrease significantly "
            f"(Avg First 3: {first_few_avg:.4f}, Avg Last 3: {last_few_avg:.4f})"
        )
    else:
        pytest.skip("Not enough loss values captured to compare halves reliably.") 