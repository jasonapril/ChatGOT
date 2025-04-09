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
from craft.training.trainer import Trainer
from craft.config.schemas import TrainingConfig

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

# Basic Config Fixture
@pytest.fixture
def base_training_config():
    """Provides a base TrainingConfig object for TrainingLoop tests."""
    # Return a Pydantic TrainingConfig object with minimal required fields
    # Add default values for fields accessed by TrainingLoop.__init__
    return TrainingConfig(
        batch_size=4, # Required field
        learning_rate=1e-4, # Required field
        use_amp=False,
        gradient_accumulation_steps=1,
        log_interval=10,
        save_interval=0, # Or some default if needed
        time_save_interval_seconds=0,
        max_steps=None,
        num_epochs=1, # Need num_epochs or max_steps
        max_grad_norm=1.0,
        # Other fields will use their Pydantic defaults
    )

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
    mock_trainer = MagicMock(spec=Trainer) # Add mock trainer
    try:
        # Pass global_step=0 as it's expected by the loop for callback context
        training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker, trainer=mock_trainer) # Pass trainer
    except Exception as e:
        pytest.fail(f"TrainingLoop.train_epoch raised an exception: {e}")

def test_training_loop_gradient_accumulation(base_training_config, mock_dataloader):
    """Test optimizer step, scaler calls, and callback frequency with gradient accumulation."""
    model = MockTrainModel()
    optimizer = MagicMock(spec=AdamW)
    optimizer.param_groups = [{'lr': 1e-3}] # Mock param_groups
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    accumulation_steps = 3 # Use 3 for clearer division
    config_for_test = base_training_config.model_copy(update={
        "gradient_accumulation_steps": accumulation_steps,
        "use_amp": False # Test without AMP first
    })
    total_batches = len(mock_dataloader)
    # Correct calculation for expected steps including the final partial step
    expected_optimizer_steps = (total_batches + accumulation_steps - 1) // accumulation_steps

    # Mock scaler
    mock_scaler = MagicMock()
    mock_scaler.scale = MagicMock(side_effect=lambda x: x) # Just return loss
    mock_scaler.step = MagicMock(side_effect=lambda opt: opt.step())
    mock_scaler.update = MagicMock()
    mock_scaler.unscale_ = MagicMock() # Needed even if AMP=False for structure

    # Mock Callback
    mock_callback = MagicMock(spec=Callback)
    mock_callback.on_step_begin = MagicMock()
    mock_callback.on_step_end = MagicMock()
    callbacks = CallbackList([mock_callback])

    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader,
        device=torch.device("cpu"),
        config=config_for_test,
        callbacks=callbacks,
    )
    # Manually replace the scaler created in __init__ with our mock
    training_loop.scaler = mock_scaler

    mock_progress_tracker = ProgressTracker(total_steps=total_batches)
    mock_progress_tracker.start()
    mock_trainer = MagicMock(spec=Trainer)
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker, trainer=mock_trainer)

    # --- Assertions --- #
    # Optimizer calls
    assert optimizer.step.call_count == expected_optimizer_steps, f"Expected {expected_optimizer_steps} optimizer steps, got {optimizer.step.call_count}"
    assert optimizer.zero_grad.call_count == expected_optimizer_steps + 1, f"Expected {expected_optimizer_steps + 1} zero_grad calls, got {optimizer.zero_grad.call_count}" # +1 for call before loop

    # Scaler calls (even with AMP=False, step/update might be called conditionally)
    # Since AMP is False, loop calls optimizer.step directly, not scaler.step
    mock_scaler.step.assert_not_called()
    mock_scaler.update.assert_not_called()
    # unscale_ should not be called if use_amp is False
    mock_scaler.unscale_.assert_not_called()

    # Callback calls
    # on_step_begin might still be called every batch depending on implementation choice
    assert mock_callback.on_step_begin.call_count == total_batches # Check step_begin is called every batch
    # on_step_end should only be called when the optimizer steps
    assert mock_callback.on_step_end.call_count == expected_optimizer_steps, f"Expected {expected_optimizer_steps} on_step_end calls, got {mock_callback.on_step_end.call_count}"

def test_training_loop_gradient_accumulation_with_amp(base_training_config, mock_dataloader):
    """Test scaler calls with gradient accumulation and AMP enabled."""
    model = MockTrainModel()
    optimizer = MagicMock(spec=AdamW)
    optimizer.param_groups = [{'lr': 1e-3}]
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock() # Mock the optimizer step itself
    accumulation_steps = 2
    config_for_test = base_training_config.model_copy(update={
        "gradient_accumulation_steps": accumulation_steps,
        "use_amp": True # Enable AMP
    })
    total_batches = len(mock_dataloader)
    # Correct calculation for expected steps including the final partial step
    expected_optimizer_steps = (total_batches + accumulation_steps - 1) // accumulation_steps

    # Mock scaler with enabled=True behavior
    mock_scaler = MagicMock()
    mock_scaler.is_enabled = MagicMock(return_value=True)
    mock_scaler.scale = MagicMock(side_effect=lambda x: x * 2.0) # Simulate scaling
    # Mock scaler.step to return non-None to indicate step was taken
    mock_scaler.step = MagicMock(return_value=1.0)
    # Mock get_scale to return changing numbers
    # Provide enough unique descending values for the side_effect list (2 per step)
    scale_values = [1000.0 - i*10 for i in range(expected_optimizer_steps * 2 + 2)]
    mock_scaler.get_scale = MagicMock(side_effect=scale_values)
    mock_scaler.update = MagicMock()
    mock_scaler.unscale_ = MagicMock()
    # Mock Callback for this test
    mock_callback = MagicMock(spec=Callback)
    mock_callback.on_step_begin = MagicMock()
    mock_callback.on_step_end = MagicMock()
    callbacks = CallbackList([mock_callback])

    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader,
        device=torch.device("cpu"), # Test on CPU, scaler handles device type internally
        config=config_for_test,
        callbacks=callbacks, # Pass callbacks
    )
    training_loop.scaler = mock_scaler # Replace scaler

    mock_progress_tracker = ProgressTracker(total_steps=total_batches)
    mock_progress_tracker.start()
    mock_trainer = MagicMock(spec=Trainer)
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker, trainer=mock_trainer)

    # Assertions focused on AMP interactions
    optimizer.step.assert_not_called() # Scaler calls step internally
    assert mock_scaler.step.call_count == expected_optimizer_steps
    assert mock_scaler.update.call_count == expected_optimizer_steps # Update called after successful step
    # unscale_ should be called before step when AMP is enabled
    assert mock_scaler.unscale_.call_count == expected_optimizer_steps, f"Expected {expected_optimizer_steps} unscale_ calls with AMP, got {mock_scaler.unscale_.call_count}"

    # Callback calls
    assert mock_callback.on_step_begin.call_count == total_batches
    # on_step_end should be called once per successful optimizer step
    assert mock_callback.on_step_end.call_count == expected_optimizer_steps

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
    mock_trainer = MagicMock(spec=Trainer) # Add mock trainer
    # Pass global_step=0
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker, trainer=mock_trainer) # Pass trainer
    total_batches = len(mock_dataloader)
    assert mock_callback.on_step_begin.call_count == total_batches
    assert mock_callback.on_step_end.call_count == total_batches

def test_training_loop_loss_decreases(base_training_config, mock_dataloader):
    """Very basic check if loss generally trends downward over a few steps."""
    model = MockTrainModel(vocab_size=10, d_model=8)
    optimizer = AdamW(model.parameters(), lr=1e-2) # Use a reasonable LR
    total_steps = len(mock_dataloader)
    # Update config directly for test-specific needs
    config_for_test = base_training_config.model_copy(update={
        "log_interval": 1,
        "gradient_accumulation_steps": 1
    })
    
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        train_dataloader=mock_dataloader, # Expects list of batches
        device=torch.device("cpu"),
        config=config_for_test, # Pass modified config
        callbacks=CallbackList([]) 
    )
    mock_progress_tracker = ProgressTracker(total_steps=total_steps)
    mock_progress_tracker.start() # Start the tracker
    mock_trainer = MagicMock(spec=Trainer) # Add mock trainer
    losses = []
    original_update = mock_progress_tracker.update
    def capture_loss_update(*args, **kwargs):
        # Capture loss passed to progress tracker update
        if 'loss' in kwargs and isinstance(kwargs['loss'], (float, int)): 
            losses.append(kwargs['loss'])
        original_update(*args, **kwargs)
    mock_progress_tracker.update = capture_loss_update

    # Pass global_step=0
    training_loop.train_epoch(current_epoch=0, global_step=0, progress=mock_progress_tracker, trainer=mock_trainer) # Pass trainer

    # Check if losses list has values and trends downwards (very basic)
    assert len(losses) > 0, "No losses were captured"

    if len(losses) > 5: 
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