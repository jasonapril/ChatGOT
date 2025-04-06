# tests/training/test_checkpointing.py
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler
from pathlib import Path
import os
import logging
import sys
from dataclasses import fields # For TrainingState comparison
import re
from unittest.mock import patch
from dataclasses import asdict

# Add src directory to path to allow imports
# Adjust based on actual location relative to tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from craft.training.checkpointing import CheckpointManager, TrainingState, CheckpointLoadError
# from craft.training.callbacks.base import CallbackList # Not needed for these tests

# Set up logging for tests if needed
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Pytest Fixtures ---

@pytest.fixture
def simple_model() -> nn.Module:
    """Provides a simple linear model for testing."""
    return nn.Linear(10, 2)

@pytest.fixture
def simple_optimizer(simple_model: nn.Module) -> AdamW:
    """Provides a simple AdamW optimizer."""
    return AdamW(simple_model.parameters(), lr=1e-3)

@pytest.fixture
def simple_scheduler(simple_optimizer: AdamW) -> StepLR:
    """Provides a simple StepLR scheduler."""
    return StepLR(simple_optimizer, step_size=10, gamma=0.1)

@pytest.fixture
def mock_scaler():
    """Mocks the AMP GradScaler, using the modern torch.amp syntax."""
    # Use the recommended way: torch.amp.GradScaler(device_type='cuda', enabled=...)
    # device_type='cuda' is implicit if enabled=True and CUDA is available
    return torch.amp.GradScaler(enabled=torch.cuda.is_available())

@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory structure matching the project output for checkpoints."""
    exp_name = "test_experiment"
    chkpt_path = tmp_path / "outputs" / "experiments" / exp_name / "checkpoints"
    chkpt_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created temporary checkpoint dir: {chkpt_path}")
    return chkpt_path

@pytest.fixture
def checkpoint_manager(
    simple_model: nn.Module,
    simple_optimizer: AdamW,
    simple_scheduler: StepLR,
    mock_scaler: GradScaler,
    checkpoint_dir: Path # Use the fixture that creates the dir
) -> CheckpointManager:
    """Provides a CheckpointManager instance configured for testing."""
    exp_name = "test_experiment"

    # CheckpointManager uses get_original_cwd internally.
    # For isolated tests, we manually set its target directory AFTER initialization.
    manager = CheckpointManager(
        model=simple_model,
        optimizer=simple_optimizer,
        scheduler=simple_scheduler,
        scaler=mock_scaler,
        experiment_name=exp_name,
        callbacks=None,
        tokenizer=None,
        device=None, # Use CPU for tests unless GPU is explicitly tested
        config={},
    )
    # Manually set the directory for the test instance to the fixture path
    manager.checkpoint_dir = checkpoint_dir
    logger.debug(f"Configured CheckpointManager with dir: {manager.checkpoint_dir}")
    return manager


@pytest.fixture
def initial_training_state(
    simple_model: nn.Module,
    simple_optimizer: AdamW,
    simple_scheduler: StepLR,
    mock_scaler: GradScaler
) -> TrainingState:
    """Provides a sample TrainingState object."""
    return TrainingState(
        epoch=1,
        global_step=50,
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict=simple_optimizer.state_dict(),
        scheduler_state_dict=simple_scheduler.state_dict(),
        scaler_state_dict=mock_scaler.state_dict(),
        best_val_metric=0.5,
        metrics={'loss': 0.1, 'acc': 0.9},
        config={'lr': 1e-3, 'batch_size': 32},
        callbacks_state=None,
        tokenizer_path=None,
        tensorboard_log_dir = None,
    )

# --- Basic Save/Load Tests ---

def test_save_checkpoint_creates_file(checkpoint_manager, initial_training_state, tmp_path):
    """Test that saving a checkpoint actually creates a file with expected keys."""
    filename = "test_checkpoint_step_000050.pt"
    # Use the manager's configured dir, which should point to tmp_path structure
    save_path = checkpoint_manager.checkpoint_dir / filename
    logger.debug(f"Attempting to save checkpoint to: {save_path}")

    assert not save_path.exists(), "Checkpoint file should not exist before saving."

    # Call the actual save method
    try:
        checkpoint_manager.save_checkpoint(state=initial_training_state, filename=filename, is_best=False)
    except Exception as e:
        pytest.fail(f"checkpoint_manager.save_checkpoint raised an unexpected exception: {e}")

    # Assert the file was actually created
    assert save_path.exists(), "Checkpoint file should exist after saving."
    assert save_path.is_file(), "Saved path should be a file."

    # Optionally, load and verify basic content
    try:
        loaded_dict = torch.load(save_path)
        assert isinstance(loaded_dict, dict), "Loaded checkpoint should be a dictionary."
        # Check for essential keys from TrainingState
        assert 'epoch' in loaded_dict
        assert 'global_step' in loaded_dict
        assert 'model_state_dict' in loaded_dict
        assert 'optimizer_state_dict' in loaded_dict # Check if it was saved
        # Check that the tokenizer path is None as expected
        assert loaded_dict.get('tokenizer_path') is None
    except Exception as e:
        pytest.fail(f"Failed to load or verify the created checkpoint file {save_path}: {e}")

    logger.debug(f"Successfully created and verified basic structure of checkpoint: {save_path}")

@patch("torch.load")
def test_load_checkpoint_returns_state(
    mock_torch_load, # Renamed for clarity
    checkpoint_manager, # Use the fixture
    initial_training_state, # Use the fixture
    checkpoint_dir # Use the fixture path
):
    """Test that loading a previously saved checkpoint returns a TrainingState object."""
    filename = "test_load_checkpoint.pt"
    load_path = checkpoint_dir / filename

    # --- Mock torch.load to return the state dictionary --- #
    # Use asdict to simulate the saved dictionary format
    saved_state_dict = asdict(initial_training_state)
    mock_torch_load.return_value = saved_state_dict

    # --- Create a dummy file for Path.exists() check --- #
    load_path.touch()

    # --- Call load_checkpoint --- #
    # Provide the path as a string, as Trainer would
    loaded_state = checkpoint_manager.load_checkpoint(str(load_path))

    # --- Assertions --- #
    mock_torch_load.assert_called_once_with(load_path, map_location=None, weights_only=False)
    assert isinstance(loaded_state, TrainingState), "Loaded state should be a TrainingState object"

    # Compare fields, ignoring state dicts for simplicity here (tested elsewhere)
    assert loaded_state.epoch == initial_training_state.epoch
    assert loaded_state.global_step == initial_training_state.global_step
    assert loaded_state.best_val_metric == initial_training_state.best_val_metric
    assert loaded_state.config == initial_training_state.config
    assert loaded_state.metrics == initial_training_state.metrics
    # Check specific state dicts are present (loading into components tested elsewhere)
    assert compare_state_dicts(loaded_state.model_state_dict, initial_training_state.model_state_dict), "Model state dict mismatch"
    assert compare_state_dicts(loaded_state.optimizer_state_dict, initial_training_state.optimizer_state_dict), "Optimizer state dict mismatch"
    assert compare_state_dicts(loaded_state.scheduler_state_dict, initial_training_state.scheduler_state_dict), "Scheduler state dict mismatch"
    assert compare_state_dicts(loaded_state.scaler_state_dict, initial_training_state.scaler_state_dict), "Scaler state dict mismatch"

# --- Helper Function for State Dict Comparison ---

def compare_state_dicts(dict1, dict2, rtol=1e-5, atol=1e-8, parent_key="root") -> bool:
    """Recursively compares two potentially nested state dictionaries for equality, handling tensors."""
    if type(dict1) != type(dict2):
        print(f"Type mismatch at '{parent_key}': {type(dict1)} vs {type(dict2)}")
        return False

    if isinstance(dict1, dict):
        if set(dict1.keys()) != set(dict2.keys()):
            print(f"Key mismatch at '{parent_key}': {sorted(dict1.keys())} vs {sorted(dict2.keys())}")
            return False
        for key in dict1:
            new_parent_key = f"{parent_key}.{key}"
            if not compare_state_dicts(dict1[key], dict2[key], rtol, atol, new_parent_key):
                # Error already printed by recursive call
                return False
    elif isinstance(dict1, (list, tuple)):
        if len(dict1) != len(dict2):
            print(f"Length mismatch at '{parent_key}': {len(dict1)} vs {len(dict2)}")
            return False
        for i in range(len(dict1)):
            new_parent_key = f"{parent_key}[{i}]"
            if not compare_state_dicts(dict1[i], dict2[i], rtol, atol, new_parent_key):
                # Error already printed by recursive call
                return False
    elif isinstance(dict1, torch.Tensor):
        # Safely check if dict2 is also a tensor before comparing
        if not isinstance(dict2, torch.Tensor):
             print(f"Type mismatch for tensor at '{parent_key}': {type(dict1)} vs {type(dict2)}")
             return False
        # Use torch.equal/allclose for tensors
        if dict1.dtype.is_floating_point or dict2.dtype.is_floating_point:
            if not torch.allclose(dict1, dict2, rtol=rtol, atol=atol):
                print(f"Tensor mismatch at '{parent_key}':\n{dict1}\nvs\n{dict2}")
                return False
        elif not torch.equal(dict1, dict2):
             print(f"Tensor mismatch at '{parent_key}':\n{dict1}\nvs\n{dict2}")
             return False
    else:
        # Handle other basic types (int, float, str, bool, None, etc.)
        if dict1 != dict2:
            # Add specific float comparison if standard != is not sufficient due to precision
            if isinstance(dict1, float) and isinstance(dict2, float):
                 if not torch.allclose(torch.tensor(dict1), torch.tensor(dict2), rtol=rtol, atol=atol):
                      print(f"Float mismatch at '{parent_key}': {dict1} vs {dict2}")
                      return False
            else:
                 print(f"Value mismatch at '{parent_key}': {dict1} ({type(dict1)}) vs {dict2} ({type(dict2)})")
                 return False

    return True # If all comparisons passed

# --- Detailed State Restoration Tests ---

def test_load_checkpoint_restores_model_state(
    checkpoint_manager: CheckpointManager,
    initial_training_state: TrainingState,
    checkpoint_dir: Path
):
    """Test that loading restores model weights correctly."""
    filename = "test_model_state.pt"
    # Save first
    checkpoint_manager.save_checkpoint(state=initial_training_state, filename=filename)
    save_path = checkpoint_dir / filename
    logger.debug(f"Model state test: Checkpoint saved to {save_path}")

    # Create NEW model instance (will have different random initial weights)
    model_new = nn.Linear(10, 2)
    # Capture state *before* loading
    state_before_load = {k: v.clone() for k, v in model_new.state_dict().items()}

    # Ensure new model state is different from saved state initially
    assert not compare_state_dicts(state_before_load, initial_training_state.model_state_dict), (
        "New model state should initially be different from saved state."
    )

    # Create a manager for loading into the new model
    optimizer_mock = AdamW(model_new.parameters())
    manager_for_load = CheckpointManager(
         model=model_new,
         optimizer=optimizer_mock,
         experiment_name=checkpoint_manager.experiment_name,
         config={},
         scheduler=None, scaler=None, callbacks=None, tokenizer=None, device=None,
    )
    manager_for_load.checkpoint_dir = checkpoint_dir

    # Load checkpoint
    loaded_state_obj = manager_for_load.load_checkpoint(str(save_path))
    logger.debug(f"Model state test: Checkpoint loaded from {save_path}")

    # Assert that the new model's state now matches the originally saved state
    state_after_load = model_new.state_dict()
    assert compare_state_dicts(state_after_load, initial_training_state.model_state_dict), (
        "Loaded model state should match the initially saved model state."
    )
    logger.debug("Model state test: Model state successfully restored.")


def test_load_checkpoint_restores_optimizer_state(
    checkpoint_manager: CheckpointManager, # Uses the initial components
    simple_model: nn.Module,
    simple_optimizer: AdamW,
    initial_training_state: TrainingState, # Used only as a template
    checkpoint_dir: Path
):
    """Test that loading restores optimizer state correctly."""
    filename = "test_optimizer_state.pt"
    logger.debug("Optimizer test: Preparing state with modified optimizer...")

    # Simulate a training step to change optimizer state (e.g., moments)
    simple_model.train()
    simple_optimizer.zero_grad()
    dummy_input = torch.randn(4, 10)
    dummy_output = simple_model(dummy_input)
    dummy_loss = dummy_output.mean()
    dummy_loss.backward()
    simple_optimizer.step()
    logger.debug("Optimizer test: Dummy step performed.")

    # Create state object AFTER the step, capturing the modified optimizer state
    state_to_save = TrainingState(
        epoch=initial_training_state.epoch,
        global_step=initial_training_state.global_step + 1, # Increment step
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict=simple_optimizer.state_dict(),
        scheduler_state_dict=initial_training_state.scheduler_state_dict,
        scaler_state_dict=initial_training_state.scaler_state_dict,
        best_val_metric=initial_training_state.best_val_metric,
        metrics=initial_training_state.metrics,
        config=initial_training_state.config,
    )
    # Get a representation of the optimizer state for comparison
    # Focus on the 'state' part which holds moments, etc.
    # Need a good way to compare complex optimizer states
    saved_optimizer_state_repr = simple_optimizer.state_dict() # Get full state dict

    # Save checkpoint
    checkpoint_manager.save_checkpoint(state=state_to_save, filename=filename)
    save_path = checkpoint_dir / filename
    logger.debug(f"Optimizer test: Checkpoint saved to {save_path}")

    # Create NEW components for loading
    model_new = nn.Linear(10, 2)
    optimizer_new = AdamW(model_new.parameters(), lr=1e-3)
    state_before_load = optimizer_new.state_dict()

    # Ensure new optimizer state is different initially (moments should be zero or different)
    assert len(state_before_load.get('state', {})) == 0 or not compare_state_dicts(state_before_load, saved_optimizer_state_repr), (
         "New optimizer state should initially differ from saved state."
    )

    # Create a manager for loading
    manager_for_load = CheckpointManager(
         model=model_new,
         optimizer=optimizer_new,
         experiment_name=checkpoint_manager.experiment_name,
         config={},
         scheduler=None, scaler=None, callbacks=None, tokenizer=None, device=None,
    )
    manager_for_load.checkpoint_dir = checkpoint_dir

    # Load checkpoint (which includes model state, necessary for optimizer load)
    loaded_state_obj = manager_for_load.load_checkpoint(str(save_path))
    logger.debug(f"Optimizer test: Checkpoint loaded from {save_path}")

    # Assert that the new optimizer's state matches the saved state
    state_after_load = optimizer_new.state_dict()
    assert compare_state_dicts(state_after_load, saved_optimizer_state_repr), (
        "Loaded optimizer state should match the saved optimizer state."
    )

    logger.debug("Optimizer test: Optimizer state successfully restored.")


def test_load_checkpoint_restores_scheduler_state(
    checkpoint_manager: CheckpointManager, # Uses initial components
    simple_optimizer: AdamW,
    simple_scheduler: StepLR,
    initial_training_state: TrainingState, # Template
    checkpoint_dir: Path
):
    """Test that loading restores scheduler state correctly."""
    filename = "test_scheduler_state.pt"
    logger.debug("Scheduler test: Preparing state with modified scheduler...")

    # Simulate scheduler steps to change its internal state
    num_steps = 5
    initial_scheduler_last_epoch = simple_scheduler.last_epoch
    for _ in range(num_steps):
        # Need optimizer step for scheduler step usually
        simple_optimizer.step() # Dummy step
        simple_scheduler.step()
    logger.debug(f"Scheduler test: Performed {num_steps} scheduler steps.")
    assert simple_scheduler.last_epoch == initial_scheduler_last_epoch + num_steps, "Scheduler last_epoch should have incremented."

    # Create state object AFTER the steps
    state_to_save = TrainingState(
        epoch=initial_training_state.epoch,
        global_step=initial_training_state.global_step + num_steps,
        model_state_dict=checkpoint_manager.model.state_dict(),
        optimizer_state_dict=simple_optimizer.state_dict(),
        scheduler_state_dict=simple_scheduler.state_dict(),
        scaler_state_dict=initial_training_state.scaler_state_dict,
        best_val_metric=initial_training_state.best_val_metric,
        metrics=initial_training_state.metrics,
        config=initial_training_state.config,
    )
    saved_scheduler_state = state_to_save.scheduler_state_dict

    # Save checkpoint
    checkpoint_manager.save_checkpoint(state=state_to_save, filename=filename)
    save_path = checkpoint_dir / filename
    logger.debug(f"Scheduler test: Checkpoint saved to {save_path}")

    # Create NEW components for loading
    model_new = nn.Linear(10, 2)
    optimizer_new = AdamW(model_new.parameters())
    scheduler_new = StepLR(optimizer_new, step_size=1, gamma=1.0)
    # Adjust initial assertion based on observed behavior (StepLR might start at 0)
    assert scheduler_new.last_epoch == 0, "New scheduler last_epoch should be 0 (or -1 depending on PyTorch version)."

    # Create a manager for loading
    manager_for_load = CheckpointManager(
         model=model_new,
         optimizer=optimizer_new,
         scheduler=scheduler_new,
         experiment_name=checkpoint_manager.experiment_name,
         config={},
         scaler=None, callbacks=None, tokenizer=None, device=None,
    )
    manager_for_load.checkpoint_dir = checkpoint_dir

    # Load checkpoint
    loaded_state_obj = manager_for_load.load_checkpoint(str(save_path))
    logger.debug(f"Scheduler test: Checkpoint loaded from {save_path}")

    # Assert that the new scheduler's state matches the saved state
    # Get the expected last_epoch from the saved state dict
    expected_last_epoch = saved_scheduler_state.get('last_epoch')
    assert scheduler_new.last_epoch == expected_last_epoch, (
        f"Loaded scheduler last_epoch ({scheduler_new.last_epoch}) should match saved state ({expected_last_epoch})."
    )
    # Compare full state dicts
    assert compare_state_dicts(scheduler_new.state_dict(), saved_scheduler_state), (
         "Loaded scheduler state dict should match saved state dict."
    )
    logger.debug("Scheduler test: Scheduler state successfully restored.")


def test_load_missing_scheduler_state_raises_error(
    checkpoint_manager: CheckpointManager,
    initial_training_state: TrainingState,
    checkpoint_dir: Path
):
    """Test that loading raises CheckpointLoadError if scheduler state is None or missing."""
    filename = "test_missing_scheduler.pt"

    # Save state WITHOUT scheduler info (explicitly None)
    state_to_save = TrainingState(
        epoch=initial_training_state.epoch,
        global_step=initial_training_state.global_step,
        model_state_dict=initial_training_state.model_state_dict,
        optimizer_state_dict=initial_training_state.optimizer_state_dict,
        scheduler_state_dict=None, # Explicitly None
        scaler_state_dict=initial_training_state.scaler_state_dict,
        config=initial_training_state.config,
    )
    checkpoint_manager.save_checkpoint(state=state_to_save, filename=filename)
    save_path = checkpoint_dir / filename
    logger.debug(f"Missing scheduler test: Checkpoint saved to {save_path}")

    # Create new components for loading (including a scheduler instance)
    model_new = nn.Linear(10, 2)
    optimizer_new = AdamW(model_new.parameters())
    scheduler_new = StepLR(optimizer_new, step_size=1)

    # Create a manager for loading
    manager_for_load = CheckpointManager(
         model=model_new,
         optimizer=optimizer_new,
         scheduler=scheduler_new, # Provide scheduler instance
         experiment_name=checkpoint_manager.experiment_name,
         config={},
         scaler=None, callbacks=None, tokenizer=None, device=None,
    )
    manager_for_load.checkpoint_dir = checkpoint_dir

    # Assert that loading raises the specific error for the None value case
    # Use re.escape for the path part
    expected_error_msg_regex = re.escape(f"Checkpoint {save_path} contains invalid None value for 'scheduler_state_dict'")
    with pytest.raises(CheckpointLoadError, match=expected_error_msg_regex):
        manager_for_load.load_checkpoint(str(save_path))
    logger.debug("Missing scheduler test (None state): Correctly raised CheckpointLoadError.")

    # Optional: Test the case where the key is completely missing (might need another save)
    filename_key_missing = "test_missing_scheduler_key.pt"
    # Create a dict without the key (can't use TrainingState directly easily)
    # Note: This bypasses the TrainingState validation, used only for testing raw load
    raw_dict_to_save = {
        "epoch": initial_training_state.epoch,
        "global_step": initial_training_state.global_step,
        "model_state_dict": initial_training_state.model_state_dict,
        "optimizer_state_dict": initial_training_state.optimizer_state_dict,
        # scheduler_state_dict is MISSING
        "scaler_state_dict": initial_training_state.scaler_state_dict,
        "config": initial_training_state.config,
        "callbacks_state": None,
        "tokenizer_path": None,
        "best_val_metric": 0.5,
        "metrics": {},
    }
    save_path_key_missing = checkpoint_dir / filename_key_missing
    torch.save(raw_dict_to_save, save_path_key_missing)
    logger.debug(f"Missing scheduler test (key missing): Raw checkpoint saved to {save_path_key_missing}")

    # Assert that loading raises the specific error for the missing key case
    expected_error_msg_regex_key_missing = re.escape(f"Checkpoint {save_path_key_missing} is missing required key for scheduler: 'scheduler_state_dict'")
    with pytest.raises(CheckpointLoadError, match=expected_error_msg_regex_key_missing):
        manager_for_load.load_checkpoint(str(save_path_key_missing))
    logger.debug("Missing scheduler test (key missing): Correctly raised CheckpointLoadError.")


# --- TODO: Add more tests ---
# - Test scaler state loading.
# - Test cleanup logic (_cleanup_checkpoints based on keep_last_n).
# - Test loading with DataParallel/DDP prefixes (basic check already exists).
# - Test map_location argument.
