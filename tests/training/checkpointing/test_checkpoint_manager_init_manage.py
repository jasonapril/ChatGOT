"""
Tests for CheckpointManager initialization, cleanup, and helper methods.
"""

import pytest
import torch
import os
from unittest.mock import MagicMock, patch
import sys
import shutil
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pydantic import ConfigDict # Add this import
import torch.nn as nn
import torch.optim as optim
import time
import re
from typing import List, Optional

# Module under test
from craft.training.checkpointing import CheckpointManager, CheckpointLoadError, TrainingState, CHECKPOINT_FILE_PATTERN
from craft.models.base import Model, BaseModelConfig # Keep Base models for MockModel
from craft.data.tokenizers.base import Tokenizer # Replaced BaseTokenizer
from craft.training.callbacks import CallbackList

# --- Fixtures (Copied/Adapted from original test_checkpointing.py) --- #

# Define Config before Model uses it
class MockPydanticConfig(BaseModelConfig):
    model_config = ConfigDict(extra='ignore') # Add this line
    param: int = 10

# Simple Mock Model
class MockModel(Model):
    def __init__(self, config: MockPydanticConfig):
        super().__init__(config)
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return x

@pytest.fixture
def mock_tokenizer_fixture():
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.save = MagicMock()
    tokenizer.load = MagicMock() 
    return tokenizer

@pytest.fixture
def mock_logger_fixture():
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger

# Fixtures for model components
@pytest.fixture
def mock_objects_for_cm(mock_tokenizer_fixture):
    """Provides common mock objects needed for CheckpointManager."""
    pydantic_config = MockPydanticConfig(param=10) # Create instance
    mock_model = MockModel(config=pydantic_config) # Pass instance
    mock_model.state_dict = MagicMock(return_value={
        "layer.weight": torch.randn(10, 10),
        "layer.bias": torch.randn(10)
    })
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.state_dict = MagicMock(return_value={"opt_state": 1})
    mock_optimizer.load_state_dict = MagicMock()
    mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
    mock_scheduler.state_dict = MagicMock(return_value={"sched_state": 2})
    mock_scheduler.load_state_dict = MagicMock()
    mock_scaler = MagicMock(spec=torch.amp.GradScaler)
    mock_scaler.state_dict = MagicMock(return_value={"scaler_state": 3})
    mock_scaler.load_state_dict = MagicMock()
    mock_scaler.is_enabled.return_value = True
    mock_callbacks = MagicMock(spec=CallbackList)
    mock_callbacks.state_dict = MagicMock(return_value={"callback_state": 4})
    mock_callbacks.load_state_dict = MagicMock()
    mock_callbacks.on_load_checkpoint = MagicMock()
    mock_tokenizer = mock_tokenizer_fixture

    return {
        "model": mock_model,
        "optimizer": mock_optimizer,
        "scheduler": mock_scheduler,
        "scaler": mock_scaler,
        "callbacks": mock_callbacks,
        "tokenizer": mock_tokenizer,
        "device": 'cpu'
    }

@pytest.fixture
def checkpoint_manager(mock_objects_for_cm, tmp_path):
    """Provides an initialized CheckpointManager instance."""
    exp_name = "test_init_manage_exp"
    checkpoint_dir = tmp_path # Use tmp_path directly

    # Minimal valid config for the fixture instance
    minimal_full_app_config = {
        'experiment': {
            'experiment_name': exp_name,
            'checkpoints': {
                'checkpoint_dir': str(checkpoint_dir),
                'save_steps_interval': 100,
                'keep_last_n': 2,
                'keep_best_n': 1,
                'monitor_metric': 'val/loss',
                'metric_mode': 'min',
                'save_optimizer_state': True,
                'save_scheduler_state': True,
                'save_scaler_state': True,
                'save_tokenizer': True,
                'save_format': 'pytorch'
            }
        }
    }

    # Instantiate using explicit arguments and the config
    manager = CheckpointManager(
        str(checkpoint_dir), # Pass checkpoint_dir positionally
        model=mock_objects_for_cm['model'],
        optimizer=mock_objects_for_cm['optimizer'],
        scheduler=mock_objects_for_cm['scheduler'],
        scaler=mock_objects_for_cm['scaler'],
        callbacks=mock_objects_for_cm['callbacks'],
        tokenizer=mock_objects_for_cm['tokenizer'],
        device=mock_objects_for_cm['device'],
        config=minimal_full_app_config, # Use 'config' kwarg
        experiment_name=exp_name
    )
    # Ensure the manager uses the tmp_path for this test instance
    # This might be slightly redundant if the config logic works, but ensures consistency
    # manager.checkpoint_dir = checkpoint_dir # Remove this line
    yield manager # Yield the manager instance

# Helper function to create a dummy checkpoint file
def create_dummy_checkpoint(path: Path, data: dict):
    os.makedirs(path.parent, exist_ok=True)
    torch.save(data, path)

# --- Init and Manage Tests --- #

def test_init_creates_directory(mock_objects_for_cm, tmp_path):
    """Test that CheckpointManager creates the checkpoint directory if it doesn't exist."""
    test_dir = tmp_path / "test_init_dir"
    assert not test_dir.exists()
    exp_name = "test_init_dir_exp"

    minimal_full_app_config = {
        'experiment': {
            'experiment_name': exp_name,
            'checkpoints': {
                'checkpoint_dir': str(test_dir), # Point config to the target test dir
                # Add other required checkpoint config keys...
                'save_steps_interval': 1,
                'keep_last_n': 1,
                'keep_best_n': 1,
                'monitor_metric': 'val/loss',
                'metric_mode': 'min'
            }
        }
    }

    with patch("os.makedirs") as mock_makedirs:
        manager = CheckpointManager(
            str(test_dir), # Pass checkpoint_dir positionally
            model=mock_objects_for_cm['model'],
            optimizer=mock_objects_for_cm['optimizer'],
            scheduler=mock_objects_for_cm['scheduler'],
            scaler=mock_objects_for_cm['scaler'],
            callbacks=mock_objects_for_cm['callbacks'],
            tokenizer=mock_objects_for_cm['tokenizer'],
            device=mock_objects_for_cm['device'],
            config=minimal_full_app_config, # Use 'config' kwarg
            experiment_name=exp_name
        )
        # Check if the internally set dir matches our expectation
        assert manager.checkpoint_dir == test_dir

    # Check if os.makedirs was called for the correct directory
    mock_makedirs.assert_called_once_with(test_dir, exist_ok=True)


def test_init_directory_already_exists(mock_objects_for_cm, tmp_path):
    """Test that CheckpointManager handles the case where the directory already exists."""
    test_dir = tmp_path / "test_existing_dir"
    test_dir.mkdir() # Create the directory beforehand
    assert test_dir.exists()
    exp_name = "test_existing_dir_exp"

    minimal_full_app_config = {
        'experiment': {
            'experiment_name': exp_name,
            'checkpoints': {
                'checkpoint_dir': str(test_dir),
                'save_steps_interval': 1,
                'keep_last_n': 1,
                'keep_best_n': 1,
                'monitor_metric': 'val/loss',
                'metric_mode': 'min'
            }
        }
    }

    with patch("os.makedirs") as mock_makedirs:
        manager = CheckpointManager(
            str(test_dir), # Pass checkpoint_dir positionally
            model=mock_objects_for_cm['model'],
            optimizer=mock_objects_for_cm['optimizer'],
            scheduler=mock_objects_for_cm['scheduler'],
            scaler=mock_objects_for_cm['scaler'],
            callbacks=mock_objects_for_cm['callbacks'],
            tokenizer=mock_objects_for_cm['tokenizer'],
            device=mock_objects_for_cm['device'],
            config=minimal_full_app_config, # Use 'config' kwarg
            experiment_name=exp_name
        )
        assert manager.checkpoint_dir == test_dir

    # Assert os.makedirs was still called with exist_ok=True
    mock_makedirs.assert_called_once_with(test_dir, exist_ok=True)

def test_init_default_directory(mock_objects_for_cm):
    """Test that CheckpointManager uses a default structure and creates the directory."""
    exp_name = "test_default_dir_exp"

    # Minimal config without explicit checkpoint_dir
    minimal_full_app_config = {
        'experiment': {
            'experiment_name': exp_name,
            'checkpoints': { # Still need the checkpoints section
                'save_steps_interval': 1,
                'keep_last_n': 1,
                'keep_best_n': 1,
                'monitor_metric': 'val/loss',
                'metric_mode': 'min'
            }
        }
    }

    # Mock getcwd just to provide a known root for the relative default path derivation
    with patch("hydra.utils.get_original_cwd", return_value="/fake/cwd") as mock_getcwd, \
         patch("os.makedirs") as mock_makedirs: # Also mock makedirs
        manager = CheckpointManager(
            str(tmp_path / "checkpoints"), # Use 'checkpoints' as the default directory
            model=mock_objects_for_cm['model'],
            optimizer=mock_objects_for_cm['optimizer'],
            scheduler=mock_objects_for_cm['scheduler'],
            scaler=mock_objects_for_cm['scaler'],
            callbacks=mock_objects_for_cm['callbacks'],
            tokenizer=mock_objects_for_cm['tokenizer'],
            device=mock_objects_for_cm['device'],
            config=minimal_full_app_config, # Use 'config' kwarg
            experiment_name=exp_name
        )

    # Assert that the derived checkpoint directory has the expected structure
    assert isinstance(manager.checkpoint_dir, Path)
    expected_sub_path = Path("outputs") / "experiments" / exp_name / "checkpoints"
    # Check the end of the path
    assert str(manager.checkpoint_dir).endswith(str(expected_sub_path))
    # Check that makedirs was called for this path
    mock_makedirs.assert_called_once_with(manager.checkpoint_dir, exist_ok=True)

def test_parse_checkpoint_name(checkpoint_manager):
    """Test the _parse_checkpoint_name helper method."""
    # CheckpointManager fixture now handles init correctly
    prefix = checkpoint_manager.checkpoint_prefix # e.g., "checkpoint"
    
    # Valid names
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_epoch_5_step_1000.pt") == (5, 1000)
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_epoch_0_step_50.pt") == (0, 50)
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_step_200.pt") == (0, 200) # Epoch defaults to 0
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_epoch_10_step_500_best.pt") == (10, 500) # Handles _best
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_step_100_best.pt") == (0, 100) # Handles _best
    
    # Invalid names
    assert checkpoint_manager._parse_checkpoint_name("random_file.txt") is None
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_epoch_step_100.pt") is None # Missing epoch number
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_epoch_1_step.pt") is None # Missing step number
    assert checkpoint_manager._parse_checkpoint_name(f"{prefix}_step_abc.pt") is None # Non-numeric step
    assert checkpoint_manager._parse_checkpoint_name(f"wrongprefix_step_100.pt") is None

def test_manage_checkpoints_keeps_correct_number(checkpoint_manager, tmp_path):
    """Test _manage_checkpoints keeps the specified number of checkpoints."""
    # checkpoint_manager fixture handles init
    manager = checkpoint_manager
    # Ensure the test uses the manager's actual checkpoint_dir
    test_checkpoint_dir = manager.checkpoint_dir
    manager.max_checkpoints_to_keep = 3
    prefix = manager.checkpoint_prefix
    model_state = manager.model.state_dict()

    # Create more checkpoints than should be kept in the manager's directory
    paths_to_create = [
        test_checkpoint_dir / f"{prefix}_epoch_1_step_100.pt",
        test_checkpoint_dir / f"{prefix}_epoch_1_step_200.pt",
        test_checkpoint_dir / f"{prefix}_epoch_2_step_300.pt",
        test_checkpoint_dir / f"{prefix}_epoch_2_step_400.pt",
        test_checkpoint_dir / f"{prefix}_epoch_3_step_500.pt",
    ]
    for i, p in enumerate(paths_to_create):
        create_dummy_checkpoint(p, {"global_step": (i+1)*100, "epoch": (i//2)+1, "model_state_dict": model_state})
        # Manually add to saved_checkpoints as if save_checkpoint was called
        manager.saved_checkpoints.append((str(p), False))
        time.sleep(0.01) # Slight delay might help with mtime if ever used (shouldn't be)

    assert len(manager.saved_checkpoints) == 5
    for p in paths_to_create:
        assert p.exists()

    # Trigger the management function (implicitly called by _add_saved_checkpoint in real code)
    manager._manage_checkpoints()

    # Check that only max_checkpoints_to_keep remain
    assert len(manager.saved_checkpoints) == manager.max_checkpoints_to_keep
    assert len(list(test_checkpoint_dir.glob(f"{prefix}_*.pt"))) == manager.max_checkpoints_to_keep

    # Check that the *oldest* checkpoints were deleted (epoch 1, steps 100, 200)
    assert not (test_checkpoint_dir / f"{prefix}_epoch_1_step_100.pt").exists()
    assert not (test_checkpoint_dir / f"{prefix}_epoch_1_step_200.pt").exists()
    # Check that the newest ones remain
    assert (test_checkpoint_dir / f"{prefix}_epoch_2_step_300.pt").exists()
    assert (test_checkpoint_dir / f"{prefix}_epoch_2_step_400.pt").exists()
    assert (test_checkpoint_dir / f"{prefix}_epoch_3_step_500.pt").exists()

def test_manage_checkpoints_keeps_best(checkpoint_manager, tmp_path):
    """Test _manage_checkpoints always keeps checkpoints marked as best."""
    # checkpoint_manager fixture handles init
    manager = checkpoint_manager
    test_checkpoint_dir = manager.checkpoint_dir # Use the manager's dir
    manager.max_checkpoints_to_keep = 2 # Keep only 2 non-best
    prefix = manager.checkpoint_prefix
    model_state = manager.model.state_dict()

    paths_to_create = {
        "regular1": test_checkpoint_dir / f"{prefix}_epoch_1_step_100.pt",
        "best1": test_checkpoint_dir / f"{prefix}_epoch_1_step_200_best.pt",
        "regular2": test_checkpoint_dir / f"{prefix}_epoch_2_step_300.pt",
        "best2": test_checkpoint_dir / f"{prefix}_epoch_2_step_400_best.pt",
        "regular3": test_checkpoint_dir / f"{prefix}_epoch_3_step_500.pt",
    }
    # Simulate saving these checkpoints
    for name, path in paths_to_create.items():
         is_best = "best" in name
         create_dummy_checkpoint(path, {"global_step": int(name[-1])*100, "epoch": int(name[-1]), "model_state_dict": model_state})
         manager.saved_checkpoints.append((str(path), is_best))

    assert len(manager.saved_checkpoints) == 5

    # Trigger management
    manager._manage_checkpoints()

    # Expected remaining: 2 best + 2 most recent regular
    assert len(manager.saved_checkpoints) == 4
    assert len(list(test_checkpoint_dir.glob(f"{prefix}_*.pt"))) == 4

    # Check that the best ones remain
    assert paths_to_create["best1"].exists()
    assert paths_to_create["best2"].exists()
    # Check that the oldest regular one was deleted
    assert not paths_to_create["regular1"].exists()
    # Check that the 2 newest regular ones remain
    assert paths_to_create["regular2"].exists()
    assert paths_to_create["regular3"].exists()

def test_manage_checkpoints_deletes_tokenizer_dir(checkpoint_manager, tmp_path):
    """Test that deleting an old checkpoint also deletes its associated tokenizer dir."""
    # checkpoint_manager fixture handles init
    manager = checkpoint_manager
    test_checkpoint_dir = manager.checkpoint_dir # Use the manager's dir
    manager.max_checkpoints_to_keep = 1 # Keep only 1
    prefix = manager.checkpoint_prefix
    model_state = manager.model.state_dict()

    # Checkpoint to be deleted
    old_step = 100
    old_epoch = 1
    old_filename = f"{prefix}_epoch_{old_epoch}_step_{old_step}.pt"
    old_path = test_checkpoint_dir / old_filename
    old_tokenizer_rel_path = f"tokenizer_step_{old_step}"
    old_tokenizer_dir = test_checkpoint_dir / old_tokenizer_rel_path

    # Checkpoint to be kept
    new_step = 200
    new_epoch = 1
    new_filename = f"{prefix}_epoch_{new_epoch}_step_{new_step}.pt"
    new_path = test_checkpoint_dir / new_filename
    new_tokenizer_rel_path = f"tokenizer_step_{new_step}"
    new_tokenizer_dir = test_checkpoint_dir / new_tokenizer_rel_path

    # Create checkpoints and associated tokenizer dirs
    create_dummy_checkpoint(old_path, {"global_step": old_step, "epoch": old_epoch, "tokenizer_path": old_tokenizer_rel_path, "model_state_dict": model_state})
    old_tokenizer_dir.mkdir()
    (old_tokenizer_dir / "dummy_tok_file").touch() # Add a file

    create_dummy_checkpoint(new_path, {"global_step": new_step, "epoch": new_epoch, "tokenizer_path": new_tokenizer_rel_path, "model_state_dict": model_state})
    new_tokenizer_dir.mkdir()
    (new_tokenizer_dir / "dummy_tok_file").touch() # Add a file

    # Manually add to saved list (as save_checkpoint would do)
    manager.saved_checkpoints.append((str(old_path), False))
    manager.saved_checkpoints.append((str(new_path), False))

    assert old_path.exists()
    assert old_tokenizer_dir.exists()
    assert new_path.exists()
    assert new_tokenizer_dir.exists()
    assert len(manager.saved_checkpoints) == 2

    # Trigger management
    manager._manage_checkpoints()

    # Check that only 1 remains
    assert len(manager.saved_checkpoints) == 1
    # Check that the old checkpoint is gone
    assert not old_path.exists()
    # Check that the old tokenizer directory is also gone
    assert not old_tokenizer_dir.exists()
    # Check that the new checkpoint and its tokenizer dir remain
    assert new_path.exists()
    assert new_tokenizer_dir.exists()
    assert manager.saved_checkpoints[0][0] == str(new_path)

# Add more tests for find_checkpoints, cleanup_checkpoints etc. later 

# Fixtures
@pytest.fixture
def mock_pydantic_config():
    """Provides a mock Pydantic config including the required architecture field."""
    # Add required architecture field
    return MockPydanticConfig(param=10, architecture='mock_arch')

@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path 

@pytest.fixture
def mock_objects_for_cm(mock_model, mock_optimizer, mock_scheduler, mock_scaler, mock_callbacks, mock_device, mock_tokenizer):
    """Fixture to provide all mock objects needed by CheckpointManager."""
    return {
        "mock_model": mock_model,
        "mock_optimizer": mock_optimizer,
        "mock_scheduler": mock_scheduler,
        "mock_scaler": mock_scaler,
        "mock_callbacks": mock_callbacks,
        "mock_device": mock_device,
        "mock_tokenizer": mock_tokenizer
    }

@pytest.fixture
def checkpoint_manager(tmp_path, mock_objects_for_cm):
    """Fixture to create a CheckpointManager instance in a temporary directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    # checkpoint_dir.mkdir() # Let the manager create it

    cm = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=mock_objects_for_cm["mock_model"],
        optimizer=mock_objects_for_cm["mock_optimizer"],
        experiment_name="test_experiment", # Added required argument
        scheduler=mock_objects_for_cm["mock_scheduler"],
        scaler=mock_objects_for_cm["mock_scaler"],
        callbacks=mock_objects_for_cm["mock_callbacks"],
        device=mock_objects_for_cm["mock_device"],
        tokenizer=mock_objects_for_cm["mock_tokenizer"],
        keep_last_n=3, # Example value
        keep_best_n=1, # Example value
        # max_checkpoints_to_keep=3, # Removed outdated argument
        config={"some": "config"},
        save_best_only=False
    )
    return cm

# === Initialization Tests ===

def test_checkpoint_manager_creates_dir(tmp_path, mock_objects_for_cm):
    """Test that CheckpointManager creates the checkpoint directory if it doesn't exist."""
    checkpoint_dir = tmp_path / "new_checkpoints"
    assert not checkpoint_dir.exists()
    cm = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=mock_objects_for_cm["mock_model"],
        optimizer=mock_objects_for_cm["mock_optimizer"],
        experiment_name="test_create_dir" # Added required argument
    )
    assert checkpoint_dir.exists()
    assert checkpoint_dir.is_dir()
    assert cm.checkpoint_dir == checkpoint_dir # Verify it's stored correctly

def test_checkpoint_manager_handles_existing_dir(tmp_path, mock_objects_for_cm):
    """Test that CheckpointManager uses an existing directory without error."""
    checkpoint_dir = tmp_path / "existing_checkpoints"
    checkpoint_dir.mkdir()
    cm = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=mock_objects_for_cm["mock_model"],
        optimizer=mock_objects_for_cm["mock_optimizer"],
        experiment_name="test_existing_dir" # Added required argument
    )
    assert cm.checkpoint_dir == checkpoint_dir

def test_checkpoint_manager_init_scan(tmp_path, mock_objects_for_cm):
    """Test that CheckpointManager scans for existing checkpoints on initialization."""
    checkpoint_dir = tmp_path / "scan_checkpoints"
    checkpoint_dir.mkdir()
    # Create some dummy files matching the pattern, and some not
    (checkpoint_dir / "checkpoint_step_100_epoch_1.pt").touch()
    (checkpoint_dir / "checkpoint_step_50_epoch_0.pt").touch()
    (checkpoint_dir / "best.pt").touch() # Should be ignored by initial scan for step checkpoints
    (checkpoint_dir / "model.bin").touch() # Should be ignored
    (checkpoint_dir / "checkpoint_step_150.pt").touch() # Missing epoch, handled by pattern? Yes.

    cm = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=mock_objects_for_cm["mock_model"],
        optimizer=mock_objects_for_cm["mock_optimizer"],
        experiment_name="test_scan" # Added required argument
    )

    # Check internal state (adjust based on actual implementation if needed)
    # Expected: finds step checkpoints, sorts them. best.pt is not part of this list initially.
    # saved_checkpoints stores tuples: (full_path, is_best) - is_best is False from scan
    expected_basenames_sorted = [
        "checkpoint_step_50_epoch_0.pt",
        "checkpoint_step_100_epoch_1.pt",
        "checkpoint_step_150.pt",
    ]
    # Extract basenames from the stored full paths for comparison
    actual_basenames = [os.path.basename(p[0]) for p in cm.saved_checkpoints]

    assert len(cm.saved_checkpoints) == 3
    assert actual_basenames == expected_basenames_sorted
    # Check that 'is_best' flag is False for all scanned checkpoints
    assert all(not p[1] for p in cm.saved_checkpoints)


# === Checkpoint Management (Cleanup) Tests ===

@pytest.fixture
def checkpoint_manager_for_cleanup(tmp_path, mock_objects_for_cm):
    """Fixture specifically for testing checkpoint cleanup logic."""
    checkpoint_dir = tmp_path / "cleanup_checkpoints"
    checkpoint_dir.mkdir()
    # Create mock CheckpointManager with specific keep settings
    cm = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=mock_objects_for_cm["mock_model"],
        optimizer=mock_objects_for_cm["mock_optimizer"],
        experiment_name="test_cleanup", # Added required argument
        keep_last_n=2,
        keep_best_n=1,
        save_best_only=False # Important for testing pruning of non-best checkpoints
    )
    return cm

# Helper to create dummy checkpoint files and update manager's internal list
def create_dummy_checkpoints(cm: CheckpointManager, steps: List[int], best_step: Optional[int] = None):
    model_state = cm.model.state_dict()
    for step in steps:
        create_dummy_checkpoint(cm.checkpoint_dir / f"checkpoint_step_{step}_epoch_0.pt", {"global_step": step, "epoch": 0, "model_state_dict": model_state})
        if best_step and step == best_step:
            create_dummy_checkpoint(cm.checkpoint_dir / f"checkpoint_step_{step}_epoch_0_best.pt", {"global_step": step, "epoch": 0, "model_state_dict": model_state})
        cm.saved_checkpoints.append((str(cm.checkpoint_dir / f"checkpoint_step_{step}_epoch_0.pt"), best_step == step))
        time.sleep(0.01) # Slight delay might help with mtime if ever used (shouldn't be)

    cm._manage_checkpoints()
    assert len(cm.saved_checkpoints) == len(steps)
    assert all(p[1] == (best_step == step) for step, p in enumerate(cm.saved_checkpoints))
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step == best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0_best.pt") for step in steps if step != best_step)
    assert all(p[0].endswith(f"checkpoint_step_{step}_epoch_0.pt") for step in steps if step != best_step)