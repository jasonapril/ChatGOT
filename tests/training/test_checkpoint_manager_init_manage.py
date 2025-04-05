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
import torch.nn as nn
import torch.optim as optim
import time
import re

# Module under test
from craft.training.checkpointing import CheckpointManager, CheckpointLoadError, TrainingState
from craft.models.base import Model, BaseModelConfig # Keep Base models for MockModel
from craft.data.tokenizers.base import BaseTokenizer
from craft.training.callbacks import CallbackList

# --- Fixtures (Copied/Adapted from original test_checkpointing.py) --- #

# Simple Mock Model
class MockModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return x

class MockPydanticConfig(BaseModelConfig):
    param: int = 10

@pytest.fixture
def mock_tokenizer_fixture():
    tokenizer = MagicMock(spec=BaseTokenizer)
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
    mock_config = {'param': 20}
    mock_model = MockModel(config=MockPydanticConfig(param=10))
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
        "config": mock_config,
        "callbacks": mock_callbacks,
        "tokenizer": mock_tokenizer,
        "device": 'cpu'
    }

@pytest.fixture
def checkpoint_manager(mock_objects_for_cm, tmp_path):
    """Provides an initialized CheckpointManager instance."""
    # Patch os.getcwd for consistent directory behavior in tests
    with patch('os.getcwd', return_value=str(tmp_path)):
        # Instantiate using the dictionary
        manager = CheckpointManager(**mock_objects_for_cm)
        # Explicitly set checkpoint_dir for clarity in tests using tmp_path
        # Ensure checkpoint_dir is a Path object, not a string
        manager.checkpoint_dir = tmp_path
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

    with patch("os.makedirs") as mock_makedirs:
        manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=test_dir)

    # Assert directory path is stored correctly as Path
    expected_path = Path(test_dir)
    assert manager.checkpoint_dir == expected_path # Compare Path objects
    # Assert os.makedirs was called correctly
    mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


def test_init_directory_already_exists(mock_objects_for_cm, tmp_path):
    """Test that CheckpointManager handles the case where the directory already exists."""
    test_dir = tmp_path / "test_existing_dir"
    test_dir.mkdir() # Create the directory beforehand
    assert test_dir.exists()

    with patch("os.makedirs") as mock_makedirs:
        manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=test_dir)

    # Assert directory path is stored correctly as Path
    expected_path = Path(test_dir)
    assert manager.checkpoint_dir == expected_path # Compare Path objects
    # Assert os.makedirs was still called with exist_ok=True
    mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)


def test_init_default_directory(mock_objects_for_cm):
    """Test that CheckpointManager uses './checkpoints' if no directory is provided."""
    # Mock os.getcwd to return a predictable path
    with patch("os.getcwd", return_value="/fake/cwd"), patch("os.makedirs") as mock_makedirs:
        manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=None)

    # expected_path_str = os.path.join("/fake/cwd", "checkpoints") # Old string version
    expected_path = Path("/fake/cwd") / "checkpoints" # Path object version

    assert manager.checkpoint_dir == expected_path # Compare Path objects
    # Assert os.makedirs was called with the correct default Path
    mock_makedirs.assert_called_once_with(expected_path, exist_ok=True)

def test_parse_checkpoint_name(checkpoint_manager):
    """Test the _parse_checkpoint_name helper method."""
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
    manager = checkpoint_manager
    manager.max_checkpoints_to_keep = 3
    prefix = manager.checkpoint_prefix
    model_state = manager.model.state_dict()
    
    # Create more checkpoints than should be kept
    paths_to_create = [
        tmp_path / f"{prefix}_epoch_1_step_100.pt",
        tmp_path / f"{prefix}_epoch_1_step_200.pt",
        tmp_path / f"{prefix}_epoch_2_step_300.pt",
        tmp_path / f"{prefix}_epoch_2_step_400.pt",
        tmp_path / f"{prefix}_epoch_3_step_500.pt",
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
    assert len(list(tmp_path.glob(f"{prefix}_*.pt"))) == manager.max_checkpoints_to_keep

    # Check that the *oldest* checkpoints were deleted (epoch 1, steps 100, 200)
    assert not (tmp_path / f"{prefix}_epoch_1_step_100.pt").exists()
    assert not (tmp_path / f"{prefix}_epoch_1_step_200.pt").exists()
    # Check that the newest ones remain
    assert (tmp_path / f"{prefix}_epoch_2_step_300.pt").exists()
    assert (tmp_path / f"{prefix}_epoch_2_step_400.pt").exists()
    assert (tmp_path / f"{prefix}_epoch_3_step_500.pt").exists()

def test_manage_checkpoints_keeps_best(checkpoint_manager, tmp_path):
    """Test _manage_checkpoints always keeps checkpoints marked as best."""
    manager = checkpoint_manager
    manager.max_checkpoints_to_keep = 2 # Keep only 2 non-best
    prefix = manager.checkpoint_prefix
    model_state = manager.model.state_dict()

    paths_to_create = {
        "regular1": tmp_path / f"{prefix}_epoch_1_step_100.pt",
        "best1": tmp_path / f"{prefix}_epoch_1_step_200_best.pt",
        "regular2": tmp_path / f"{prefix}_epoch_2_step_300.pt",
        "best2": tmp_path / f"{prefix}_epoch_2_step_400_best.pt",
        "regular3": tmp_path / f"{prefix}_epoch_3_step_500.pt",
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
    assert len(list(tmp_path.glob(f"{prefix}_*.pt"))) == 4

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
    manager = checkpoint_manager
    manager.max_checkpoints_to_keep = 1
    prefix = manager.checkpoint_prefix
    model_state = manager.model.state_dict()
    
    # Checkpoint 1 (will be deleted)
    step1 = 100
    epoch1 = 1
    ckpt1_path = tmp_path / f"{prefix}_epoch_{epoch1}_step_{step1}.pt"
    tokenizer1_dir = tmp_path / f"tokenizer_step_{step1}"
    create_dummy_checkpoint(ckpt1_path, {"global_step": step1, "epoch": epoch1, "model_state_dict": model_state, "tokenizer_path": f"tokenizer_step_{step1}"})
    os.makedirs(tokenizer1_dir, exist_ok=True)
    (tokenizer1_dir / "file.txt").touch()
    manager.saved_checkpoints.append((str(ckpt1_path), False))
    
    # Checkpoint 2 (will be kept)
    step2 = 200
    epoch2 = 1
    ckpt2_path = tmp_path / f"{prefix}_epoch_{epoch2}_step_{step2}.pt"
    tokenizer2_dir = tmp_path / f"tokenizer_step_{step2}"
    create_dummy_checkpoint(ckpt2_path, {"global_step": step2, "epoch": epoch2, "model_state_dict": model_state, "tokenizer_path": f"tokenizer_step_{step2}"})
    os.makedirs(tokenizer2_dir, exist_ok=True)
    (tokenizer2_dir / "file.txt").touch()
    manager.saved_checkpoints.append((str(ckpt2_path), False))
    
    assert ckpt1_path.exists()
    assert tokenizer1_dir.exists()
    assert ckpt2_path.exists()
    assert tokenizer2_dir.exists()
    assert len(manager.saved_checkpoints) == 2

    # Trigger management
    manager._manage_checkpoints()

    # Checkpoint 1 and its tokenizer dir should be deleted
    assert not ckpt1_path.exists()
    assert not tokenizer1_dir.exists()
    # Checkpoint 2 and its tokenizer dir should remain
    assert ckpt2_path.exists()
    assert tokenizer2_dir.exists()
    assert len(manager.saved_checkpoints) == 1
    assert manager.saved_checkpoints[0][0] == str(ckpt2_path)

def test_get_latest_checkpoint_path_no_checkpoints(checkpoint_manager, tmp_path):
    """Test load_checkpoint with 'latest' when the directory is empty returns None and logs warning."""
    # Ensure the directory exists but is empty
    assert checkpoint_manager.checkpoint_dir == tmp_path
    assert not list(checkpoint_manager.checkpoint_dir.glob("*.pt"))

    # Patch logger specifically for this test
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        # Re-assign the logger instance inside the CheckpointManager to our mock
        checkpoint_manager.logger = mock_logger 

        # Call load_checkpoint with path="latest"
        result = checkpoint_manager.load_checkpoint(path="latest")

        # Assert that the result is None
        assert result is None

        # Check for the warning log
        mock_logger.warning.assert_called_once_with("No checkpoints found to load.")

# Add more tests for find_checkpoints, cleanup_checkpoints etc. later 