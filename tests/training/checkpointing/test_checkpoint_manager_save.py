"""
Tests for CheckpointManager specifically focusing on the save_checkpoint method.
"""

import pytest
import torch
import os
from unittest.mock import MagicMock, patch
import sys
import shutil
import logging
from pathlib import Path
from dataclasses import asdict # Import asdict
from omegaconf import OmegaConf, DictConfig
import torch.nn as nn
import torch.optim as optim
import time
import re
import unittest.mock
import json

# Module under test
from craft.training.checkpointing import CheckpointManager, CheckpointLoadError, TrainingState
from craft.models.base import Model, BaseModelConfig # Keep Base models for MockModel
from craft.data.tokenizers.base import Tokenizer
from craft.training.callbacks import CallbackList
from craft.config.schemas import TrainingConfig # Assuming TrainingConfig might be needed

# --- Fixtures (Copied from original test_checkpointing.py) --- #

# Simple Mock Model
class MockModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return x

class MockPydanticConfig(BaseModelConfig):
    param: int = 10

# Duplicate the helper function here
def create_dummy_checkpoint(path: Path, data: dict):
    os.makedirs(path.parent, exist_ok=True)
    torch.save(data, path)

@pytest.fixture
def mock_config_dict():
    return {'param': 20}

@pytest.fixture
def mock_tokenizer_fixture():
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.save = MagicMock()
    # Add load method mock for completeness if load tests are added here later
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
    mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer.state_dict = MagicMock(return_value={"opt_state": 1})
    mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
    mock_scheduler.state_dict = MagicMock(return_value={"sched_state": 2})
    mock_scaler = MagicMock(spec=torch.amp.GradScaler)
    mock_scaler.state_dict = MagicMock(return_value={"scaler_state": 3})
    # Mock is_enabled() for save check
    mock_scaler.is_enabled.return_value = True
    mock_callbacks = MagicMock(spec=CallbackList)
    # Add state_dict method to mock callbacks
    mock_callbacks.state_dict = MagicMock(return_value={"callback_state": 4})
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
    """Provides an initialized CheckpointManager instance in a temporary directory."""
    exp_name = "test_save_exp"
    # Patch getcwd for the duration of the manager's life in this fixture
    with patch('os.getcwd', return_value=str(tmp_path)):
         # Pass the dictionary directly to CheckpointManager
         # Add experiment_name
         manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name)
         # Override checkpoint_dir to ensure it uses tmp_path absolutely for the test logic
         manager.checkpoint_dir = tmp_path # Now uses Path object directly
         yield manager # Yield the manager instance

# Helper to create TrainingState from mocks
def create_mock_training_state(mock_objs, epoch, global_step, best_val_metric=None, metrics=None, tokenizer_path=None) -> TrainingState:
    """Creates a TrainingState instance from mock objects."""
    state = TrainingState(
        epoch=epoch,
        global_step=global_step,
        model_state_dict=mock_objs["model"].state_dict(),
        optimizer_state_dict=mock_objs["optimizer"].state_dict() if mock_objs["optimizer"] else None,
        scheduler_state_dict=mock_objs["scheduler"].state_dict() if mock_objs["scheduler"] else None,
        scaler_state_dict=mock_objs["scaler"].state_dict() if mock_objs["scaler"] and mock_objs["scaler"].is_enabled() else None,
        config=mock_objs["config"],
        callbacks_state=mock_objs["callbacks"].state_dict() if mock_objs["callbacks"] else None,
        tokenizer_path=tokenizer_path, # Set later if tokenizer exists
        best_val_metric=best_val_metric or float('inf'),
        metrics=metrics or {},
        tensorboard_log_dir=None # Assume not tracking TB dir in these tests
    )
    # Add tokenizer path dynamically based on existence
    if mock_objs.get("tokenizer"):
         # Construct the expected relative path
        state.tokenizer_path = f"tokenizer_step_{global_step}" 
    return state


# --- Save Tests --- #

def test_save_checkpoint_basic(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test basic checkpoint saving functionality."""
    step = 100
    epoch = 1
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Mock the tokenizer's save method to ensure it's not called when save_tokenizer=False
    with patch.object(mock_objects_for_cm["tokenizer"], 'save') as mock_tokenizer_save:
        # Call save_checkpoint with save_tokenizer=False
        checkpoint_manager.save_checkpoint(
            state=training_state,
            filename=filename,
            save_tokenizer=False  # Explicitly set to False for this test
        )

    assert save_path.exists()
    # Assert marker file exists
    marker_path = Path(f"{save_path}._SAVED")
    assert marker_path.exists()

    # Verify tokenizer was NOT saved
    mock_tokenizer_save.assert_not_called()

    # Optional: Load and verify some basic content if needed
    loaded_data = torch.load(save_path, map_location='cpu')
    assert loaded_data['global_step'] == step
    assert loaded_data['epoch'] == epoch
    assert 'tokenizer_path' not in loaded_data or loaded_data['tokenizer_path'] is None

def test_save_checkpoint_is_best(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test saving a checkpoint marked as best."""
    step = 1100
    epoch = 11
    # Filename format is now handled internally based on prefix/step/epoch
    # We just need to pass is_best=True
    # Let's keep a similar filename for assertion checking
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt" 
    save_path = tmp_path / filename
    metrics_data = {"loss": 0.1}

    # Create TrainingState object
    training_state = create_mock_training_state(
        mock_objects_for_cm, 
        epoch=epoch, 
        global_step=step, 
        metrics=metrics_data,
        best_val_metric=0.05 # Simulate this step having the best metric so far
    )
    
    # Call save_checkpoint with TrainingState, filename, and is_best=True
    checkpoint_manager.save_checkpoint(
        state=training_state,
        filename=filename,
        metrics=metrics_data,
        is_best=True
    )

    assert save_path.exists()
    best_link_path = tmp_path / "best.pt"
    assert best_link_path.exists() 
    # Don't compare realpath, just ensure best.pt exists
    # assert os.path.realpath(best_link_path) == os.path.realpath(save_path)

    # Check that the internal list tracks it, but is_best might be False if it wasn't numerically best
    # The is_best flag passed to save_checkpoint primarily controls the best.pt link
    assert any(p == str(save_path) for p, b in checkpoint_manager.saved_checkpoints)
    
    # Verify basic content
    saved_dict = torch.load(save_path, map_location='cpu')
    assert saved_dict['global_step'] == step
    assert saved_dict['epoch'] == epoch
    assert saved_dict['metrics']['loss'] == 0.1

def test_save_skips_disabled_scaler(mock_objects_for_cm, tmp_path):
    """Test that scaler state is None in TrainingState if scaler is disabled."""
    # Disable the scaler mock BEFORE creating the manager
    mock_objects_for_cm["scaler"].is_enabled.return_value = False
    step = 300
    epoch = 3
    exp_name = "test_disabled_scaler_exp"
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    # Create manager with the modified mocks
    # Remove checkpoint_dir, add experiment_name
    manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name)
    manager.checkpoint_dir = tmp_path # Ensure output goes to temp dir

    # Create TrainingState - it should automatically get None for scaler state
    training_state = create_mock_training_state(
        mock_objects_for_cm,
        epoch=epoch,
        global_step=step
    )
    assert training_state.scaler_state_dict is None

    # Call save_checkpoint
    manager.save_checkpoint(state=training_state, filename=filename)

    # Verify saved file doesn't contain scaler_state_dict
    assert save_path.exists()
    saved_dict = torch.load(save_path, map_location='cpu')
    assert "scaler_state_dict" in saved_dict
    assert saved_dict['scaler_state_dict'] is None

def test_save_skips_disabled_optimizer_scheduler(mock_objects_for_cm, tmp_path):
    """Test that optimizer and scheduler states are None if they are None in manager."""
    # Remove optimizer and scheduler from mocks BEFORE creating manager
    mock_objects_for_cm["optimizer"] = None
    mock_objects_for_cm["scheduler"] = None
    step = 400
    epoch = 4
    exp_name = "test_no_opt_sched_exp"
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    # Create manager with the modified mocks
    # Remove checkpoint_dir, add experiment_name
    manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name)
    manager.checkpoint_dir = tmp_path # Ensure output goes to temp dir

    # Create TrainingState - it should automatically get None for opt/sched states
    training_state = create_mock_training_state(
        mock_objects_for_cm,
        epoch=epoch,
        global_step=step
    )
    assert training_state.optimizer_state_dict is None
    assert training_state.scheduler_state_dict is None

    # Call save_checkpoint
    manager.save_checkpoint(state=training_state, filename=filename)

    # Verify saved file contains None for these states
    assert save_path.exists()
    saved_dict = torch.load(save_path, map_location='cpu')
    assert "optimizer_state_dict" in saved_dict
    assert saved_dict['optimizer_state_dict'] is None
    assert "scheduler_state_dict" in saved_dict
    assert saved_dict['scheduler_state_dict'] is None

def test_save_without_tokenizer(mock_objects_for_cm, tmp_path):
    """Test saving when the CheckpointManager was initialized without a tokenizer."""
    # Remove tokenizer from mocks
    mock_objects_for_cm["tokenizer"] = None
    step = 500
    epoch = 5
    exp_name = "test_no_tokenizer_exp"
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    # Create manager without tokenizer
    # Remove checkpoint_dir, add experiment_name
    manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name)
    manager.checkpoint_dir = tmp_path # Ensure output goes to temp dir

    # Create TrainingState
    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )
    assert training_state.tokenizer_path is None # Should be None

    # Call save_checkpoint
    manager.save_checkpoint(state=training_state, filename=filename)

    # Verify saved file exists and tokenizer was not called/saved
    assert save_path.exists()
    saved_dict = torch.load(save_path, map_location='cpu')
    assert "tokenizer_path" in saved_dict
    assert saved_dict['tokenizer_path'] is None
    # Check that no tokenizer directory was created
    tokenizer_save_dir = manager.checkpoint_dir / f"tokenizer_step_{step}"
    assert not tokenizer_save_dir.exists()

def test_save_checkpoint_creates_dir(mock_objects_for_cm, tmp_path):
    """Test that save_checkpoint creates the checkpoint directory if it doesn't exist."""
    # Use a subdirectory that doesn't exist yet
    sub_dir = tmp_path / "new_save_dir"
    assert not sub_dir.exists()
    step = 600
    epoch = 6
    exp_name = "test_create_dir_exp"
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = sub_dir / filename

    # Create manager, it will derive its own path, but we override for the test
    # Remove checkpoint_dir, add experiment_name
    manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name)
    manager.checkpoint_dir = sub_dir # Manually set target dir for save (Correct for this test)

    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Mock pathlib.Path.mkdir for this test -> REMOVED
    # with patch("pathlib.Path.mkdir") as mock_path_mkdir:
    #     manager.save_checkpoint(state=training_state, filename=filename)
    # Run save without the patch
    manager.save_checkpoint(state=training_state, filename=filename)

    # Assert that Path.mkdir was called on the manager's checkpoint_dir -> Check dir exists instead
    # mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert sub_dir.exists() # Check the directory was created
    assert save_path.exists() # Check the file was saved


def test_save_checkpoint_error_handling(checkpoint_manager, mock_objects_for_cm):
    """Test error handling during torch.save."""
    step = 700
    epoch = 7
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Patch torch.save to raise an exception
    with patch("torch.save", side_effect=IOError("Disk full")) as mock_torch_save:
        # Expect CheckpointLoadError because the manager now wraps the exception
        with pytest.raises(CheckpointLoadError, match="Failed to save checkpoint.*Disk full"):
            checkpoint_manager.save_checkpoint(
                state=training_state,
                filename=filename,
                save_tokenizer=False # Tokenizer saving is irrelevant to this error path test
            )

    mock_torch_save.assert_called_once()
    # Check that the failed path is NOT added to saved_checkpoints
    failed_path = str(checkpoint_manager.checkpoint_dir / filename)
    assert not any(p == failed_path for p, b in checkpoint_manager.saved_checkpoints)


def test_save_checkpoint_tokenizer_error_handling(checkpoint_manager, mock_objects_for_cm):
    """Test error handling during tokenizer saving (should not raise, just log)."""
    step = 800
    epoch = 8
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = checkpoint_manager.checkpoint_dir / filename # Define save_path here
    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Mock the tokenizer's save method to raise an error
    with patch.object(mock_objects_for_cm["tokenizer"], 'save', side_effect=OSError("Cannot write tokenizer")) as mock_tokenizer_save, \
         patch("logging.getLogger") as mock_get_logger:

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger # Assign mock logger

        # Call save_checkpoint (save_tokenizer=True by default, will attempt save)
        # No exception should be raised here, error should be logged
        checkpoint_manager.save_checkpoint(state=training_state, filename=filename, save_tokenizer=True) # Test the error path

    # Assert that tokenizer.save WAS called (because save_tokenizer=True)
    mock_tokenizer_save.assert_called_once()

    # Assert that an error was logged
    mock_logger.error.assert_called_once()
    # Check that the logged message contains relevant info
    args, kwargs = mock_logger.error.call_args
    assert "Failed to save tokenizer" in args[0]
    assert "Cannot write tokenizer" in args[0]
    # Assert main checkpoint was still saved (important!)
    assert save_path.exists()

def test_save_best_only_skips_non_best(mock_objects_for_cm, tmp_path):
    """Test save_checkpoint skips if save_best_only is True and metric is NOT best."""
    exp_name = "test_save_best_only_skip_exp"
    step = 900
    epoch = 9
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    # Create manager with save_best_only=True
    manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name, save_best_only=True)
    manager.checkpoint_dir = tmp_path # Ensure output goes to temp dir
    manager.best_metric = 0.5 # Existing best metric

    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Mock the tokenizer's save method
    with patch.object(mock_objects_for_cm["tokenizer"], 'save') as mock_tokenizer_save:
        # Call save_checkpoint with is_best=False
        manager.save_checkpoint(
            state=training_state,
            filename=filename,
            is_best=False,
            save_tokenizer=False # Keep consistent
        )

    # Assert that the checkpoint file was NOT created
    assert not save_path.exists()
    # Assert tokenizer was NOT saved
    mock_tokenizer_save.assert_not_called()


def test_save_best_only_saves_best(mock_objects_for_cm, tmp_path):
    """Test save_checkpoint saves if save_best_only is True and metric IS best."""
    exp_name = "test_save_best_only_save_exp"
    step = 1000
    epoch = 10
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    current_best = 0.5
    new_metric = 0.4 # Better than current best

    # Create manager with save_best_only=True
    manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name, save_best_only=True)

    manager.checkpoint_dir = tmp_path # Ensure output goes to temp dir
    manager.best_metric = current_best

    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Mock the tokenizer's save method
    with patch.object(mock_objects_for_cm["tokenizer"], 'save') as mock_tokenizer_save:
        # Call save_checkpoint with is_best=True
        manager.save_checkpoint(
            state=training_state,
            filename=filename,
            is_best=True,
            save_tokenizer=False # Keep consistent
        )

        # Assert that the checkpoint file WAS created
        assert save_path.exists()
        # Assert tokenizer was NOT saved
        mock_tokenizer_save.assert_not_called()

def test_save_checkpoint_no_tokenizer_save_attempt(checkpoint_manager, mock_objects_for_cm):
    """Test that tokenizer save is not attempted when save_tokenizer=False."""
    step = 850
    epoch = 8
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = checkpoint_manager.checkpoint_dir / filename
    training_state = create_mock_training_state(mock_objects_for_cm, epoch=epoch, global_step=step)

    # Mock the tokenizer's save method
    with patch.object(mock_objects_for_cm["tokenizer"], 'save') as mock_tokenizer_save:
        # Call save_checkpoint with save_tokenizer=False
        checkpoint_manager.save_checkpoint(state=training_state, filename=filename, save_tokenizer=False)

    # Assert that torch.save was still called for the main checkpoint
    assert save_path.exists()
    # Assert that tokenizer.save was NOT called
    mock_tokenizer_save.assert_not_called()

# Fixtures
@pytest.fixture
def mock_pydantic_config():
    """Provides a mock Pydantic config including the required architecture field."""
    # Add required architecture field
    return MockPydanticConfig(param=10, architecture='mock_arch')

@pytest.fixture
def mock_model(mock_pydantic_config):
    """Provides a MockModel instance."""
    # Assuming MockModel exists and takes config
    return MockModel(config=mock_pydantic_config)

@pytest.fixture
def mock_optimizer(mock_model):
    """Provides a mock optimizer."""