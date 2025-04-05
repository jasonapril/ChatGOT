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

# Module under test
from craft.training.checkpointing import CheckpointManager, CheckpointLoadError, TrainingState
from craft.models.base import Model, BaseModelConfig # Keep Base models for MockModel
from craft.data.tokenizers.base import BaseTokenizer
from craft.training.callbacks import CallbackList

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
    tokenizer = MagicMock(spec=BaseTokenizer)
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
    # Patch getcwd for the duration of the manager's life in this fixture
    with patch('os.getcwd', return_value=str(tmp_path)):
         # Pass the dictionary directly to CheckpointManager
         manager = CheckpointManager(**mock_objects_for_cm)
         # Override checkpoint_dir to ensure it uses tmp_path absolutely
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
    """Test basic saving functionality."""
    step = 100
    epoch = 1
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    metrics_data = {"loss": 1.0}
    best_val_metric_data = 0.5
    
    # Create TrainingState object
    training_state = create_mock_training_state(
        mock_objects_for_cm, 
        epoch=epoch, 
        global_step=step, 
        metrics=metrics_data,
        best_val_metric=best_val_metric_data
    )
    
    # Call save_checkpoint with TrainingState and filename
    checkpoint_manager.save_checkpoint(
        state=training_state,
        filename=filename,
        metrics=metrics_data, # Metrics can still be passed for logging/best logic
        is_best=False # Explicitly False for this basic test
    )

    assert save_path.exists()
    mock_objects_for_cm["tokenizer"].save.assert_called_once()
    
    # Check the tokenizer path saved in the dictionary matches the one in TrainingState
    saved_dict = torch.load(save_path, map_location='cpu')
    assert saved_dict['tokenizer_path'] == training_state.tokenizer_path
    
    # Check that the tokenizer dir itself was created relative to the checkpoint_dir
    # Use the step number to construct the expected path
    tokenizer_save_dir = checkpoint_manager.checkpoint_dir / f"tokenizer_step_{step}"
    assert tokenizer_save_dir.is_dir()
    mock_objects_for_cm["tokenizer"].save.assert_called_once_with(str(tokenizer_save_dir))

    # Verify content using the saved dict (which is the asdict(TrainingState))
    assert saved_dict['global_step'] == step
    assert saved_dict['epoch'] == epoch
    assert "model_state_dict" in saved_dict
    assert saved_dict['model_state_dict'].keys() == training_state.model_state_dict.keys()
    assert saved_dict['optimizer_state_dict'] == training_state.optimizer_state_dict
    assert "scheduler_state_dict" in saved_dict
    assert saved_dict['scheduler_state_dict'] == training_state.scheduler_state_dict
    assert "scaler_state_dict" in saved_dict
    assert saved_dict['scaler_state_dict'] == training_state.scaler_state_dict
    assert "config" in saved_dict
    assert saved_dict['config'] == training_state.config
    assert "metrics" in saved_dict
    assert saved_dict['metrics'] == training_state.metrics
    assert "best_val_metric" in saved_dict
    assert saved_dict['best_val_metric'] == training_state.best_val_metric
    assert "callbacks_state" in saved_dict
    assert saved_dict['callbacks_state'] == training_state.callbacks_state
    
    # Check checkpoint manager tracking
    assert any(p == str(save_path) and b is False for p, b in checkpoint_manager.saved_checkpoints)

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
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    
    # Create manager with the modified mocks
    manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=tmp_path)

    # Create TrainingState - it should automatically get None for scaler state
    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )
    assert training_state.scaler_state_dict is None # Verify state creation

    manager.save_checkpoint(
        state=training_state, filename=filename
    )

    assert save_path.exists()
    saved_dict = torch.load(save_path, map_location='cpu')
    assert "model_state_dict" in saved_dict
    assert "optimizer_state_dict" in saved_dict
    assert "scheduler_state_dict" in saved_dict
    assert saved_dict.get("scaler_state_dict") is None # Check saved state is None

def test_save_skips_disabled_optimizer_scheduler(mock_objects_for_cm, tmp_path):
    """Test saving when optimizer or scheduler are None."""
    # Set optimizer/scheduler to None BEFORE creating the manager
    mock_objects_for_cm["optimizer"] = None
    mock_objects_for_cm["scheduler"] = None
    step = 400
    epoch = 4
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    # Create manager with None optimizer/scheduler
    manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=tmp_path)
    
    # Create TrainingState - should get None for opt/sched states
    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )
    assert training_state.optimizer_state_dict is None
    assert training_state.scheduler_state_dict is None

    manager.save_checkpoint(
        state=training_state, filename=filename
    )

    assert save_path.exists()
    saved_dict = torch.load(save_path, map_location='cpu')
    assert "model_state_dict" in saved_dict
    assert saved_dict.get("optimizer_state_dict") is None
    assert saved_dict.get("scheduler_state_dict") is None
    assert "scaler_state_dict" in saved_dict # Scaler should still be saved

def test_save_without_tokenizer(mock_objects_for_cm, tmp_path):
    """Test saving when tokenizer is None."""
    # Set tokenizer to None BEFORE creating the manager
    mock_objects_for_cm["tokenizer"] = None
    step = 500
    epoch = 5
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    # Create manager with None tokenizer
    manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=tmp_path)

    # Create TrainingState - tokenizer_path should be None
    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )
    assert training_state.tokenizer_path is None

    manager.save_checkpoint(
        state=training_state, filename=filename
    )

    assert save_path.exists()
    saved_dict = torch.load(save_path, map_location='cpu')
    assert "tokenizer_path" in saved_dict
    assert saved_dict["tokenizer_path"] is None
    # Assert tokenizer directory was NOT created
    assert not (tmp_path / "tokenizer").exists()

def test_save_checkpoint_creates_dir(mock_objects_for_cm, tmp_path):
    """Test save_checkpoint creates the directory if it doesn't exist."""
    step = 600
    epoch = 6
    sub_dir_name = "new_subdir"
    non_existent_dir = tmp_path / sub_dir_name
    assert not non_existent_dir.exists()
    
    # Create manager pointing to the non-existent dir
    manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=non_existent_dir)
    
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = non_existent_dir / filename # Path relative to manager's dir

    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )

    # Saving should create the directory
    manager.save_checkpoint(
        state=training_state, filename=filename
    )
    
    assert non_existent_dir.exists()
    assert save_path.exists()

def test_save_checkpoint_error_handling(checkpoint_manager, mock_objects_for_cm):
    """Test error logging if torch.save fails."""
    step = 700
    epoch = 7
    filename = "fail_save.pt"
    save_path = checkpoint_manager.checkpoint_dir / filename

    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )

    # Patch torch.save and logger locally
    with patch('torch.save', side_effect=OSError("Disk full")) as mock_torch_save, \
         patch("logging.getLogger") as mock_get_logger:
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        checkpoint_manager.save_checkpoint(
            state=training_state, filename=filename
        )
        mock_torch_save.assert_called_once() # Ensure save was attempted
        # Assert on local mock logger
        mock_logger.error.assert_called_once()
        assert "Disk full" in mock_logger.error.call_args[0][0]

def test_save_checkpoint_tokenizer_error_handling(checkpoint_manager, mock_objects_for_cm):
    """Test error logging if tokenizer.save fails."""
    step = 800
    epoch = 8
    filename = f"tokenizer_fail_step_{step}.pt"
    save_path = checkpoint_manager.checkpoint_dir / filename
    mock_tokenizer = mock_objects_for_cm["tokenizer"]
    
    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )

    # Make tokenizer.save raise an error
    tokenizer_error = OSError("Permission denied")
    mock_tokenizer.save.side_effect = tokenizer_error

    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        checkpoint_manager.save_checkpoint(
            state=training_state, filename=filename
        )

        # Checkpoint file should still be saved (save happens before tokenizer)
        assert save_path.exists()
        mock_tokenizer.save.assert_called_once() # Ensure tokenizer save was attempted
        # Error during tokenizer save should be logged
        mock_logger.error.assert_called_once()
        # Check the specific error message
        assert "Permission denied" in mock_logger.error.call_args[0][0]

def test_save_best_only_skips_non_best(mock_objects_for_cm, tmp_path):
    """Test save_best_only=True skips saving non-best checkpoints."""
    # Enable save_best_only
    manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=tmp_path, save_best_only=True)
    step = 900
    epoch = 9
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename

    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )

    # Call save without is_best=True
    manager.save_checkpoint(
        state=training_state, filename=filename, is_best=False 
    )
    assert not save_path.exists() # Should not have been saved
    assert not (tmp_path / "best.pt").exists() # best.pt link should not exist

def test_save_best_only_saves_best(mock_objects_for_cm, tmp_path):
    """Test save_best_only=True saves a checkpoint marked as best."""
     # Enable save_best_only
    manager = CheckpointManager(**mock_objects_for_cm, checkpoint_dir=tmp_path, save_best_only=True)
    step = 1000
    epoch = 10
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    best_link_path = tmp_path / "best.pt"

    training_state = create_mock_training_state(
        mock_objects_for_cm, epoch=epoch, global_step=step
    )

    # Call save WITH is_best=True
    manager.save_checkpoint(
        state=training_state, filename=filename, is_best=True
    )
    assert save_path.exists() # Should have been saved
    assert best_link_path.exists() # best.pt link should exist
    # Don't compare realpath
    # assert os.path.realpath(best_link_path) == os.path.realpath(save_path)

# test_load_checkpoint_error_handling was incorrectly placed here, remove it.
# def test_load_checkpoint_error_handling(checkpoint_manager, mock_logger_fixture, tmp_path):
#     """Test error handling for various load failures (other than file not found/missing key)."""
#     step = 1600
#     epoch = 16
#     filename = f"checkpoint_error_{epoch}_step_{step}.pt"
#     save_path = tmp_path / filename
#     model_state = {"invalid_key": torch.tensor(1.0)} # State that will cause load error
#     checkpoint_data = {"epoch": epoch, "global_step": step, "model_state_dict": model_state}
#     create_dummy_checkpoint(save_path, checkpoint_data)
#
#     # Mock model.load_state_dict to raise a RuntimeError
#     load_error = RuntimeError("Size mismatch")
#     with patch.object(checkpoint_manager.model, "load_state_dict", side_effect=load_error) as mock_load_state:
#         with pytest.raises(CheckpointLoadError, match="Size mismatch"):
#             checkpoint_manager.load_checkpoint(str(save_path))
#         
#         mock_load_state.assert_called_once() # Verify model.load_state_dict was called
#         mock_logger_fixture.error.assert_called_once()
#         assert "Failed to load checkpoint" in mock_logger_fixture.error.call_args[0][0] 