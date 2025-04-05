"""
Tests for CheckpointManager specifically focusing on the load_checkpoint method.
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
        # Define layers that match expected state dict keys
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return x
    
    # Mock load_state_dict to accept strict=False
    def load_state_dict(self, state_dict, strict=True):
         # Basic check for expected keys if needed for testing errors
         expected_keys = {"layer.weight", "layer.bias"}
         missing = expected_keys - state_dict.keys()
         unexpected = state_dict.keys() - expected_keys
         if strict and (missing or unexpected):
             raise RuntimeError("strict mode error")
         # Simulate loading - actual mock won't change internal state
         return MagicMock(missing_keys=list(missing), unexpected_keys=list(unexpected))

class MockPydanticConfig(BaseModelConfig):
    param: int = 10

@pytest.fixture
def mock_tokenizer_fixture():
    tokenizer = MagicMock(spec=BaseTokenizer)
    tokenizer.save = MagicMock()
    tokenizer.load = MagicMock() # Add mock load method
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
    # Use the actual MockModel now
    mock_model = MockModel(config=MockPydanticConfig(param=10))
    # Provide a state_dict matching MockModel layers
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
    # Add load_state_dict to callbacks mock
    mock_callbacks.load_state_dict = MagicMock()
    # Add on_load_checkpoint hook
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
    with patch('os.getcwd', return_value=str(tmp_path)):
        manager = CheckpointManager(**mock_objects_for_cm)
        # Ensure checkpoint_dir is a Path object, not a string
        manager.checkpoint_dir = tmp_path
        yield manager

# Helper function to create a dummy checkpoint file
def create_dummy_checkpoint(path: Path, data: dict):
    os.makedirs(path.parent, exist_ok=True)
    torch.save(data, path)

# --- Load Tests --- #

def test_load_checkpoint_basic(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test basic loading of a valid checkpoint."""
    step = 1100
    epoch = 11
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    model, optimizer, scheduler, scaler, config, callbacks, tokenizer = map(
        mock_objects_for_cm.get, 
        ["model", "optimizer", "scheduler", "scaler", "config", "callbacks", "tokenizer"]
    )
    # Create a dummy checkpoint with all expected components
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "callbacks_state": callbacks.state_dict(),
        "tokenizer_path": None, # No tokenizer saved in this test yet
        "best_val_metric": 0.1,
        "metrics": {"loss": 0.2}
    }
    create_dummy_checkpoint(save_path, checkpoint_data)

    # --- Patch the actual model instance's load_state_dict --- #
    with patch.object(checkpoint_manager.model, 'load_state_dict') as mock_model_load_state_dict:
        
        # Load the checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

        assert loaded_state is not None
        assert isinstance(loaded_state, TrainingState)
        assert loaded_state.epoch == epoch
        assert loaded_state.global_step == step
        assert loaded_state.best_val_metric == 0.1
        assert loaded_state.metrics["loss"] == 0.2
        assert loaded_state.config["param"] == 20

        # Verify that load_state_dict was called on components
        mock_optimizer_in_fixture = mock_objects_for_cm["optimizer"]
        mock_scheduler_in_fixture = mock_objects_for_cm["scheduler"]
        mock_scaler_in_fixture = mock_objects_for_cm["scaler"]
        mock_callbacks_in_fixture = mock_objects_for_cm["callbacks"]
        mock_tokenizer_in_fixture = mock_objects_for_cm["tokenizer"]
        
        # Assert against the patched model method and other fixture mocks
        mock_model_load_state_dict.assert_called_once()
        mock_optimizer_in_fixture.load_state_dict.assert_called_once_with(checkpoint_data["optimizer_state_dict"])
        mock_scheduler_in_fixture.load_state_dict.assert_called_once_with(checkpoint_data["scheduler_state_dict"])
        mock_scaler_in_fixture.load_state_dict.assert_called_once_with(checkpoint_data["scaler_state_dict"])
        mock_callbacks_in_fixture.load_state_dict.assert_called_once_with(checkpoint_data["callbacks_state"])
        mock_tokenizer_in_fixture.load.assert_not_called() # No tokenizer path saved

def test_load_checkpoint_latest(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test loading the latest checkpoint when path is None."""
    # Create multiple dummy checkpoints
    model_state = mock_objects_for_cm["model"].state_dict()
    create_dummy_checkpoint(tmp_path / "checkpoint_epoch_1_step_100.pt", {"global_step": 100, "epoch": 1, "model_state_dict": model_state})
    time.sleep(0.1) # Ensure different modification times if sorting relied on it (it shouldn't anymore)
    create_dummy_checkpoint(tmp_path / "checkpoint_epoch_2_step_200.pt", {"global_step": 200, "epoch": 2, "model_state_dict": model_state})
    time.sleep(0.1)
    latest_path = tmp_path / "checkpoint_epoch_2_step_300.pt" # This one is latest by step
    create_dummy_checkpoint(latest_path, {"global_step": 300, "epoch": 2, "model_state_dict": model_state})

    # --- Patch the actual model instance's load_state_dict --- #
    with patch.object(checkpoint_manager.model, 'load_state_dict') as mock_model_load_state_dict:
        # Load with path=None
        loaded_state = checkpoint_manager.load_checkpoint(path=None)

        assert loaded_state is not None
        assert loaded_state.global_step == 300
        assert loaded_state.epoch == 2
        # Verify model was loaded from the correct file using the patched method
        mock_model_load_state_dict.assert_called_once()

def test_load_checkpoint_not_found(checkpoint_manager):
    """Test loading a non-existent checkpoint file."""
    non_existent_path = str(checkpoint_manager.checkpoint_dir / "non_existent.pt")
    
    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(path=non_existent_path)
        
        # Assert logger.error was called (at least once)
        mock_logger.error.assert_called()
        # Optionally check *one* of the calls if needed
        # assert any(f"Checkpoint file not found during load attempt: {non_existent_path}" in call.args[0] for call in mock_logger.error.call_args_list)

def test_load_checkpoint_latest_no_checkpoints(checkpoint_manager):
    """Test loading latest when no checkpoints exist."""
    assert not list(Path(checkpoint_manager.checkpoint_dir).glob("*.pt")) # Ensure dir is empty
    
    # Patch logger specifically for this test
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        # Re-assign the logger instance inside the CheckpointManager to our mock
        checkpoint_manager.logger = mock_logger

        loaded_state = checkpoint_manager.load_checkpoint(path=None)
        assert loaded_state is None
        mock_logger.warning.assert_called_once_with("No checkpoints found to load.")

def test_load_checkpoint_missing_keys(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test loading a checkpoint missing essential keys."""
    save_path = tmp_path / "missing_key.pt"
    # Create checkpoint *without* model_state_dict
    create_dummy_checkpoint(save_path, {"global_step": 10, "epoch": 0})

    # Update regex to match the wrapped error format
    expected_error_msg_regex = re.escape(f"Failed to load checkpoint from {save_path}:") + ".*missing required key: 'model_state_dict'"
    with pytest.raises(CheckpointLoadError, match=expected_error_msg_regex):
        checkpoint_manager.load_checkpoint(str(save_path))

def test_load_checkpoint_loads_tokenizer(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test that the tokenizer is loaded if path exists."""
    step = 1100
    epoch = 11
    tokenizer_rel_path = f"tokenizer_step_{step}"
    tokenizer_abs_path = tmp_path / tokenizer_rel_path
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    model_state = mock_objects_for_cm["model"].state_dict()
    optimizer_state = mock_objects_for_cm["optimizer"].state_dict()
    
    # Create dummy checkpoint referencing the tokenizer path
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "tokenizer_path": tokenizer_rel_path # Relative path
    }
    create_dummy_checkpoint(save_path, checkpoint_data)
    # Create the dummy tokenizer directory
    os.makedirs(tokenizer_abs_path, exist_ok=True)
    (tokenizer_abs_path / "dummy_tokenizer_file.txt").touch() # Indicate presence

    # Get the mock tokenizer to check its load method
    mock_tokenizer = mock_objects_for_cm["tokenizer"]

    loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

    assert loaded_state is not None
    # Assert tokenizer.load was called with the ABSOLUTE path
    mock_tokenizer.load.assert_called_once_with(str(tokenizer_abs_path))

def test_load_checkpoint_skips_missing_tokenizer_dir(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test loading skips tokenizer if directory specified in checkpoint is missing."""
    step = 1200
    epoch = 12
    tokenizer_rel_path = f"tokenizer_step_{step}_MISSING"
    tokenizer_abs_path = tmp_path / tokenizer_rel_path
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    model_state = mock_objects_for_cm["model"].state_dict()
    optimizer_state = mock_objects_for_cm["optimizer"].state_dict()

    # Create dummy checkpoint referencing the tokenizer path
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "tokenizer_path": tokenizer_rel_path
    }
    create_dummy_checkpoint(save_path, checkpoint_data)
    # --- DO NOT create the tokenizer directory --- #
    assert not tokenizer_abs_path.exists()

    mock_tokenizer = mock_objects_for_cm["tokenizer"]

    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

        assert loaded_state is not None
        mock_tokenizer.load.assert_not_called() # Load should not be called
        # Check warning log for the correct message when dir is missing
        expected_warning = f"Tokenizer directory '{str(tokenizer_abs_path)}' specified in checkpoint not found. Skipping tokenizer load."
        mock_logger.warning.assert_any_call(expected_warning)

def test_load_checkpoint_handles_tokenizer_load_error(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test logging when tokenizer.load() raises an error."""
    step = 1300
    epoch = 13
    tokenizer_rel_path = f"tokenizer_step_{step}"
    tokenizer_abs_path = tmp_path / tokenizer_rel_path
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    model_state = mock_objects_for_cm["model"].state_dict()
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model_state,
        "tokenizer_path": tokenizer_rel_path
    }
    create_dummy_checkpoint(save_path, checkpoint_data)
    os.makedirs(tokenizer_abs_path, exist_ok=True)

    mock_tokenizer = mock_objects_for_cm["tokenizer"]
    # Make tokenizer.load raise an error
    load_error = ValueError("Corrupted tokenizer file")
    mock_tokenizer.load.side_effect = load_error

    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

        assert loaded_state is not None
        mock_tokenizer.load.assert_called_once_with(str(tokenizer_abs_path))
        # Manually check call_args_list for the specific warning
        expected_warning_start = f"Failed to load tokenizer state from {str(tokenizer_abs_path)}: {load_error}"
        found_warning = False
        for call_args in mock_logger.warning.call_args_list:
            if call_args[0][0].startswith(expected_warning_start):
                found_warning = True
                break
        assert found_warning, f"Expected warning starting with '{expected_warning_start}' not found in logger calls."

def test_load_checkpoint_optional_states(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test loading works correctly when optional states (optimizer, scheduler, scaler, callbacks, tokenizer) are missing."""
    step = 1400
    epoch = 14
    filename = f"checkpoint_minimal_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    model, _, _, _, config, _, _ = map(mock_objects_for_cm.get, ["model", "optimizer", "scheduler", "scaler", "config", "callbacks", "tokenizer"])
    
    # Create checkpoint with only model state and minimal info
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": model.state_dict(),
        "config": config
        # Missing optimizer, scheduler, scaler, callbacks, tokenizer_path, metrics, best_val_metric
    }
    create_dummy_checkpoint(save_path, checkpoint_data)

    # Get mocks to check calls
    mock_optimizer = mock_objects_for_cm["optimizer"]
    mock_scheduler = mock_objects_for_cm["scheduler"]
    mock_scaler = mock_objects_for_cm["scaler"]
    mock_callbacks = mock_objects_for_cm["callbacks"]
    mock_tokenizer = mock_objects_for_cm["tokenizer"]

    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        # Load the minimal checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

        assert loaded_state is not None
        assert loaded_state.epoch == epoch
        assert loaded_state.global_step == step
        assert loaded_state.config == config
        # Check that optional fields are None or default in loaded state
        assert loaded_state.optimizer_state_dict is None
        assert loaded_state.scheduler_state_dict is None
        assert loaded_state.scaler_state_dict is None
        assert loaded_state.callbacks_state is None
        assert loaded_state.tokenizer_path is None
        # Assert against the actual default values from TrainingState.from_dict
        assert loaded_state.best_val_metric == float('inf')
        assert loaded_state.metrics == {}

        # Verify load_state_dict was NOT called for missing components
        mock_optimizer.load_state_dict.assert_not_called()
        mock_scheduler.load_state_dict.assert_not_called()
        mock_scaler.load_state_dict.assert_not_called()
        mock_callbacks.load_state_dict.assert_not_called()
        mock_tokenizer.load.assert_not_called()
        
        # Manually check call_args_list for specific warnings
        expected_warnings = [
            "Checkpoint does not contain 'optimizer_state_dict'. Optimizer state not loaded.",
            "Checkpoint does not contain 'scheduler_state_dict'. Scheduler state not loaded.",
            "Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.",
            "Checkpoint does not contain 'callbacks_state'.",
            "Checkpoint does not contain 'tokenizer_path'. Tokenizer state not loaded."
        ]
        logged_warnings = [call.args[0] for call in mock_logger.warning.call_args_list]
        for expected in expected_warnings:
            # Adjust the expected message for callbacks based on the actual log output
            if expected == "Checkpoint does not contain 'callbacks_state'.":
                expected_actual = "Callbacks state not found in checkpoint."
                assert any(expected_actual in logged for logged in logged_warnings), f"Expected warning '{expected_actual}' not found in {logged_warnings}"
            else:
                assert any(expected in logged for logged in logged_warnings), f"Expected warning '{expected}' not found in {logged_warnings}"

def test_load_checkpoint_callback_called(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test that the on_load_checkpoint callback is called with the loaded state."""
    step = 1500
    epoch = 15
    filename = f"checkpoint_callback_test_{epoch}_{step}.pt"
    save_path = tmp_path / filename
    model_state = mock_objects_for_cm["model"].state_dict()
    mock_callbacks = mock_objects_for_cm["callbacks"]

    checkpoint_data = {"epoch": epoch, "global_step": step, "model_state_dict": model_state}
    create_dummy_checkpoint(save_path, checkpoint_data)

    # Load checkpoint
    loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

    assert loaded_state is not None
    # Verify the callback method was called
    mock_callbacks.on_load_checkpoint.assert_called_once()
    # Verify it was called with the correct TrainingState object
    call_args, call_kwargs = mock_callbacks.on_load_checkpoint.call_args
    assert call_kwargs.get('trainer_state') == loaded_state
    assert isinstance(call_kwargs.get('trainer_state'), TrainingState)
    assert call_kwargs.get('trainer_state').global_step == step

def test_load_checkpoint_error_handling(checkpoint_manager, tmp_path):
    """Test error handling for various load failures (other than file not found/missing key)."""
    step = 1600
    epoch = 16
    filename = f"checkpoint_error_{epoch}_{step}.pt"
    save_path = tmp_path / filename
    model_state = {"invalid_key": torch.tensor(1.0)} # State that will cause load error
    checkpoint_data = {"epoch": epoch, "global_step": step, "model_state_dict": model_state}
    create_dummy_checkpoint(save_path, checkpoint_data)

    # Patch logger locally for this test
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger

        # Mock model.load_state_dict to raise a RuntimeError
        load_error = RuntimeError("Size mismatch")
        with patch.object(checkpoint_manager.model, 'load_state_dict', side_effect=load_error) as mock_load_state:
            with pytest.raises(CheckpointLoadError, match="Size mismatch"):
                checkpoint_manager.load_checkpoint(str(save_path))
            
            mock_load_state.assert_called_once() # Verify model.load_state_dict was called
            # Assert error log on the local mock logger
            mock_logger.error.assert_called_once()
            assert f"Failed to load checkpoint from {save_path}: {load_error}" in mock_logger.error.call_args[0][0]

    # Test generic exception during torch.load (patch logger again)
    with patch("logging.getLogger") as mock_get_logger_2:
        mock_logger_2 = MagicMock()
        mock_get_logger_2.return_value = mock_logger_2
        checkpoint_manager.logger = mock_logger_2

        import pickle # Need pickle for the exception type
        with patch('torch.load', side_effect=pickle.UnpicklingError("Invalid pickle data")) as mock_torch_load:
            with pytest.raises(CheckpointLoadError, match="Invalid pickle data"):
                checkpoint_manager.load_checkpoint(str(save_path))
            # Assert error log on the local mock logger
            mock_logger_2.error.assert_called_once()
            assert f"Failed to load checkpoint from {save_path}: Invalid pickle data" in mock_logger_2.error.call_args[0][0] 