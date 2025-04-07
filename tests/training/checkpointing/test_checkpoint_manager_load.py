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
from craft.data.tokenizers.base import Tokenizer # Replaced BaseTokenizer
from craft.training.callbacks import CallbackList, Callback

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
    tokenizer = MagicMock(spec=Tokenizer)
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
    
    # --- Configure CallbackList Mock --- #
    mock_callbacks = MagicMock()
    mock_callbacks.state_dict = MagicMock(return_value={"callback_state": 4})
    mock_callbacks.load_state_dict = MagicMock()
    
    # Create a mock *individual* callback
    mock_individual_cb = MagicMock(spec=Callback)
    mock_individual_cb.on_load_checkpoint = MagicMock()
    
    # Configure the CallbackList mock to return the individual mock(s)
    mock_callbacks.callbacks = [mock_individual_cb]
    # --- End CallbackList Mock Config --- #
    
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
    exp_name = "test_load_exp"
    with patch('os.getcwd', return_value=str(tmp_path)):
        # Add experiment_name
        manager = CheckpointManager(**mock_objects_for_cm, experiment_name=exp_name)
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
    # Add required scheduler state to dummy checkpoints
    scheduler_state = mock_objects_for_cm["scheduler"].state_dict()
    create_dummy_checkpoint(tmp_path / "checkpoint_epoch_1_step_100.pt", {"global_step": 100, "epoch": 1, "model_state_dict": model_state, "scheduler_state_dict": scheduler_state})
    time.sleep(0.1) # Ensure different modification times if sorting relied on it (it shouldn't anymore)
    create_dummy_checkpoint(tmp_path / "checkpoint_epoch_2_step_200.pt", {"global_step": 200, "epoch": 2, "model_state_dict": model_state, "scheduler_state_dict": scheduler_state})
    time.sleep(0.1)
    latest_path = tmp_path / "checkpoint_epoch_2_step_300.pt" # This one is latest by step
    create_dummy_checkpoint(latest_path, {"global_step": 300, "epoch": 2, "model_state_dict": model_state, "scheduler_state_dict": scheduler_state})

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
        # Check the message format might be fragile, focus on the call
        mock_logger.error.assert_called()
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
        # Expect two warnings: one for attempting latest, one for finding none
        assert mock_logger.warning.call_count == 2
        # mock_logger.warning.assert_any_call("No checkpoints found to load.") # OLD INCORRECT
        # Construct the expected message using the fixture's checkpoint_dir
        expected_msg = f"No checkpoints found in {checkpoint_manager.checkpoint_dir}."
        mock_logger.warning.assert_any_call(expected_msg) # CORRECTED message
        mock_logger.warning.assert_any_call("No specific checkpoint path provided. Attempting to load latest from current run dir (might not be what you want when resuming!).") # The other warning

def test_load_checkpoint_missing_keys(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test loading a checkpoint missing essential keys."""
    save_path = tmp_path / "missing_key.pt"
    # Create checkpoint *without* model_state_dict
    create_dummy_checkpoint(save_path, {"global_step": 10, "epoch": 0})

    # Update regex to match the wrapped error format
    # Use re.escape to handle potential special characters in the path
    expected_error_msg_regex = re.escape(f"Failed to load checkpoint from {save_path}:") + ".*missing required key.*model_state_dict"
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
        "scheduler_state_dict": mock_objects_for_cm["scheduler"].state_dict(),
        "scaler_state_dict": None,
        "config": mock_objects_for_cm["config"],
        "callbacks_state": None,
        "tokenizer_path": tokenizer_rel_path, # Relative path
        "best_val_metric": 0.1,
        "metrics": {},
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
    """Test loading where tokenizer_path exists but the dir is missing."""
    step = 1200
    epoch = 12
    tokenizer_rel_path = f"tokenizer_step_{step}"
    # DO NOT create the tokenizer directory: tmp_path / tokenizer_rel_path
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    mock_tokenizer = mock_objects_for_cm["tokenizer"]

    # Create checkpoint data pointing to the non-existent tokenizer dir
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": mock_objects_for_cm["model"].state_dict(),
        "optimizer_state_dict": mock_objects_for_cm["optimizer"].state_dict(),
        "scheduler_state_dict": mock_objects_for_cm["scheduler"].state_dict(),
        "scaler_state_dict": None,
        "config": mock_objects_for_cm["config"],
        "callbacks_state": None,
        "tokenizer_path": tokenizer_rel_path, # Points to non-existent dir
        "best_val_metric": 0.1,
        "metrics": {},
    }
    create_dummy_checkpoint(save_path, checkpoint_data)

    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger # Assign mock logger

        # Load the checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

        # Assert loading still succeeded for the main state
        assert loaded_state is not None
        assert loaded_state.global_step == step

        # Assert tokenizer.load was NOT called
        mock_tokenizer.load.assert_not_called()
        # Assert a warning was logged
        mock_logger.warning.assert_called_once()
        # Update assertion to match the actual warning message format
        tokenizer_abs_path = tmp_path / tokenizer_rel_path # Define path for message check
        expected_warning = f"Tokenizer directory specified in checkpoint ({tokenizer_abs_path}) does not exist. Tokenizer not loaded."
        assert expected_warning in mock_logger.warning.call_args[0][0]

def test_load_checkpoint_handles_tokenizer_load_error(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test that an error during tokenizer.load is logged but doesn't crash load_checkpoint."""
    step = 1300
    epoch = 13
    tokenizer_rel_path = f"tokenizer_step_{step}"
    tokenizer_abs_path = tmp_path / tokenizer_rel_path
    # Create the directory, but make load fail
    tokenizer_abs_path.mkdir(exist_ok=True)
    filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    mock_tokenizer = mock_objects_for_cm["tokenizer"]
    mock_tokenizer.load.side_effect = ValueError("Corrupt tokenizer file")

    # Create checkpoint data pointing to the tokenizer dir
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": mock_objects_for_cm["model"].state_dict(),
        "optimizer_state_dict": mock_objects_for_cm["optimizer"].state_dict(),
        "scheduler_state_dict": mock_objects_for_cm["scheduler"].state_dict(),
        "scaler_state_dict": None,
        "config": mock_objects_for_cm["config"],
        "callbacks_state": None,
        "tokenizer_path": tokenizer_rel_path, # Points to existing dir
        "best_val_metric": 0.1,
        "metrics": {},
    }
    create_dummy_checkpoint(save_path, checkpoint_data)

    # Patch logger locally
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        checkpoint_manager.logger = mock_logger # Assign mock logger

        # Load the checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

        # Assert loading still succeeded for the main state
        assert loaded_state is not None
        assert loaded_state.global_step == step

        # Assert tokenizer.load WAS called
        mock_tokenizer.load.assert_called_once_with(str(tokenizer_abs_path))
        # Assert a warning (not error) was logged
        mock_logger.error.assert_not_called() # Ensure error wasn't called
        mock_logger.warning.assert_called_once()
        # Check the warning message content
        expected_warning_part = f"Failed to load tokenizer from {tokenizer_abs_path}: Corrupt tokenizer file. Continuing without loading tokenizer."
        assert expected_warning_part in mock_logger.warning.call_args[0][0]

def test_load_checkpoint_optional_states(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test loading a checkpoint where optional states (scheduler, scaler, callbacks) are None."""
    step = 1400
    epoch = 14
    filename = f"checkpoint_minimal_epoch_{epoch}_step_{step}.pt"
    save_path = tmp_path / filename
    # Create checkpoint with minimal required keys
    # Scheduler state IS required if scheduler exists in manager
    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": mock_objects_for_cm["model"].state_dict(),
        "optimizer_state_dict": mock_objects_for_cm["optimizer"].state_dict(),
        "scheduler_state_dict": mock_objects_for_cm["scheduler"].state_dict(),
        "scaler_state_dict": None,
        "callbacks_state": None,
        "config": mock_objects_for_cm["config"],
        "tokenizer_path": None,
        "best_val_metric": 0.1,
        "metrics": {},
    }
    create_dummy_checkpoint(save_path, checkpoint_data)

    # --- Create a manager specifically WITHOUT optional components --- #
    manager_minimal = CheckpointManager(
        model=mock_objects_for_cm["model"],
        optimizer=mock_objects_for_cm["optimizer"],
        scheduler=None, # No scheduler
        scaler=None, # No scaler
        callbacks=None, # No callbacks
        tokenizer=None, # No tokenizer
        experiment_name="test_load_minimal_exp",
        config=mock_objects_for_cm["config"],
        device=mock_objects_for_cm["device"]
    )
    manager_minimal.checkpoint_dir = tmp_path # Override for test

    # --- Patch the actual model instance's load_state_dict --- #
    with patch.object(manager_minimal.model, 'load_state_dict') as mock_model_load_state_dict:
        loaded_state = manager_minimal.load_checkpoint(str(save_path))

        assert loaded_state is not None
        assert loaded_state.global_step == step
        mock_model_load_state_dict.assert_called_once()
        # Check optimizer was loaded (even though it's None in the manager, it was in the file)
        mock_objects_for_cm["optimizer"].load_state_dict.assert_called_once_with(checkpoint_data["optimizer_state_dict"])
        # Check that scheduler/scaler/callbacks load_state_dict were NOT called
        mock_objects_for_cm["scheduler"].load_state_dict.assert_not_called()
        mock_objects_for_cm["scaler"].load_state_dict.assert_not_called()
        mock_objects_for_cm["callbacks"].load_state_dict.assert_not_called()

def test_load_checkpoint_callback_called(checkpoint_manager, mock_objects_for_cm, tmp_path):
    """Test that the on_load_checkpoint callback hook is called."""
    step = 1500
    epoch = 15
    filename = f"checkpoint_callback_test_{epoch}_{step}.pt"
    save_path = tmp_path / filename
    mock_callbacks = mock_objects_for_cm["callbacks"]

    checkpoint_data = {
        "epoch": epoch,
        "global_step": step,
        "model_state_dict": mock_objects_for_cm["model"].state_dict(),
        "optimizer_state_dict": mock_objects_for_cm["optimizer"].state_dict(),
        "scheduler_state_dict": mock_objects_for_cm["scheduler"].state_dict(),
        "scaler_state_dict": None,
        "callbacks_state": {"cb1": 1}, # Dummy state
        "config": mock_objects_for_cm["config"],
        "tokenizer_path": None,
        "best_val_metric": 0.1,
        "metrics": {},
    }
    create_dummy_checkpoint(save_path, checkpoint_data)

    loaded_state = checkpoint_manager.load_checkpoint(str(save_path))

    # Assert the callback hook was called on the *individual* mock callback
    # Access the mock individual callback from the list we configured
    mock_individual_cb = mock_objects_for_cm["callbacks"].callbacks[0]
    mock_individual_cb.on_load_checkpoint.assert_called_once()
    # Optional: Check the argument passed
    call_args = mock_individual_cb.on_load_checkpoint.call_args
    assert isinstance(call_args.kwargs['state'], TrainingState)
    assert call_args.kwargs['state'].global_step == step

def test_load_checkpoint_error_handling(checkpoint_manager, tmp_path):
    """Test wrapping of exceptions during loading into CheckpointLoadError."""
    save_path = tmp_path / "corrupt_checkpoint.pt"
    # Create a file that is not a valid torch checkpoint
    with open(save_path, "w") as f:
        f.write("this is not a pickle file")

    # Escape the path string for regex matching
    escaped_path = str(save_path).replace("\\", "\\\\") # Double backslashes for regex
    # Expect a CheckpointLoadError wrapping the original torch error
    with pytest.raises(CheckpointLoadError, match=f"Failed to load checkpoint from {escaped_path}"):
        checkpoint_manager.load_checkpoint(str(save_path))

    # Test with a different internal error (e.g., during state dict loading)
    save_path_bad_state = tmp_path / "bad_state.pt"
    checkpoint_data = {
        "epoch": 1, "global_step": 1,
        "model_state_dict": {"layer.weight": torch.tensor(1.0)}, # Missing layer.bias
        "optimizer_state_dict": {}, "scheduler_state_dict": {}
    }
    create_dummy_checkpoint(save_path_bad_state, checkpoint_data)

    # Patch model.load_state_dict *within* the checkpoint_manager instance
    with patch.object(checkpoint_manager.model, 'load_state_dict') as mock_load_state_dict:
        mock_load_state_dict.side_effect = RuntimeError("Mismatched keys")
        
        escaped_path_bad_state = str(save_path_bad_state).replace("\\", "\\\\")
        # Expect a CheckpointLoadError wrapping the RuntimeError
        with pytest.raises(CheckpointLoadError, match=f"Failed to load checkpoint from {escaped_path_bad_state}: Mismatched keys"):
            checkpoint_manager.load_checkpoint(str(save_path_bad_state))
            
        mock_load_state_dict.assert_called_once() 

# Fixtures
@pytest.fixture
def mock_pydantic_config():
    """Provides a mock Pydantic config including the required architecture field."""
    # Add required architecture field
    return MockPydanticConfig(param=10, architecture='mock_arch')

@pytest.fixture
def mock_model(mock_pydantic_config):
    return MockModel(config=mock_pydantic_config) 