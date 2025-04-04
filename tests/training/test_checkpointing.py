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

# Module under test
from craft.training.checkpointing import CheckpointManager
from craft.models.base import Model, BaseModelConfig
from craft.data.tokenizers.base import BaseTokenizer
from craft.training.callbacks import CallbackList

# Simple Mock Model
class MockModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return x

# Simple Mock Config (Can be Pydantic or just a dict for mocking)
# Pydantic allows type checking, but dict is simpler if not validating
class MockPydanticConfig(BaseModelConfig):
    param: int = 10

@pytest.fixture
def mock_config_dict():
    """Provides a simple config dictionary."""
    return {'param': 20}

@pytest.fixture
def mock_tokenizer_fixture():
    """Fixture for a mock BaseTokenizer."""
    tokenizer = MagicMock(spec=BaseTokenizer)
    # Mock methods needed by CheckpointManager
    tokenizer.save = MagicMock()
    # Add other methods if CheckpointManager uses them
    return tokenizer

# Re-use logger fixture if available or define locally
@pytest.fixture
def mock_logger_fixture():
    """Patches logging.getLogger and returns the mock logger instance."""
    # Use CheckpointManager specific logger name if needed
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger 

@pytest.fixture
def mock_model():
    """Fixture for a mock torch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.state_dict.return_value = {'param1': torch.tensor(1.0), 'param2': torch.tensor(2.0)}
    # Add device attribute for map_location in load
    model.device = 'cpu' 
    # Allow load_state_dict to be called
    model.load_state_dict = MagicMock()
    return model

@pytest.fixture
def mock_optimizer():
    """Fixture for a mock torch optimizer."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.state_dict.return_value = {'state': {}, 'param_groups': [{}] }
    optimizer.load_state_dict = MagicMock()
    return optimizer

@pytest.fixture
def mock_scheduler():
    """Fixture for a mock torch scheduler."""
    scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
    scheduler.state_dict.return_value = {'last_epoch': 5}
    scheduler.load_state_dict = MagicMock()
    return scheduler

@pytest.fixture
def mock_scaler():
    """Fixture for a mock GradScaler."""
    # Check if torch.amp exists, otherwise mock a generic object
    try:
        scaler_spec = torch.amp.GradScaler
    except AttributeError:
        scaler_spec = object # Fallback if torch.amp isn't available
        
    scaler = MagicMock(spec=scaler_spec)
    scaler.state_dict.return_value = {'_scale': torch.tensor(1024.0)}
    scaler.load_state_dict = MagicMock()
    return scaler

def test_placeholder():
    """Placeholder test to ensure the file is collected."""
    assert True 

# --- Test CheckpointManager --- 

class TestCheckpointManager:

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        """Use tmp_path provided by pytest for automatic cleanup.
           Also patch os.getcwd to return tmp_path for the test duration.
        """
        self.tmp_path = tmp_path
        with patch('os.getcwd', return_value=str(self.tmp_path)):
             yield # Run the test within the patched context

    @pytest.fixture
    def mock_objects_for_cm(self, mock_tokenizer_fixture):
        """Provides mock model, optimizer, scheduler needed for CM init."""
        mock_config = {'param': 20} # Use dict for simplicity
        mock_model = MockModel(config=MockPydanticConfig(param=10))
        mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
        mock_optimizer.state_dict = MagicMock(return_value={"opt_state": 1})
        mock_optimizer.load_state_dict = MagicMock()
        mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
        mock_scheduler.state_dict = MagicMock(return_value={"sched_state": 2})
        mock_scheduler.load_state_dict = MagicMock()
        mock_scaler = MagicMock(spec=torch.amp.GradScaler)
        mock_scaler.state_dict = MagicMock(return_value={"scaler_state": 3})
        mock_scaler.load_state_dict = MagicMock()
        mock_callbacks = MagicMock(spec=CallbackList)
        mock_tokenizer = mock_tokenizer_fixture # Use the argument directly

        return mock_model, mock_optimizer, mock_scheduler, mock_scaler, mock_config, mock_callbacks, mock_tokenizer

    @pytest.fixture
    def checkpoint_manager(self, mock_objects_for_cm):
        """Provides an initialized CheckpointManager instance.
           Uses patched os.getcwd() for directory.
        """
        model, optimizer, scheduler, scaler, config, callbacks, tokenizer = mock_objects_for_cm
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            callbacks=callbacks,
            tokenizer=tokenizer,
            device='cpu' # Keep device explicit for tests
        )
        return manager

    def test_init(self, checkpoint_manager, tmp_path):
        """Test CheckpointManager initialization."""
        assert checkpoint_manager.checkpoint_dir == str(tmp_path)
        assert Path(checkpoint_manager.checkpoint_dir).exists()
        assert checkpoint_manager.model is not None
        assert checkpoint_manager.optimizer is not None
        assert checkpoint_manager.tokenizer is not None

    def test_save_checkpoint_basic(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test basic saving of model and optimizer state."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 100
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename

        # Construct the state dictionary expected by the new signature
        state = {
            'epoch': 1,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': config, # Pass the config dict
            'metrics': {"loss": 1.0}, # Example metrics
            'best_val_metric': 0.5
        }

        checkpoint_manager.save_checkpoint(
            state=state,
            filename=filename,
            metrics=state['metrics'] # Pass metrics from state
        )

        assert save_path.exists()
        tokenizer.save.assert_called_once_with(str(tmp_path / "tokenizer"))

        # Verify content of the saved state dict
        saved_data = torch.load(save_path, weights_only=False)
        assert saved_data['global_step'] == step
        assert saved_data['epoch'] == 1
        assert "model_state_dict" in saved_data
        assert "optimizer_state_dict" in saved_data
        assert saved_data['optimizer_state_dict']["opt_state"] == 1
        assert "scheduler_state_dict" in saved_data
        assert saved_data['scheduler_state_dict']["sched_state"] == 2
        assert "scaler_state_dict" in saved_data
        assert saved_data['scaler_state_dict']["scaler_state"] == 3
        assert "config" in saved_data
        assert isinstance(saved_data['config'], dict)
        assert saved_data['config'].get('param') == 20
        assert "metrics" in saved_data
        assert saved_data['metrics']["loss"] == 1.0

    def test_save_checkpoint_with_optional(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test saving with optional scheduler and config."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 200
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename

        state = {
            'epoch': 2,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': config,
            'metrics': {"loss": 0.8},
            'best_val_metric': 0.4
        }

        checkpoint_manager.save_checkpoint(
            state=state,
            filename=filename,
            metrics=state['metrics'],
            is_best=False
        )

        assert save_path.exists()
        tokenizer.save.assert_called_once_with(str(tmp_path / "tokenizer"))

        # Verify content of the saved state dict
        saved_data = torch.load(save_path, weights_only=False)
        assert saved_data['global_step'] == step
        assert saved_data['epoch'] == 2
        assert "model_state_dict" in saved_data
        assert "optimizer_state_dict" in saved_data
        assert saved_data['optimizer_state_dict']["opt_state"] == 1
        assert "scheduler_state_dict" in saved_data
        assert saved_data['scheduler_state_dict']["sched_state"] == 2
        assert "scaler_state_dict" in saved_data
        assert saved_data['scaler_state_dict']["scaler_state"] == 3
        assert "config" in saved_data
        assert isinstance(saved_data['config'], dict)
        assert saved_data['config'].get('param') == 20
        assert "metrics" in saved_data
        assert saved_data['metrics']["loss"] == 0.8

    def test_save_checkpoint_is_best(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test saving 'best' checkpoint creates a copy and tracks."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 300
        filename = f"step_{step}.pt"
        step_save_path = tmp_path / filename
        best_save_path = tmp_path / "best.pt"

        state = {
            'epoch': 3,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': config,
            'metrics': {"loss": 0.6},
            'best_val_metric': 0.3
        }

        with patch.object(checkpoint_manager, '_add_saved_checkpoint') as mock_add_saved, \
             patch.object(checkpoint_manager, '_add_best_checkpoint') as mock_add_best:

            checkpoint_manager.save_checkpoint(
                state=state,
                filename=filename,
                metrics=state['metrics'],
                is_best=True
            )

            assert step_save_path.exists()
            assert best_save_path.exists()
            tokenizer.save.assert_called_once_with(str(tmp_path / "tokenizer"))
            mock_add_saved.assert_called_once_with(str(step_save_path))
            mock_add_best.assert_called_once_with(str(best_save_path))

    def test_save_checkpoint_is_best_same_path(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test saving 'best' when filename is already 'best_model.pt'."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 400
        filename = "best_model.pt" # Use a specific name
        save_path = tmp_path / filename
        best_path = tmp_path / "best.pt"

        state = {
            'epoch': 4,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': config,
            'metrics': {"loss": 0.4},
            'best_val_metric': 0.2
        }

        checkpoint_manager.save_checkpoint(
            state=state,
            filename=filename,
            metrics=state['metrics'],
            is_best=True
        )

        assert save_path.exists() # The specific file should exist
        assert best_path.exists() # best.pt should also exist due to _add_best_checkpoint logic
        # Check for the specific file, best.pt, and the tokenizer dir
        assert len(list(tmp_path.iterdir())) == 3

    @patch('torch.save', side_effect=Exception("Disk full"))
    def test_save_checkpoint_torch_save_fails(self, mock_torch_save, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test graceful failure if torch.save raises an exception."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 800
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename

        with patch.object(checkpoint_manager.logger, 'error') as mock_log_error:
            checkpoint_manager.save_checkpoint(
                state={
                    'epoch': 8,
                    'global_step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config': config,
                    'metrics': {"loss": 0.2},
                    'best_val_metric': 0.1
                },
                filename=filename,
                metrics={"loss": 0.2}
            )
            mock_torch_save.assert_called_once()
            mock_log_error.assert_called_once()
            # Adjust assertion to match the logged message format
            assert f"Failed to save checkpoint {filename}" in mock_log_error.call_args[0][0]
            assert "Disk full" in mock_log_error.call_args[0][0]

        assert not save_path.exists()

    def test_load_checkpoint_basic(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test loading basic model and optimizer states."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 100
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename
        torch.save({
            "epoch": 1,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "metrics": {"loss": 1.0},
            "best_val_metric": 0.5
        }, save_path)

        loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

        assert loaded_state is not None
        assert loaded_state['global_step'] == step
        assert loaded_state['epoch'] == 1
        assert loaded_state['best_val_metric'] == 0.5
        assert loaded_state['metrics'] == {"loss": 1.0}
        assert loaded_state['config'] == config
        optimizer.load_state_dict.assert_called_once_with(optimizer.state_dict())
        scheduler.load_state_dict.assert_called_once_with(scheduler.state_dict())
        scaler.load_state_dict.assert_called_once_with(scaler.state_dict())

    def test_load_checkpoint_with_optional(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test loading with optional scheduler and config."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 200
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename
        torch.save({
            "epoch": 2,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "metrics": {"loss": 0.8},
            "best_val_metric": 0.4
        }, save_path)

        loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

        assert loaded_state is not None
        assert loaded_state['global_step'] == step
        assert loaded_state['epoch'] == 2
        assert loaded_state['config'] == config
        optimizer.load_state_dict.assert_called_once()
        scheduler.load_state_dict.assert_called_once_with(scheduler.state_dict())
        scaler.load_state_dict.assert_called_once_with(scaler.state_dict())

    def test_load_checkpoint_optional_missing_in_ckpt(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test loading when optional elements are requested but not in checkpoint."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 250
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename
        # Save checkpoint WITHOUT scheduler_state_dict
        torch.save({
            "epoch": 3,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "metrics": {"loss": 0.7},
            "best_val_metric": 0.3
        }, save_path)

        # The checkpoint_manager fixture has a scheduler and scaler
        with patch.object(checkpoint_manager.logger, 'warning') as mock_log_warning:
            loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

            assert loaded_state is not None
            assert loaded_state['global_step'] == step
            assert loaded_state['config'] == config # Config SHOULD be loaded
            # assert loaded_state['config'] is None # INCORRECT assertion

            optimizer.load_state_dict.assert_called_once()
            scaler.load_state_dict.assert_called_once() # Scaler WAS in checkpoint
            scheduler.load_state_dict.assert_not_called() # Scheduler was NOT in checkpoint

            # Check ONLY for the scheduler warning
            scheduler_warning_found = any(
                "scheduler_state_dict" in call.args[0] and "not loaded" in call.args[0]
                for call in mock_log_warning.call_args_list
            )
            scaler_warning_found = any(
                "scaler_state_dict" in call.args[0] and "not loaded" in call.args[0]
                for call in mock_log_warning.call_args_list
            )
            assert scheduler_warning_found
            assert not scaler_warning_found

    def test_load_checkpoint_with_module_prefix(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test loading state dict when keys have 'module.' prefix."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 300
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename
        
        original_state_dict = model.state_dict()
        prefixed_state_dict = {f"module.{k}": v for k, v in original_state_dict.items()}
        
        torch.save({
            "epoch": 3,
            "global_step": step,
            "model_state_dict": prefixed_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "metrics": {"loss": 0.5},
            "best_val_metric": 0.3
        }, save_path)

        mock_model_instance = checkpoint_manager.model 
        mock_model_instance.load_state_dict = MagicMock()

        loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

        assert loaded_state is not None
        assert loaded_state['global_step'] == step
        
        mock_model_instance.load_state_dict.assert_called_once()
        optimizer.load_state_dict.assert_called_once()
        scheduler.load_state_dict.assert_called_once()
        scaler.load_state_dict.assert_called_once()

    def test_load_checkpoint_file_not_found(self, checkpoint_manager, tmp_path):
        """Test load returns None for non-existent file."""
        non_existent_path = tmp_path / "non_existent.pt"
        
        with patch.object(checkpoint_manager.logger, 'error') as mock_log_error:
            result = checkpoint_manager.load_checkpoint(path=str(non_existent_path))
            assert result is None
            mock_log_error.assert_called_once_with(f"Checkpoint file not found: {non_existent_path}")

    @patch('torch.load', side_effect=Exception("Corrupt file"))
    def test_load_checkpoint_torch_load_fails(self, mock_torch_load, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test graceful failure if torch.load raises an exception."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 900
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename
        save_path.touch()

        with patch.object(checkpoint_manager.logger, 'error') as mock_log_error:
            result = checkpoint_manager.load_checkpoint(path=str(save_path))
            assert result is None
            mock_torch_load.assert_called_once()
            mock_log_error.assert_called_once()
            assert f"Failed to load checkpoint from {save_path}: Corrupt file" in mock_log_error.call_args[0][0]

    def test_load_checkpoint_missing_keys(self, checkpoint_manager, mock_objects_for_cm, tmp_path):
        """Test load handles missing essential keys gracefully (logs warnings)."""
        model, optimizer, scheduler, scaler, config, _, tokenizer = mock_objects_for_cm
        step = 1000
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename
        
        torch.save({
            "epoch": 10,
            "global_step": step,
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)

        model.load_state_dict = MagicMock()
        optimizer.load_state_dict = MagicMock()

        with patch.object(checkpoint_manager.logger, 'warning') as mock_log_warning:
            loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))
            
            assert loaded_state is not None
            assert loaded_state['global_step'] == step
            assert loaded_state['epoch'] == 10
            
            model.load_state_dict.assert_not_called()
            optimizer.load_state_dict.assert_called_once()
            mock_log_warning.assert_any_call("Checkpoint does not contain 'model_state_dict'.")

        torch.save({"model_state_dict": model.state_dict()}, save_path, _use_new_zipfile_serialization=False)

        model.load_state_dict = MagicMock()
        optimizer.load_state_dict = MagicMock()

        with patch.object(checkpoint_manager.logger, 'warning') as mock_log_warning:
             loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))
             assert loaded_state is not None
             assert loaded_state['global_step'] == 0
             assert loaded_state['epoch'] == 0
             model.load_state_dict.assert_called_once()
             optimizer.load_state_dict.assert_not_called()
             mock_log_warning.assert_any_call("Checkpoint does not contain 'optimizer_state_dict'. Optimizer state not loaded.")

    # --- Keep Last / Cleanup Tests ---
    def test_cleanup_old_checkpoints(self, checkpoint_manager, tmp_path):
        # Implementation of test_cleanup_old_checkpoints method
        pass