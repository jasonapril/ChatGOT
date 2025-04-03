import pytest
import torch
import os
from unittest.mock import MagicMock, patch
import sys
import shutil
import logging

# Module under test
from craft.training.checkpointing import CheckpointManager

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

    def test_init(self, mock_model, mock_optimizer, mock_scheduler, mock_scaler):
        """Test CheckpointManager initialization."""
        config = {'lr': 0.001}
        checkpoint_dir = "test_ckpts"
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=mock_scaler,
            config=config,
            checkpoint_dir=checkpoint_dir,
            device=mock_model.device
        )
        
        assert manager.model is mock_model
        assert manager.optimizer is mock_optimizer
        assert manager.scheduler is mock_scheduler
        assert manager.scaler is mock_scaler
        assert manager.config == config
        assert manager.checkpoint_dir == checkpoint_dir
        assert manager.logger is not None # Check logger was created 

    @patch("torch.save") # Mock torch.save to avoid actual file IO during check
    @patch("os.makedirs") # Mock makedirs
    def test_save_checkpoint_basic(self, mock_makedirs, mock_torch_save, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test basic saving of a checkpoint."""
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        
        epoch = 5
        step = 1000
        best_metric = 0.5
        metrics = {'loss': [1.0, 0.8]}
        save_path = tmp_path / "epoch_5.pt"

        manager.save_checkpoint(
            path=str(save_path),
            current_epoch=epoch,
            global_step=step,
            best_val_metric=best_metric,
            metrics=metrics,
            is_best=False
        )

        # Check that makedirs was called correctly (using os.path.dirname)
        mock_makedirs.assert_called_once_with(os.path.dirname(save_path), exist_ok=True)
        
        # Check that torch.save was called once
        mock_torch_save.assert_called_once()
        
        # Get the arguments passed to torch.save
        saved_checkpoint = mock_torch_save.call_args[0][0]
        saved_path = mock_torch_save.call_args[0][1]
        
        # Verify path
        assert saved_path == str(save_path)
        
        # Verify content
        assert saved_checkpoint['epoch'] == epoch
        assert saved_checkpoint['global_step'] == step
        assert saved_checkpoint['best_val_metric'] == best_metric
        assert saved_checkpoint['metrics'] == metrics
        # Check state dicts were retrieved and put in checkpoint
        assert saved_checkpoint['model_state_dict'] == mock_model.state_dict()
        assert saved_checkpoint['optimizer_state_dict'] == mock_optimizer.state_dict()
        # Check optional components are NOT present
        assert 'scheduler_state_dict' not in saved_checkpoint
        assert 'scaler_state_dict' not in saved_checkpoint
        # Check config is present (default None)
        assert saved_checkpoint['config'] is None
        
        # Check logger info message (use any_call due to potential dir creation logs)
        mock_logger_fixture.info.assert_any_call(f"Checkpoint saved successfully to {save_path}") 

    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_with_optional(self, mock_makedirs, mock_torch_save, mock_model, mock_optimizer, mock_scheduler, mock_scaler, tmp_path, mock_logger_fixture):
        """Test saving a checkpoint with scheduler and scaler."""
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=mock_scaler,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        
        epoch = 10
        step = 2000
        best_metric = 0.4
        metrics = {'accuracy': [0.8, 0.85]}
        save_path = tmp_path / "epoch_10.pt"

        manager.save_checkpoint(
            path=str(save_path),
            current_epoch=epoch,
            global_step=step,
            best_val_metric=best_metric,
            metrics=metrics,
            is_best=False
        )

        mock_makedirs.assert_called_once_with(os.path.dirname(save_path), exist_ok=True)
        mock_torch_save.assert_called_once()
        
        saved_checkpoint = mock_torch_save.call_args[0][0]
        
        # Verify optional components ARE present
        assert 'scheduler_state_dict' in saved_checkpoint
        assert saved_checkpoint['scheduler_state_dict'] == mock_scheduler.state_dict()
        assert 'scaler_state_dict' in saved_checkpoint
        assert saved_checkpoint['scaler_state_dict'] == mock_scaler.state_dict()
        
        # Verify other components are still present
        assert saved_checkpoint['epoch'] == epoch
        assert saved_checkpoint['model_state_dict'] == mock_model.state_dict()

        # Check logger info message (use any_call due to potential dir creation logs)
        mock_logger_fixture.info.assert_any_call(f"Checkpoint saved successfully to {save_path}")

    @patch("shutil.copyfile")
    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_is_best(self, mock_makedirs, mock_torch_save, mock_copyfile, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test saving a checkpoint with is_best=True."""
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        
        epoch = 15
        step = 3000
        best_metric = 0.3
        metrics = {'f1': [0.7]}
        save_path = tmp_path / f"epoch_{epoch}.pt"
        best_path = tmp_path / "best_model.pt"

        manager.save_checkpoint(
            path=str(save_path),
            current_epoch=epoch,
            global_step=step,
            best_val_metric=best_metric,
            metrics=metrics,
            is_best=True
        )

        # Check torch.save was called
        mock_torch_save.assert_called_once()
        # Check shutil.copyfile was called with correct paths
        mock_copyfile.assert_called_once_with(str(save_path), str(best_path))
        
        # Check logger messages
        mock_logger_fixture.info.assert_any_call(f"Checkpoint saved successfully to {save_path}")
        mock_logger_fixture.info.assert_any_call(f"Updated best model link to {best_path}")

    @patch("shutil.copyfile")
    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_is_best_same_path(self, mock_makedirs, mock_torch_save, mock_copyfile, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test saving directly as best_model.pt when path matches."""
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        
        save_path = tmp_path / "best_model.pt" # Save directly as best

        manager.save_checkpoint(
            path=str(save_path),
            current_epoch=1, global_step=100, best_val_metric=0.1, metrics={},
            is_best=True
        )

        mock_torch_save.assert_called_once()
        # Copy should NOT be called if paths are the same
        mock_copyfile.assert_not_called()
        mock_logger_fixture.info.assert_any_call(f"Checkpoint saved successfully to {save_path}")
        mock_logger_fixture.info.assert_any_call(f"Saved best model directly as {save_path}")

    # --- Tests for Config Serialization --- 

    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_with_dict_config(self, mock_makedirs, mock_torch_save, mock_model, mock_optimizer, tmp_path):
        """Test saving with a simple dictionary config."""
        config = {'lr': 0.01, 'model': {'name': 'test_model'}}
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            config=config,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        save_path = tmp_path / "dict_config.pt"
        manager.save_checkpoint(path=str(save_path), current_epoch=1, global_step=10, best_val_metric=0.9, metrics={})

        mock_torch_save.assert_called_once()
        saved_checkpoint = mock_torch_save.call_args[0][0]
        # Verify the dict config was saved directly
        assert saved_checkpoint['config'] == config

    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_with_omegaconf_success(self, mock_makedirs, mock_torch_save, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test saving with successful OmegaConf serialization."""
        # Mock OmegaConf and DictConfig
        mock_omegaconf = MagicMock()
        original_config = {'lr': 0.01, 'model': {'name': 'test_model'}} # Resolved dict
        mock_omegaconf.to_container.return_value = original_config
        
        # Create a real dummy class to represent DictConfig for isinstance
        class DummyDictConfig:
            pass
        # Create an instance of the dummy class to pass as config
        dummy_config_instance = DummyDictConfig()

        # Patch OmegaConf and the *name* DictConfig in the target module
        with patch('craft.training.checkpointing.OmegaConf', mock_omegaconf), \
             patch('craft.training.checkpointing.DictConfig', DummyDictConfig, create=True): 
            
            manager = CheckpointManager(
                model=mock_model,
                optimizer=mock_optimizer,
                config=dummy_config_instance,
                checkpoint_dir=str(tmp_path),
                device=mock_model.device
            )
            save_path = tmp_path / "omegaconf_success.pt"
            manager.save_checkpoint(path=str(save_path), current_epoch=1, global_step=10, best_val_metric=0.9, metrics={})

            mock_torch_save.assert_called_once()
            saved_checkpoint = mock_torch_save.call_args[0][0]
            mock_omegaconf.to_container.assert_called_once_with(dummy_config_instance, resolve=True)
            assert saved_checkpoint['config'] == original_config

    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_with_omegaconf_failure(self, mock_makedirs, mock_torch_save, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test saving with failed OmegaConf serialization (logs warning, saves None)."""
        mock_omegaconf = MagicMock()
        error_message = "Serialization failed!"
        mock_omegaconf.to_container.side_effect = Exception(error_message)
        
        # Use the same dummy class approach
        class DummyDictConfigFail:
             pass
        dummy_config_instance = DummyDictConfigFail()

        with patch('craft.training.checkpointing.OmegaConf', mock_omegaconf), \
             patch('craft.training.checkpointing.DictConfig', DummyDictConfigFail, create=True): 
            
            manager = CheckpointManager(
                model=mock_model,
                optimizer=mock_optimizer,
                config=dummy_config_instance, 
                checkpoint_dir=str(tmp_path),
                device=mock_model.device
            )
            save_path = tmp_path / "omegaconf_fail.pt"
            manager.save_checkpoint(path=str(save_path), current_epoch=1, global_step=10, best_val_metric=0.9, metrics={})

            mock_torch_save.assert_called_once()
            saved_checkpoint = mock_torch_save.call_args[0][0]
            mock_omegaconf.to_container.assert_called_once_with(dummy_config_instance, resolve=True)
            mock_logger_fixture.warning.assert_called_once_with(f"Could not serialize OmegaConf config for checkpoint: {error_message}")
            assert saved_checkpoint['config'] is None

    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_checkpoint_torch_save_fails(self, mock_makedirs, mock_torch_save, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test error handling when torch.save raises an exception."""
        error_message = "Disk full!"
        mock_torch_save.side_effect = OSError(error_message) # Simulate save failure
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        save_path = tmp_path / "fail.pt"
        
        # Expect no exception to be raised from save_checkpoint itself
        manager.save_checkpoint(path=str(save_path), current_epoch=1, global_step=10, best_val_metric=0.9, metrics={})

        # Check save was attempted
        mock_torch_save.assert_called_once()
        # Check error was logged
        mock_logger_fixture.error.assert_called_once_with(
            f"Failed to save checkpoint to {save_path}: {error_message}", exc_info=True
        )

    # --- Tests for load_checkpoint --- 

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_basic(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test basic loading of model and optimizer state."""
        mock_exists.return_value = True # Assume file exists
        
        # Prepare mock checkpoint data
        model_state = {'param1': torch.tensor(10.0)}
        optimizer_state = {'state': {'step': 100}}
        mock_checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'epoch': 5, # Include other data that might be present
            'config': {'lr': 0.1}
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "checkpoint_to_load.pt"
        
        loaded_config = manager.load_checkpoint(path=str(load_path))
        
        # Check file existence was checked
        mock_exists.assert_any_call(str(load_path))
        # Check torch.load was called with correct path and map_location
        mock_torch_load.assert_called_once_with(
            str(load_path),
            map_location=mock_model.device,
            weights_only=False
        )
        # Check model and optimizer load methods were called with correct state dicts
        mock_model.load_state_dict.assert_called_once_with(model_state)
        mock_optimizer.load_state_dict.assert_called_once_with(optimizer_state)
        # Check logger message
        mock_logger_fixture.info.assert_any_call(f"Successfully loaded checkpoint from {load_path}")
        # Check that loading completed and returned something (structure seems inconsistent)
        assert loaded_config is not None 

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_with_optional(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, mock_scheduler, mock_scaler, tmp_path, mock_logger_fixture):
        """Test loading checkpoint with scheduler and scaler state."""
        mock_exists.return_value = True
        
        scheduler_state = {'last_epoch': 15}
        scaler_state = {'_scale': torch.tensor(2048.0)}
        mock_checkpoint_data = {
            'model_state_dict': {'p':1}, # Dummy required state
            'optimizer_state_dict': {'s':1},
            'scheduler_state_dict': scheduler_state,
            'scaler_state_dict': scaler_state,
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=mock_scaler,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "checkpoint_with_optional.pt"
        
        manager.load_checkpoint(path=str(load_path))
        
        mock_torch_load.assert_called_once_with(
            str(load_path),
            map_location=mock_model.device,
            weights_only=False
        )
        # Check components were loaded
        mock_scheduler.load_state_dict.assert_called_once_with(scheduler_state)
        mock_scaler.load_state_dict.assert_called_once_with(scaler_state)
        
    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_optional_missing_in_ckpt(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, mock_scheduler, mock_scaler, tmp_path, mock_logger_fixture):
        """Test loading when scheduler/scaler exist in manager but not in checkpoint."""
        mock_exists.return_value = True
        
        # Checkpoint *without* scheduler/scaler state
        mock_checkpoint_data = {
            'model_state_dict': {'p':1},
            'optimizer_state_dict': {'s':1},
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=mock_scaler,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "checkpoint_missing_optional.pt"
        
        manager.load_checkpoint(path=str(load_path))
        
        # Check scheduler and scaler load methods were NOT called
        mock_scheduler.load_state_dict.assert_not_called()
        mock_scaler.load_state_dict.assert_not_called()
        # Check warnings were logged
        mock_logger_fixture.warning.assert_any_call("Checkpoint does not contain 'scheduler_state_dict'. Scheduler state not loaded.")
        mock_logger_fixture.warning.assert_any_call("Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.")

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_with_module_prefix(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test loading model state dict with 'module.' prefix."""
        mock_exists.return_value = True
        
        # Model state with prefix
        prefixed_model_state = {
            'module.param1': torch.tensor(10.0),
            'module.param2': torch.tensor(20.0)
        }
        # Expected state after stripping prefix
        expected_model_state = {
            'param1': torch.tensor(10.0),
            'param2': torch.tensor(20.0)
        }
        mock_checkpoint_data = {
            'model_state_dict': prefixed_model_state,
            'optimizer_state_dict': {'s':1} # Dummy optimizer state
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "prefixed_checkpoint.pt"
        
        manager.load_checkpoint(path=str(load_path))
        
        mock_torch_load.assert_called_once_with(
            str(load_path),
            map_location=mock_model.device,
            weights_only=False
        )
        # Check model state was loaded correctly (prefix stripped)
        mock_model.load_state_dict.assert_called_once_with(expected_model_state)
        # Check logger info message about stripping
        mock_logger_fixture.info.assert_any_call(
            "Detected 'module.' prefix in checkpoint state_dict, attempting to load into unwrapped model."
        )

    # --- Tests for load_checkpoint Error Handling ---

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_file_not_found(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test loading when the checkpoint file does not exist."""
        mock_exists.return_value = False # File does not exist
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "nonexistent_checkpoint.pt"
        
        loaded_config = manager.load_checkpoint(path=str(load_path))
        
        # Check file existence was checked (use any_call due to dir checks in init)
        mock_exists.assert_any_call(str(load_path))
        # Check logger error message (use any_call due to potential dir checks in init)
        mock_logger_fixture.error.assert_any_call(f"Checkpoint file not found: {load_path}")
        # Ensure load wasn't attempted
        mock_torch_load.assert_not_called()
        # Check None was returned
        assert loaded_config is None

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_torch_load_fails(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test error handling when torch.load raises an exception."""
        mock_exists.return_value = True # File exists
        error_message = "Corrupted file!"
        mock_torch_load.side_effect = Exception(error_message) # Simulate load failure
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "corrupted_checkpoint.pt"
        
        loaded_config = manager.load_checkpoint(path=str(load_path))
        
        # Check load was attempted
        mock_torch_load.assert_called_once_with(
            str(load_path),
            map_location=mock_model.device,
            weights_only=False
        )
        # Check error was logged
        mock_logger_fixture.error.assert_called_once_with(
            f"Failed to load checkpoint from {load_path}: {error_message}", exc_info=True
        )
        # Check return value is None
        assert loaded_config is None

    @patch("torch.load")
    @patch("os.path.exists")
    def test_load_checkpoint_missing_keys(self, mock_exists, mock_torch_load, mock_model, mock_optimizer, tmp_path, mock_logger_fixture):
        """Test loading checkpoint with missing mandatory keys (model/optimizer state)."""
        mock_exists.return_value = True
        
        # Checkpoint missing model_state_dict
        mock_checkpoint_data_no_model = {
            'optimizer_state_dict': {'s':1}
        }
        mock_torch_load.return_value = mock_checkpoint_data_no_model
        
        manager = CheckpointManager(
            model=mock_model,
            optimizer=mock_optimizer,
            checkpoint_dir=str(tmp_path),
            device=mock_model.device
        )
        load_path = tmp_path / "missing_keys.pt"
        
        manager.load_checkpoint(path=str(load_path))
        
        # Check model load wasn't called, optimizer was
        mock_model.load_state_dict.assert_not_called()
        mock_optimizer.load_state_dict.assert_called_once()
        # Check warning for missing model state
        mock_logger_fixture.warning.assert_any_call("Checkpoint does not contain 'model_state_dict'.")

        # Reset mocks and try again missing optimizer state
        mock_model.reset_mock()
        mock_optimizer.reset_mock()
        mock_logger_fixture.reset_mock()
        mock_checkpoint_data_no_optimizer = {
            'model_state_dict': {'p':1}
        }
        mock_torch_load.return_value = mock_checkpoint_data_no_optimizer
        
        manager.load_checkpoint(path=str(load_path)) # Path doesn't matter as load is mocked
        mock_model.load_state_dict.assert_called_once()
        mock_optimizer.load_state_dict.assert_not_called()
        mock_logger_fixture.warning.assert_any_call("Checkpoint does not contain 'optimizer_state_dict'. Optimizer state not loaded.")