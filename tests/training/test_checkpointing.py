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

# Module under test
from craft.training.checkpointing import CheckpointManager
from craft.models.base import Model, BaseModelConfig

# Simple Mock Model
class MockModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return x

# Simple Mock Config
class MockConfig(BaseModelConfig):
    param: int = 10

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
        """Use tmp_path provided by pytest for automatic cleanup."""
        self.tmp_path = tmp_path 
        yield

    @pytest.fixture
    def mock_objects(self):
        """Provides mock model, optimizer, scheduler, and config."""
        mock_config = MockConfig(param=20)
        mock_model = MockModel(config=mock_config)
        mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
        mock_optimizer.state_dict = MagicMock(return_value={"opt_state": 1})
        mock_optimizer.load_state_dict = MagicMock()
        mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
        mock_scheduler.state_dict = MagicMock(return_value={"sched_state": 2})
        mock_scheduler.load_state_dict = MagicMock()
        return mock_model, mock_optimizer, mock_scheduler, mock_config

    @pytest.fixture
    def checkpoint_manager(self, tmp_path, mock_objects):
        """Provides an initialized CheckpointManager instance."""
        model, optimizer, scheduler, config = mock_objects
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            checkpoint_dir=str(tmp_path),
            device='cpu'
        )
        return manager

    def test_init(self, checkpoint_manager, tmp_path):
        """Test CheckpointManager initialization."""
        assert Path(checkpoint_manager.checkpoint_dir) == Path(tmp_path)
        assert Path(checkpoint_manager.checkpoint_dir).exists()
        assert checkpoint_manager.model is not None
        assert checkpoint_manager.optimizer is not None

    def test_save_checkpoint_basic(self, checkpoint_manager, mock_objects, tmp_path):
        """Test basic saving of model and optimizer."""
        model, optimizer, _, _ = mock_objects
        step = 100
        epoch = 1
        best_val = 0.5
        metrics = {"loss": [1.0]}
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename

        checkpoint_manager.save_checkpoint(
            path=str(save_path), 
            current_epoch=epoch, 
            global_step=step, 
            best_val_metric=best_val,
            metrics=metrics
        )

        assert save_path.exists()
        checkpoint = torch.load(save_path, weights_only=False)
        assert checkpoint['global_step'] == step
        assert checkpoint['epoch'] == epoch
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint['optimizer_state_dict']["opt_state"] == 1
        assert "scheduler_state_dict" in checkpoint
        assert "config" in checkpoint
        assert isinstance(checkpoint['config'], dict)
        assert checkpoint['config'].get('param') == 20

    def test_save_checkpoint_with_optional(self, checkpoint_manager, mock_objects, tmp_path):
        """Test saving with optional scheduler and config."""
        model, optimizer, scheduler, config = mock_objects
        step = 200
        epoch = 2
        best_val = 0.4
        metrics = {"loss": [0.8]}
        filename = f"step_{step}.pt"
        save_path = tmp_path / filename

        checkpoint_manager.save_checkpoint(
            path=str(save_path), 
            current_epoch=epoch, 
            global_step=step, 
            best_val_metric=best_val,
            metrics=metrics
        )

        assert save_path.exists()
        checkpoint = torch.load(save_path, weights_only=False)
        assert checkpoint['global_step'] == step
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert checkpoint['scheduler_state_dict']["sched_state"] == 2
        assert "config" in checkpoint
        assert isinstance(checkpoint['config'], dict)
        assert checkpoint['config'].get('param') == 20

    def test_save_checkpoint_is_best(self, checkpoint_manager, mock_objects, tmp_path):
        """Test saving 'best' checkpoint creates a symlink or copy."""
        model, optimizer, _, _ = mock_objects
        step = 300
        epoch = 3
        best_val = 0.3
        metrics = {"loss": [0.6]}
        filename = f"step_{step}.pt"
        step_save_path = tmp_path / filename
        best_save_path = tmp_path / "best_model.pt"

        checkpoint_manager.save_checkpoint(
            path=str(step_save_path), 
            current_epoch=epoch, 
            global_step=step, 
            best_val_metric=best_val,
            metrics=metrics,
            is_best=True
        )

        assert step_save_path.exists()
        assert best_save_path.exists()
        step_ckpt = torch.load(step_save_path, weights_only=False)
        best_ckpt = torch.load(best_save_path, weights_only=False)
        assert step_ckpt['global_step'] == best_ckpt['global_step']

    def test_save_checkpoint_is_best_same_path(self, checkpoint_manager, mock_objects, tmp_path):
        """Test saving 'best' when filename is already 'best_model.pt'."""
        model, optimizer, _, _ = mock_objects
        step = 400
        epoch = 4
        best_val = 0.2
        metrics = {"loss": [0.4]}
        filename = "best_model.pt"
        save_path = tmp_path / filename

        checkpoint_manager.save_checkpoint(
            path=str(save_path),
            current_epoch=epoch, 
            global_step=step, 
            best_val_metric=best_val,
            metrics=metrics,
            is_best=True
        )

        assert save_path.exists()
        assert len(list(tmp_path.iterdir())) == 1

    @patch('torch.save', side_effect=Exception("Disk full"))
    def test_save_checkpoint_torch_save_fails(self, mock_torch_save, checkpoint_manager, mock_objects, tmp_path):
        """Test graceful failure if torch.save raises an exception."""
        model, optimizer, _, _ = mock_objects
        step = 800
        epoch = 8
        best_val = 0.1
        metrics = {"loss": [0.2]}
        save_path = tmp_path / f"step_{step}.pt"

        with patch.object(checkpoint_manager.logger, 'error') as mock_log_error:
            checkpoint_manager.save_checkpoint(
                path=str(save_path), 
                current_epoch=epoch, 
                global_step=step, 
                best_val_metric=best_val,
                metrics=metrics
            )
            mock_torch_save.assert_called_once()
            mock_log_error.assert_called_once()
            assert f"Failed to save checkpoint to {save_path}" in mock_log_error.call_args[0][0]

        assert not save_path.exists()

    def test_load_checkpoint_basic(self, checkpoint_manager, mock_objects, tmp_path):
        """Test loading basic model and optimizer states."""
        model, optimizer, scheduler, config = mock_objects
        step = 100
        epoch = 1
        best_val = 0.5
        metrics = {"loss": [1.0]}
        tb_log_dir = "logs/dummy"
        save_path = tmp_path / f"step_{step}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,
            "best_val_metric": best_val,
            "metrics": metrics,
            "config": config.model_dump(),
            "tensorboard_log_dir": tb_log_dir
        }, save_path)

        loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

        assert loaded_state is not None
        assert loaded_state['global_step'] == step
        assert loaded_state['epoch'] == epoch
        assert loaded_state['best_val_metric'] == best_val
        assert loaded_state['metrics'] == metrics
        assert loaded_state['tensorboard_log_dir'] == tb_log_dir
        optimizer.load_state_dict.assert_called_once_with(optimizer.state_dict())
        scheduler.load_state_dict.assert_called_once_with(scheduler.state_dict())

    def test_load_checkpoint_with_optional(self, checkpoint_manager, mock_objects, tmp_path):
        """Test loading with optional scheduler and config."""
        model, optimizer, scheduler, config = mock_objects
        step = 200
        epoch = 2
        best_val = 0.4
        metrics = {"loss": [0.8]}
        save_path = tmp_path / f"step_{step}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.model_dump(),
            "best_val_metric": best_val,
            "metrics": metrics,
        }, save_path)

        loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

        assert loaded_state is not None
        assert loaded_state['global_step'] == step
        assert loaded_state['epoch'] == epoch
        assert loaded_state['config'] == config.model_dump()
        optimizer.load_state_dict.assert_called_once()
        scheduler.load_state_dict.assert_called_once_with(scheduler.state_dict())

    def test_load_checkpoint_optional_missing_in_ckpt(self, checkpoint_manager, mock_objects, tmp_path):
        """Test loading when optional elements are requested but not in checkpoint."""
        model, optimizer, scheduler, _ = mock_objects
        step = 250
        epoch = 3
        best_val = 0.3
        metrics = {"loss": [0.7]}
        save_path = tmp_path / f"step_{step}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_metric": best_val,
            "metrics": metrics,
        }, save_path)

        with patch.object(checkpoint_manager.logger, 'warning') as mock_log_warning:
            loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

            assert loaded_state is not None
            assert loaded_state['global_step'] == step
            optimizer.load_state_dict.assert_called_once()
            scheduler.load_state_dict.assert_not_called()
            mock_log_warning.assert_any_call(
                "Checkpoint does not contain 'scheduler_state_dict'. Scheduler state not loaded."
            )
            assert loaded_state['config'] is None
            scaler_warning_found = any(
                 "scaler_state_dict" in call.args[0] for call in mock_log_warning.call_args_list
            )
            assert not scaler_warning_found

    def test_load_checkpoint_with_module_prefix(self, checkpoint_manager, mock_objects, tmp_path):
        """Test loading state dict when keys have 'module.' prefix."""
        model, optimizer, scheduler, config = mock_objects
        step = 300
        epoch = 3
        best_val = 0.3
        metrics = {"loss": [0.5]}
        save_path = tmp_path / f"step_{step}.pt"
        
        original_state_dict = model.state_dict()
        prefixed_state_dict = {f"module.{k}": v for k, v in original_state_dict.items()}
        
        torch.save({
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": prefixed_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config.model_dump(),
            "best_val_metric": best_val,
            "metrics": metrics,
        }, save_path)

        mock_model_instance = checkpoint_manager.model 
        mock_model_instance.load_state_dict = MagicMock()

        loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))

        assert loaded_state is not None
        assert loaded_state['global_step'] == step
        
        mock_model_instance.load_state_dict.assert_called_once()
        optimizer.load_state_dict.assert_called_once()
        scheduler.load_state_dict.assert_called_once()

    def test_load_checkpoint_file_not_found(self, checkpoint_manager, tmp_path):
        """Test load returns None for non-existent file."""
        non_existent_path = tmp_path / "non_existent.pt"
        
        with patch.object(checkpoint_manager.logger, 'error') as mock_log_error:
            result = checkpoint_manager.load_checkpoint(path=str(non_existent_path))
            assert result is None
            mock_log_error.assert_called_once_with(f"Checkpoint file not found: {non_existent_path}")

    @patch('torch.load', side_effect=Exception("Corrupt file"))
    def test_load_checkpoint_torch_load_fails(self, mock_torch_load, checkpoint_manager, mock_objects, tmp_path):
        """Test graceful failure if torch.load raises an exception."""
        model, _, _, _ = mock_objects
        step = 900
        save_path = tmp_path / f"step_{step}.pt"
        save_path.touch()

        with patch.object(checkpoint_manager.logger, 'error') as mock_log_error:
            result = checkpoint_manager.load_checkpoint(path=str(save_path))
            assert result is None
            mock_torch_load.assert_called_once()
            mock_log_error.assert_called_once()
            assert f"Failed to load checkpoint from {save_path}: Corrupt file" in mock_log_error.call_args[0][0]

    def test_load_checkpoint_missing_keys(self, checkpoint_manager, mock_objects, tmp_path):
        """Test load handles missing essential keys gracefully (logs warnings)."""
        model, optimizer, _, _ = mock_objects
        step = 1000
        epoch = 10
        save_path = tmp_path / f"step_{step}.pt"
        
        torch.save({
            "epoch": epoch, 
            "global_step": step, 
            "optimizer_state_dict": optimizer.state_dict()
        }, save_path)

        model.load_state_dict = MagicMock()
        optimizer.load_state_dict = MagicMock()

        with patch.object(checkpoint_manager.logger, 'warning') as mock_log_warning:
            loaded_state = checkpoint_manager.load_checkpoint(path=str(save_path))
            
            assert loaded_state is not None
            assert loaded_state['global_step'] == step
            assert loaded_state['epoch'] == epoch
            
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