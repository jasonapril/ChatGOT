import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
import logging
from pydantic import ValidationError

# Import the class to test
from craft.training.trainer import Trainer
from craft.training.callbacks import CallbackList
from craft.training.checkpointing import CheckpointManager
from torch.cuda.amp import GradScaler as CudaGradScaler # Alias to avoid conflict if torch.amp is used
from craft.config.schemas import TrainingConfig # Import TrainingConfig


class TestTrainerInit:
    """Tests for Trainer initialization."""

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    def test_init_minimal_required(self,
                           mock_torch_device,
                           mock_grad_scaler,
                           mock_checkpoint_manager,
                           mock_get_logger,
                           mock_model,             # Fixture from conftest
                           mock_dataloader,        # Fixture from conftest
                           mock_optimizer,         # Fixture from conftest
                           default_training_config # Fixture from conftest
                          ):
        """Test Trainer initialization with minimal required arguments."""
        # --- Setup Mocks ---
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device
        mock_scaler_instance = MagicMock(spec=CudaGradScaler)
        mock_grad_scaler.return_value = mock_scaler_instance
        mock_logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger_instance
        mock_checkpoint_manager_instance = MagicMock(spec=CheckpointManager)
        mock_checkpoint_manager.return_value = mock_checkpoint_manager_instance
        
        # Mock config model_dump
        mock_config_dict = default_training_config.model_dump()


        # --- Initialize Trainer ---
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=default_training_config # Pass TrainingConfig object
        )

        # --- Assertions ---
        assert trainer.model == mock_model
        assert trainer.train_dataloader == mock_dataloader
        assert trainer.val_dataloader is None
        assert trainer.optimizer == mock_optimizer
        assert trainer.scheduler is None
        assert trainer.config == default_training_config # Check config object stored
        assert trainer.device == mock_cpu_device
        # assert trainer.checkpoint_dir is None # REMOVED checkpoint_dir
        assert trainer.use_amp == default_training_config.use_amp # Check derived from config
        assert trainer.gradient_accumulation_steps == default_training_config.gradient_accumulation_steps
        assert trainer.max_grad_norm == default_training_config.max_grad_norm
        assert trainer.log_interval == default_training_config.log_interval
        assert trainer.eval_interval == default_training_config.eval_interval
        assert trainer.save_interval == default_training_config.save_interval # Should compare against save_interval
        assert trainer.num_epochs == default_training_config.num_epochs
        assert trainer.resume_from_checkpoint is None

        assert trainer.logger == mock_logger_instance
        mock_get_logger.assert_any_call('Trainer')

        assert trainer.callbacks is not None
        assert trainer.callbacks.callbacks == []

        # Check CheckpointManager call (no checkpoint_dir, uses config.model_dump())
        mock_checkpoint_manager.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=None,
            scaler=trainer.scaler,
            config=mock_config_dict, # Expects dumped dict
            # checkpoint_dir=None, # REMOVED
            callbacks=trainer.callbacks,
            device=trainer.device,
            tokenizer=None, # Default tokenizer is None
            checkpoint_dir=None, # Expect None from default config
            max_checkpoints_to_keep=5 # Expect default 5 as keep_last is None
        )
        assert trainer.checkpoint_manager == mock_checkpoint_manager_instance

        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_metric == float('inf')
        assert trainer.metrics == {}

        mock_model.to.assert_called_once_with(mock_cpu_device)

        # Check scaler initialization (using the correct path with device type)
        mock_grad_scaler.assert_called_once_with(enabled=default_training_config.use_amp)

        assert trainer.compile_model == default_training_config.compile_model # Check default

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    @patch('craft.training.trainer.torch.cuda.is_available')
    def test_init_auto_device_detection(self,
                                mock_torch_device,
                                mock_cuda_available,
                                mock_checkpoint_manager, 
                                mock_get_logger,
                                mock_model,
                                mock_dataloader,
                                mock_optimizer,
                                default_training_config):
        """Test Trainer initializes with correct device when not specified."""
        # Test CUDA path
        mock_cuda_available.return_value = True
        mock_gpu_device = torch.device("cuda")
        mock_torch_device.return_value = mock_gpu_device
        
        Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=default_training_config,
            device=None # Explicitly None to trigger auto-detection
        )
        mock_model.to.assert_called_with(mock_gpu_device)
        mock_torch_device.assert_called_with("cuda") # Check device requested

        # Reset mock for CPU path
        mock_model.reset_mock()
        mock_torch_device.reset_mock()

        # Test CPU path
        mock_cuda_available.return_value = False
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device
        
        Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=default_training_config,
            device=None
        )
        mock_model.to.assert_called_with(mock_cpu_device)
        mock_torch_device.assert_called_with("cpu") # Check device requested 

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    def test_init_all_args_provided(self,
                           mock_torch_device,
                           mock_grad_scaler,
                           mock_checkpoint_manager,
                           mock_get_logger,
                           mock_model,             # Fixture from conftest
                           mock_dataloader,        # Fixture from conftest
                           mock_optimizer,         # Fixture from conftest
                           mock_scheduler,         # Fixture from conftest
                           mock_tokenizer,         # Fixture from conftest
                           mock_callback,          # Fixture from conftest
                           full_training_config    # Fixture from conftest
                          ):
        """Test Trainer initialization with most arguments provided."""
        # --- Setup Mocks & Args ---
        mock_gpu_device = torch.device("cuda")
        mock_torch_device.return_value = mock_gpu_device

        mock_scaler_instance = mock_grad_scaler.return_value
        # Test with use_amp = False
        mock_scaler_instance.is_enabled.return_value = False 
        use_amp_test_value = False 

        mock_logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger_instance
        mock_checkpoint_manager_instance = MagicMock(spec=CheckpointManager)
        mock_checkpoint_manager.return_value = mock_checkpoint_manager_instance
    
        mock_val_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
        mock_callback = MagicMock()
        callbacks = [mock_callback]
        mock_tokenizer = MagicMock() # Mock a tokenizer
        resume_path = "/tmp/checkpoints/latest"

        # Use the fixture directly
        test_config = full_training_config
        test_config.use_amp = use_amp_test_value # Override specific value if needed

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_val_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=test_config, # Pass the config object
            device=mock_gpu_device, # Explicitly pass device
            callbacks=callbacks,
            tokenizer=mock_tokenizer, # Pass tokenizer
            resume_from_checkpoint=resume_path
        )

        # --- Assertions ---
        assert trainer.model == mock_model
        assert trainer.train_dataloader == mock_dataloader
        assert trainer.val_dataloader == mock_val_dataloader
        assert trainer.optimizer == mock_optimizer
        assert trainer.scheduler == mock_scheduler
        assert trainer.config == test_config # Check config object stored
        assert trainer.device == mock_gpu_device
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.callbacks is not None
        assert trainer.callbacks.callbacks == callbacks
        assert trainer.callbacks.trainer == trainer # Check callbacks know trainer
        assert trainer.resume_from_checkpoint == resume_path

        # Check CheckpointManager call
        mock_checkpoint_manager.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=trainer.scaler,
            config=test_config.model_dump(), # Check dumped config is passed
            callbacks=trainer.callbacks,
            device=trainer.device,
            tokenizer=mock_tokenizer,
            checkpoint_dir=test_config.checkpoint_dir, # Expect None from fixture
            max_checkpoints_to_keep=5 # Expect default 5 as keep_last is None in fixture
        )

        # Verify resume path triggered the load_checkpoint call in CheckpointManager
        mock_checkpoint_manager_instance.load_checkpoint.assert_called_once_with(resume_path)

        mock_model.to.assert_called_once_with(mock_gpu_device)

        # Check scaler initialization
        mock_grad_scaler.assert_called_once_with(enabled=use_amp_test_value)

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.cuda.is_available')
    @patch('craft.training.trainer.torch.device')
    def test_init_auto_device_detection(self,
                                mock_torch_device,
                                mock_cuda_available,
                                mock_checkpoint_manager, 
                                mock_get_logger,
                                mock_model,
                                mock_dataloader,
                                mock_optimizer,
                                default_training_config):
        """Test Trainer initializes with correct device when not specified."""
        # Test CUDA path
        mock_cuda_available.return_value = True
        mock_gpu_device = torch.device("cuda")
        mock_torch_device.return_value = mock_gpu_device
        
        Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=default_training_config,
            device=None # Explicitly None to trigger auto-detection
        )
        mock_model.to.assert_called_with(mock_gpu_device)
        mock_torch_device.assert_called_with("cuda") # Check device requested

        # Reset mock for CPU path
        mock_model.reset_mock()
        mock_torch_device.reset_mock()

        # Test CPU path
        mock_cuda_available.return_value = False
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device
        
        Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=default_training_config,
            device=None
        )
        mock_model.to.assert_called_with(mock_cpu_device)
        mock_torch_device.assert_called_with("cpu") # Check device requested 