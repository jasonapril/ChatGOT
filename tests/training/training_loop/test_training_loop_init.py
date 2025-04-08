import pytest
import torch
from unittest.mock import MagicMock, patch
import logging
# Import GradScaler from torch.amp to match TrainingLoop implementation
from torch.amp import GradScaler

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.callbacks import CallbackList
from craft.config.schemas import TrainingConfig # Import TrainingConfig

# --- Test Class ---

class TestTrainingLoopInit:
    """Tests for TrainingLoop initialization."""

    def test_init_defaults(self, mock_model, mock_optimizer, mock_dataloader, mock_device):
        """Test TrainingLoop initialization with default arguments."""
        # Create minimal required TrainingConfig
        minimal_config = TrainingConfig(batch_size=1, learning_rate=1e-4)
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=minimal_config # Pass TrainingConfig object
        )
        assert loop.model == mock_model
        assert loop.optimizer == mock_optimizer
        assert loop.train_dataloader == mock_dataloader
        assert loop.device == mock_device
        assert loop.scheduler is None
        assert loop.gradient_accumulation_steps == 1
        assert loop.max_grad_norm is None
        assert loop.use_amp is False
        assert isinstance(loop.scaler, GradScaler) # Check type
        assert not loop.scaler.is_enabled() # Disabled by default
        assert isinstance(loop.callbacks, CallbackList)
        assert loop.callbacks.callbacks == [] # Check the internal list
        assert isinstance(loop.logger, logging.Logger) # Check logger type

    def test_init_all_args(self, mock_model, mock_optimizer, mock_dataloader, mock_scheduler, mock_device, mock_scaler, mock_callback_fixture):
        """Test TrainingLoop initialization with all arguments provided."""
        callbacks = [mock_callback_fixture]
        
        # Create TrainingConfig with non-default values for testing
        test_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            use_amp=True,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            # Let other fields use defaults
            num_epochs=1 # Need num_epochs or max_steps
        )

        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            scheduler=mock_scheduler,
            device=mock_device,
            config=test_config,
            callbacks=callbacks,
        )

        assert loop.model == mock_model
        assert loop.optimizer == mock_optimizer
        assert loop.train_dataloader == mock_dataloader
        assert loop.scheduler == mock_scheduler
        assert loop.device == mock_device
        assert loop.gradient_accumulation_steps == test_config.gradient_accumulation_steps
        assert loop.max_grad_norm == test_config.max_grad_norm
        assert loop.use_amp == test_config.use_amp
        assert isinstance(loop.scaler, GradScaler) # Check type
        assert loop.scaler.is_enabled() == test_config.use_amp
        assert loop.config == test_config
        assert isinstance(loop.callbacks, CallbackList)
        assert loop.callbacks.callbacks == callbacks # Check the internal list 