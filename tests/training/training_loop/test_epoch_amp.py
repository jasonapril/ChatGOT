import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
from torch.utils.data import DataLoader # Need this
from torch.optim import AdamW
from craft.config.schemas import TrainingConfig # <-- Correct import
from craft.models.base import Model # <-- Correct class name is Model
import logging

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.trainer import Trainer

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    class ProgressTracker: pass

class TestEpochAMP:
    """Tests for TrainingLoop train_epoch AMP functionality."""

    @patch('logging.getLogger')
    @patch('craft.training.training_loop.torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_amp_enabled(self,
                                     mock_tqdm,
                                     mock_cross_entropy,
                                     mock_autocast,
                                     mock_get_logger,
                                     mock_model,
                                     mock_optimizer,
                                     mock_dataloader,
                                     mock_device,
                                     mock_progress_tracker_instance,
                                     mock_scaler,
                                    ):
        """Test train_epoch when AMP is enabled."""
        # --- Setup Mocks ---
        mock_loop_logger = MagicMock(spec=logging.Logger)
        mock_loop_logger.level = logging.DEBUG
        mock_get_logger.return_value = mock_loop_logger
        
        mock_optimizer.reset_mock() # Reset mock at start of test
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # --- Setup Config for Loop (AMP Enabled) ---
        # Define required config values
        config = TrainingConfig(
            use_amp=True,
            gradient_accumulation_steps=1,
            batch_size=2, # Add required field
            log_interval=10, # Add required field
            num_epochs=1, # Add required field
            learning_rate=1e-4 # Add required field
            # Add other required fields with default values if necessary
        )

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=config # Pass the config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_scaler.is_enabled = MagicMock(return_value=True)
        mock_scaler.scale.return_value = mock_loss
        # <<< ADD side_effect to scaler.step >>>
        def scaler_step_side_effect(optimizer):
            optimizer.step()
        mock_scaler.step.side_effect = scaler_step_side_effect
        
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer
        mock_progress_tracker_instance.progress_bar = MagicMock() # <<< ADD progress_bar attribute

        # --- Run Epoch ---
        start_global_step = 0
        loop.train_epoch(trainer=mock_trainer, current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)

        # --- Assertions ---
        mock_autocast.assert_called_once_with(device_type=mock_device.type, enabled=True)
        mock_scaler.scale.assert_called_once_with(mock_loss)
        mock_scaler.step.assert_called_once_with(mock_optimizer)
        mock_scaler.update.assert_called_once()
        # Called at start + after step
        assert mock_optimizer.zero_grad.call_count == 2
        mock_progress_tracker_instance.update.assert_called_with(
            step=start_global_step + 1,
            loss=mock_loss.item(),
            learning_rate=ANY,
            tokens_per_second=ANY,
        )
        assert mock_cross_entropy.call_count == 1
        assert mock_autocast.return_value.__enter__.call_count == 1
        assert mock_autocast.return_value.__exit__.call_count == 1
        assert mock_optimizer.step.call_count == 1
        assert mock_optimizer.zero_grad.call_count == 2
        assert mock_scaler.scale.call_count == 1
        assert mock_scaler.step.call_count == 1
        assert mock_scaler.update.call_count == 1 