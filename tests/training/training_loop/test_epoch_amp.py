import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
from torch.utils.data import DataLoader # Need this

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

    @patch('craft.training.training_loop.torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_amp_enabled(self,
                                     mock_tqdm,
                                     mock_cross_entropy,
                                     mock_autocast,
                                     mock_model,
                                     mock_optimizer,
                                     mock_dataloader,
                                     mock_device,
                                     mock_scaler,
                                     mock_progress_tracker_instance,
                                     mock_callback_fixture):
        """Test that train_epoch runs with AMP enabled and uses the scaler."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.5, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # mock_tqdm.return_value = enumerate(mock_dataloader) # Removed
        mock_scaler.is_enabled = MagicMock(return_value=True) # AMP is ON

        # --- Setup Loop (AMP Enabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            use_amp=True,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler

        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance,
            trainer=mock_trainer # Pass mock trainer
        )
        # --- Assertions ---
        mock_autocast.assert_called_with(device_type=mock_device.type, enabled=True)
        accumulation_steps = 1
        mock_scaler.scale.assert_called_once_with(mock_loss / accumulation_steps)
        mock_scaler.mock_scaled_loss.backward.assert_called_once()
        mock_scaler.unscale_.assert_called_once_with(mock_optimizer)
        mock_scaler.step.assert_called_once_with(mock_optimizer)
        mock_scaler.update.assert_called_once()
        assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
        mock_progress_tracker_instance.update.assert_called_with(
            step=start_global_step + 1, 
            loss=mock_loss.item(),
            learning_rate=None, # No scheduler in this test
            tokens_per_second=ANY,
            additional_metrics=None
        ) 