import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
from torch.utils.data import DataLoader # Need this

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.callbacks.base import Callback # Ensure Base Callback is imported
from craft.training.progress import ProgressTracker # Import ProgressTracker
from craft.training.trainer import Trainer # Import Trainer

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    class ProgressTracker: pass

# Import Callback for spec
from craft.training.callbacks import Callback

class TestEpochCallbacks:
    """Tests for TrainingLoop train_epoch callback interactions."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_callbacks(self,
                                   mock_tqdm,
                                   mock_cross_entropy,
                                   mock_autocast,
                                   mock_model,
                                   mock_optimizer,
                                   mock_dataloader,
                                   mock_device,
                                   mock_progress_tracker_instance,
                                   mock_scaler,
                                   mock_logger_fixture,
                                   mock_callback_fixture
                                  ):
        """Test that step begin/end callbacks are called."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.5, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_callback = mock_callback_fixture
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=1,
            callbacks=[mock_callback]
        )
        loop.scaler = mock_scaler
        # --- Run Epoch ---
        start_global_step = 10
        loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance,
            trainer=mock_trainer # Pass mock trainer
        )
        # --- Assertions ---
        mock_callback.on_step_begin.assert_called_once_with(start_global_step, logs=ANY)
        mock_callback.on_step_end.assert_called_once_with(step=0, global_step=start_global_step + 1, metrics=ANY)
        assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
        mock_progress_tracker_instance.update.assert_called_with(
            step=start_global_step + 1, 
            loss=mock_loss.item(),
            learning_rate=ANY, # LR is updated by scheduler
            tokens_per_second=ANY,
            additional_metrics=None
        ) 