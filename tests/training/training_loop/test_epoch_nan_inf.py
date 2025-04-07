import pytest
import torch
from unittest.mock import MagicMock, patch, call
from torch.utils.data import DataLoader # Need this

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.progress import ProgressTracker # Import ProgressTracker
from craft.training.trainer import Trainer # Import Trainer

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    class ProgressTracker: pass

class TestEpochNanInf:
    """Tests for TrainingLoop train_epoch NaN/Inf handling."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_nan_inf_loss(self,
                                      mock_tqdm,
                                      mock_cross_entropy,
                                      mock_autocast,
                                      mock_model,
                                      mock_optimizer,
                                      mock_dataloader,
                                      mock_device,
                                      mock_progress_tracker_instance,
                                      mock_scaler,
                                      mock_logger_fixture
                                     ):
        """Test that NaN/Inf loss skips the optimizer step and logs a warning."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(float('nan'), requires_grad=True)
        mock_cross_entropy.return_value = mock_loss

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # mock_tqdm.return_value = enumerate(mock_dataloader) # Removed - train_epoch no longer uses tqdm
        mock_scaler.is_enabled = MagicMock(return_value=False)

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer
        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance,
            trainer=mock_trainer # Pass mock trainer
        )
        # --- Assertions ---
        mock_model.assert_called_once()
        mock_cross_entropy.assert_called_once()
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        mock_optimizer.step.assert_not_called()
        mock_logger_fixture.warning.assert_called_once()
        assert "NaN/Inf loss detected" in mock_logger_fixture.warning.call_args[0][0]
        assert epoch_metrics.get('final_global_step') == start_global_step # Step should not have incremented
        mock_progress_tracker_instance.update.assert_not_called() 