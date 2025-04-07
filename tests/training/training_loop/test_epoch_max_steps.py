import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset # Need these

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.progress import ProgressTracker # Import ProgressTracker
from craft.training.trainer import Trainer # Import Trainer

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    class ProgressTracker: pass

class TestEpochMaxSteps:
    """Tests for TrainingLoop train_epoch max_steps functionality."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_max_steps(self,
                                   mock_tqdm,
                                   mock_cross_entropy,
                                   mock_autocast,
                                   mock_model,
                                   mock_optimizer,
                                   mock_device,
                                   mock_progress_tracker_instance,
                                   mock_scaler,
                                   mock_logger_fixture):
        """Test that training stops early if max_steps is reached."""
        # --- Setup Mocks & Data ---
        num_batches = 5
        max_steps_to_run = 2 # Set max_steps less than total batches
        batches = [(torch.randn(2, 4), torch.randint(0, 10, (2,))) for _ in range(num_batches)]
        multi_batch_dataloader = MagicMock(spec=DataLoader)
        multi_batch_dataloader.__iter__.return_value = iter(batches)
        multi_batch_dataloader.__len__.return_value = num_batches

        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config={}, # Initialize with empty config
            use_amp=False,
            gradient_accumulation_steps=1,
            max_steps=max_steps_to_run # PASS max_steps here
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance, trainer=mock_trainer) # Pass trainer

        # --- Assertions ---
        # Verify the loop ran exactly for max_steps
        assert mock_model.call_count == max_steps_to_run
        assert mock_optimizer.step.call_count == max_steps_to_run
        assert mock_scaler.step.call_count == 0 # Scaler shouldn't be called if use_amp=False

        # Check metrics returned
        assert "loss" in epoch_metrics 