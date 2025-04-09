import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset # Need these
from torch.optim import Optimizer

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.progress import ProgressTracker # Import ProgressTracker
from craft.training.trainer import Trainer # Import Trainer
from craft.config.schemas import TrainingConfig # Import TrainingConfig

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
        mock_progress_tracker_instance.start_time = None # Add start_time

        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)

        # --- Setup Config ---
        config = TrainingConfig(
            batch_size=2,
            log_interval=10,
            num_epochs=1,
            learning_rate=1e-4,
            use_amp=False,
            gradient_accumulation_steps=1,
            max_steps=max_steps_to_run # Set max_steps
        )

        # --- Setup Loop ---
        mock_optimizer.reset_mock() # Reset mock before use in this test
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config=config # Pass config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(trainer=mock_trainer, current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)

        # --- Assertions ---
        assert mock_model.call_count == config.max_steps # Model called max_steps times
        assert mock_optimizer.step.call_count == config.max_steps # Optimizer stepped max_steps times
        # Scaler step/update are not called when AMP is False
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        assert 'average_epoch_loss' in epoch_metrics # Check key in the returned dict
        assert 'steps_completed_in_epoch' in epoch_metrics # Check key in the returned dict
        assert epoch_metrics['steps_completed_in_epoch'] == config.max_steps # Check steps completed
        mock_progress_tracker_instance.update.assert_called()
        assert mock_progress_tracker_instance.update.call_count == config.max_steps
        assert mock_cross_entropy.call_count == config.max_steps
        assert mock_optimizer.zero_grad.call_count == 1 + config.max_steps
        assert "average_epoch_loss" in epoch_metrics # Check average loss key exists 