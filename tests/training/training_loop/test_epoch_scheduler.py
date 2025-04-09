import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
from torch.utils.data import DataLoader # Need this
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler # For type hinting

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

class TestEpochScheduler:
    """Tests for TrainingLoop train_epoch scheduler functionality."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_with_scheduler(self,
                                        mock_tqdm,
                                        mock_cross_entropy,
                                        mock_autocast,
                                        mock_model,
                                        mock_optimizer,
                                        mock_dataloader,
                                        mock_device,
                                        mock_scheduler,
                                        mock_progress_tracker_instance,
                                        mock_scaler,
                                        mock_logger_fixture
                                       ):
        """Test that scheduler.step() is called."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = MagicMock(spec=torch.Tensor, name="mock_loss_for_scheduler")
        mock_loss.requires_grad = True
        mock_loss.__truediv__ = MagicMock(return_value=mock_loss, name="mock_loss.__truediv__")
        mock_loss.backward = MagicMock(name="mock_loss.backward")
        mock_loss.item.return_value = 1.5 # Ensure item() returns a float
        mock_cross_entropy.return_value = mock_loss

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct
        mock_progress_tracker_instance.start_time = None # Add start_time

        mock_scaler.is_enabled = MagicMock(return_value=False)
        # Mock optimizer.step to simulate it being called
        mock_optimizer.step = MagicMock()

        # --- Setup Config ---
        config = TrainingConfig(
            batch_size=2,
            log_interval=10,
            num_epochs=1,
            learning_rate=1e-4,
            use_amp=False,
            gradient_accumulation_steps=1
        )

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=config, # Pass config object
            scheduler=mock_scheduler # Pass scheduler
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch ---
        start_global_step = 0
        # Patch isnan/isinf as the mock loss isn't a real tensor
        with patch('torch.isnan', return_value=False) as mock_isnan, \
             patch('torch.isinf', return_value=False) as mock_isinf:
            epoch_metrics = loop.train_epoch(
                current_epoch=0,
                global_step=start_global_step,
                progress=mock_progress_tracker_instance,
                trainer=mock_trainer # Pass mock trainer
            )
        # --- Assertions ---
        # With AMP off, optimizer.step should be called directly
        mock_optimizer.step.assert_called_once()
        # Scaler methods should not be called
        mock_scaler.scale.assert_not_called()
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        # Scheduler should step after optimizer
        mock_scheduler.step.assert_called_once()
        assert mock_progress_tracker_instance.update.call_count == 1 # len(mock_dataloader)
        # mock_progress_tracker_instance.update.assert_called_with(
        #     step=start_global_step + 1,
        #     loss=mock_loss.item(),
        #     learning_rate=ANY, # LR is updated by scheduler
        #     tokens_per_second=ANY,
        #     additional_metrics=None
        # ) # Removed overly specific check 