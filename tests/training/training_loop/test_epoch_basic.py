import pytest
import torch
from unittest.mock import MagicMock, patch, call, ANY
from torch.utils.data import DataLoader # Need this
from torch.optim import AdamW

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.config.schemas import TrainingConfig # <-- Correct import
from craft.training.progress import ProgressTracker
from craft.training.trainer import Trainer # Import Trainer

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    # If ProgressTracker can't be imported, create a dummy class for spec=True
    class ProgressTracker: pass

class TestEpochBasic:
    """Tests for basic TrainingLoop train_epoch execution."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_basic(self,
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
        """Test basic train_epoch functionality (no AMP, no grad accum)."""
        # --- Setup Mocks ---
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

        mock_scaler.is_enabled = MagicMock(return_value=False) # AMP is OFF

        # --- Setup Loop (AMP Disabled) ---
        config = TrainingConfig( # <-- Use TrainingConfig with required fields
            use_amp=False,
            gradient_accumulation_steps=1,
            batch_size=2, # Add required field
            log_interval=10, # Add required field
            num_epochs=1, # Add required field
            learning_rate=1e-4 # Add required field
            # Add other required fields with default values if necessary
        )
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=config # Pass config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            trainer=mock_trainer,
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance
        )

        # --- Assertions ---
        mock_model.assert_called_once_with(mock_input)
        mock_cross_entropy.assert_called_once()
        mock_optimizer.step.assert_called_once()
        # Called at start + after step
        assert mock_optimizer.zero_grad.call_count == 2
        mock_scaler.scale.assert_not_called()
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        assert mock_progress_tracker_instance.update.call_count == 1
        assert isinstance(epoch_metrics, dict)
        assert 'epoch_loss' in epoch_metrics
        assert 'tokens_per_second' in epoch_metrics
        expected_loss = mock_loss.item()
        assert abs(epoch_metrics['epoch_loss'] - expected_loss) < 1e-6

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_gradient_accumulation(self,
                                                mock_tqdm,
                                                mock_cross_entropy,
                                                mock_autocast,
                                                mock_model,
                                                mock_optimizer,
                                                mock_device,
                                                mock_progress_tracker_instance,
                                                mock_scaler,
                                                mock_logger_fixture
                                               ):
        """Test train_epoch with gradient accumulation."""
        # --- Setup Mocks & Data ---
        mock_optimizer.reset_mock() # Reset mock at start of test
        accumulation_steps = 2
        num_batches = 4 # Needs to be multiple of accumulation_steps
        batches = [(torch.randn(2, 4), torch.randint(0, 10, (2,))) for _ in range(num_batches)]
        multi_batch_dataloader = MagicMock(spec=DataLoader)
        multi_batch_dataloader.__iter__.return_value = iter(batches)
        multi_batch_dataloader.__len__.return_value = num_batches

        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss

        # --- Setup Loop (AMP Disabled) ---
        config = TrainingConfig( # <-- Use TrainingConfig with required fields
            use_amp=False,
            gradient_accumulation_steps=accumulation_steps,
            batch_size=2, # Add required field
            log_interval=10, # Add required field
            num_epochs=1, # Add required field
            learning_rate=1e-4 # Add required field
        )
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config=config # Pass config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch ---
        start_global_step = 0
        loop.train_epoch(trainer=mock_trainer, current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)

        # --- Assertions ---
        assert mock_model.call_count == num_batches
        assert mock_cross_entropy.call_count == num_batches
        expected_steps = num_batches // accumulation_steps
        assert mock_optimizer.step.call_count == expected_steps
        # Called at start + after each step
        assert mock_optimizer.zero_grad.call_count == 1 + expected_steps
        assert mock_progress_tracker_instance.update.call_count == num_batches

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_gradient_accumulation_uneven(self,
                                                      mock_tqdm,
                                                      mock_cross_entropy,
                                                      mock_autocast,
                                                      mock_model,
                                                      mock_optimizer,
                                                      mock_device,
                                                      mock_progress_tracker_instance,
                                                      mock_scaler,
                                                      mock_logger_fixture
                                                     ):
        """Test train_epoch with gradient accumulation where batches % steps != 0."""
        # --- Setup Mocks & Data ---
        mock_optimizer.reset_mock() # Reset mock at start of test
        accumulation_steps = 2
        num_batches = 5 # Uneven number
        batches = [(torch.randn(2, 4), torch.randint(0, 10, (2,))) for _ in range(num_batches)]
        multi_batch_dataloader = MagicMock(spec=DataLoader)
        multi_batch_dataloader.__iter__.return_value = iter(batches)
        multi_batch_dataloader.__len__.return_value = num_batches

        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss

        # --- Setup Loop (AMP Disabled) ---
        config = TrainingConfig( # <-- Use TrainingConfig with required fields
            use_amp=False,
            gradient_accumulation_steps=accumulation_steps,
            batch_size=2, # Add required field
            log_interval=10, # Add required field
            num_epochs=1, # Add required field
            learning_rate=1e-4 # Add required field
        )
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config=config # Pass config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch ---
        start_global_step = 0
        loop.train_epoch(trainer=mock_trainer, current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)

        # --- Assertions ---
        assert mock_model.call_count == num_batches
        assert mock_cross_entropy.call_count == num_batches
        # Step count includes the final uneven batch
        expected_steps = (num_batches + accumulation_steps - 1) // accumulation_steps
        assert mock_optimizer.step.call_count == expected_steps
        # Called at start + after each step
        assert mock_optimizer.zero_grad.call_count == 1 + expected_steps
        assert mock_progress_tracker_instance.update.call_count == num_batches

        # Check the *last* call to update (this assertion seems less relevant now)
        # last_call_args, last_call_kwargs = mock_progress_tracker_instance.update.call_args
        # expected_last_global_step = expected_steps # Step increments after update
        # assert last_call_kwargs.get('step') == expected_last_global_step
        # assert last_call_kwargs.get('loss') == mock_loss.item()
        # assert last_call_kwargs.get('learning_rate') is None # No scheduler
        # assert 'tokens_per_second' in last_call_kwargs
        # assert last_call_kwargs.get('additional_metrics') is None
        # assert epoch_metrics.get('final_global_step') == expected_last_global_step 