import pytest
import torch
from unittest.mock import MagicMock, patch, call, ANY
from torch.utils.data import DataLoader # Need this

# Import the class to test
from craft.training.training_loop import TrainingLoop

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
                               mock_dataloader, # Use the fixed dataloader
                               mock_device,
                               mock_progress_tracker_instance, # Use the new fixture
                               mock_scaler,
                               mock_logger_fixture,
                               start_global_step # Added for update call check
                              ):
        """Test a basic run of train_epoch with minimal setup."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None # Simple context manager mock
        mock_autocast.return_value.__exit__.return_value = None
        # Define the mock loss value to be returned by cross_entropy
        mock_loss_value = 1.0
        mock_loss = torch.tensor(mock_loss_value, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss # Mock returns a tensor

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # --- Setup Loop (Defaults, AMP Disabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={}
        )
        loop.scaler = mock_scaler # Use the mocked scaler
        loop.scaler.is_enabled = MagicMock(return_value=False) # Ensure AMP is off

        # --- Run Epoch ---
        current_epoch = 1
        start_global_step_for_epoch = start_global_step
        epoch_metrics = loop.train_epoch(
            current_epoch=current_epoch,
            global_step=start_global_step_for_epoch,
            progress=mock_progress_tracker_instance
        )

        # --- Assertions ---
        mock_model.train.assert_called_once()
        assert mock_optimizer.zero_grad.call_count == 1 + len(mock_dataloader)
        assert mock_cross_entropy.call_count == len(mock_dataloader)
        assert mock_model.call_count == len(mock_dataloader)
        # Scaler.scale IS called, but step/update are not when use_amp=False
        # mock_scaler.scale.assert_not_called() # Removed
        mock_optimizer.step.assert_called_once() # Optimizer step is called directly
        mock_scaler.step.assert_not_called() # Scaler step is NOT called
        mock_scaler.update.assert_called_once() # Scaler update IS called (handles no-op internally)
        assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
        last_step = start_global_step_for_epoch + len(mock_dataloader)
        mock_progress_tracker_instance.update.assert_called_with(
            step=last_step, 
            loss=mock_loss_value,
            learning_rate=None, 
            tokens_per_second=ANY, 
            additional_metrics=None
        )
        assert epoch_metrics.get('loss') == mock_loss_value # Check avg loss

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
        accumulation_steps = 2
        num_batches = 4
        batches = [(torch.randn(2, 4), torch.randint(0, 10, (2,))) for _ in range(num_batches)]
        multi_batch_dataloader = MagicMock(spec=DataLoader)
        multi_batch_dataloader.__iter__.return_value = iter(batches)
        multi_batch_dataloader.__len__.return_value = num_batches

        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)

        # --- Setup Loop (AMP Disabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=accumulation_steps
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance
        )

        # --- Assertions ---
        mock_model.train.assert_called_once()
        assert mock_cross_entropy.call_count == num_batches
        assert mock_model.call_count == num_batches

        expected_steps = num_batches / accumulation_steps
        assert mock_optimizer.step.call_count == expected_steps
        assert mock_scaler.update.call_count == expected_steps
        assert mock_optimizer.zero_grad.call_count == 1 + expected_steps

        assert mock_progress_tracker_instance.update.call_count == expected_steps
        assert mock_progress_tracker_instance.update.call_count == expected_steps
        assert 'loss' in epoch_metrics

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
        # --- Setup Mocks & Data ---\
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
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)

        # --- Setup Loop (AMP Disabled) ---\
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=accumulation_steps
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch ---\
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance
        )

        # --- Assertions ---\
        mock_model.train.assert_called_once()
        assert mock_cross_entropy.call_count == num_batches
        assert mock_model.call_count == num_batches

        # Expect step count to be ceiling(num_batches / accumulation_steps)
        expected_steps = 3 # ceil(5 / 2)
        assert mock_optimizer.step.call_count == expected_steps
        assert mock_scaler.update.call_count == expected_steps
        # zero_grad is called once at the start, then after each step
        assert mock_optimizer.zero_grad.call_count == 1 + expected_steps

        # Progress updated after each optimizer step
        assert mock_progress_tracker_instance.update.call_count == expected_steps
        assert 'loss' in epoch_metrics
        # Check the *last* call to update
        last_call_args, last_call_kwargs = mock_progress_tracker_instance.update.call_args
        expected_last_global_step = expected_steps # Step increments after update
        assert last_call_kwargs.get('step') == expected_last_global_step
        assert last_call_kwargs.get('loss') == mock_loss.item()
        assert last_call_kwargs.get('learning_rate') is None # No scheduler
        assert 'tokens_per_second' in last_call_kwargs
        assert last_call_kwargs.get('additional_metrics') is None
        assert epoch_metrics.get('final_global_step') == expected_last_global_step 