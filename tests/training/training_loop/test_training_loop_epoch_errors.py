import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader # Need this

# Import the class to test
from craft.training.training_loop import TrainingLoop
from craft.training.progress import ProgressTracker # Import ProgressTracker
from craft.training.trainer import Trainer # Import Trainer
from craft.config.schemas import TrainingConfig # Import TrainingConfig

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    # If ProgressTracker can't be imported, create a dummy class for spec=True
    class ProgressTracker: pass

class TestTrainingLoopTrainEpochErrorHandling:
    """Tests for train_epoch error handling scenarios."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_runtime_error(
        self,
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
        """Test train_epoch handles runtime errors during forward pass."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_model.side_effect = RuntimeError("Simulated error")

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # --- Setup Loop ---
        minimal_config = TrainingConfig(batch_size=2, use_amp=False, gradient_accumulation_steps=1, log_interval=10, num_epochs=1, learning_rate=1e-4) # Create config object
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=minimal_config # Pass TrainingConfig object
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch & Assert Error Logged ---
        start_global_step = 0
        loop.train_epoch(trainer=mock_trainer, current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)
        # Assert error was logged, not raised
        mock_logger_fixture.error.assert_called_once()
        assert "Simulated error" in mock_logger_fixture.error.call_args[0][0]

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_tqdm_exception(
        self,
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
        """Test train_epoch handles exceptions during tqdm initialization."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # --- Setup Loop ---
        minimal_config = TrainingConfig(batch_size=2, use_amp=False, gradient_accumulation_steps=1, log_interval=10, num_epochs=1, learning_rate=1e-4) # Create config object
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=minimal_config # Pass TrainingConfig object
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch & Assert Warning --- #
        start_global_step = 0
        tqdm_error_message = "TQDM error"
        with patch('craft.training.training_loop.tqdm', side_effect=Exception(tqdm_error_message)) as mock_tqdm_local:
            epoch_metrics = loop.train_epoch(
                current_epoch=0,
                global_step=start_global_step,
                progress=mock_progress_tracker_instance, # Add progress
                trainer=mock_trainer # Pass mock trainer
            )

        # Assertions
        mock_model.train.assert_called_once()
        mock_progress_tracker_instance.update.assert_called()

    def test_train_epoch_bad_batch_format(
        self,
        mock_model,
        mock_optimizer,
        mock_device,
        mock_progress_tracker_instance,
        mock_scaler,
        mock_logger_fixture
    ):
        """Test train_epoch handles incorrect batch format from dataloader."""
        # --- Setup Mocks & Data ---
        bad_batch_dataloader = MagicMock(spec=DataLoader)
        bad_batch_dataloader.__iter__.return_value = iter([{'data': torch.randn(2, 4)}])
        bad_batch_dataloader.__len__.return_value = 1

        # --- Setup Loop ---
        minimal_config = TrainingConfig(batch_size=2, use_amp=False, gradient_accumulation_steps=1, log_interval=10, num_epochs=1, learning_rate=1e-4) # Create config object
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=bad_batch_dataloader,
            device=mock_device,
            config=minimal_config # Pass TrainingConfig object
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch & Assert Error Log --- #
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance, # Add progress
            trainer=mock_trainer # Pass mock trainer
        )

        # Assertions
        mock_model.train.assert_called_once()
        mock_logger_fixture.error.assert_called_once()
        args, kwargs = mock_logger_fixture.error.call_args
        assert "Unexpected batch format" in args[0]
        assert mock_model.call_count == 0 # Should only have been called for .train()
        mock_progress_tracker_instance.update.assert_not_called()

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_cuda_oom_error(
        self,
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
        """Test train_epoch handles CUDA OOM errors specifically."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss_tensor = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss_tensor

        # Explicitly define iterator behavior for this test
        mock_input = torch.randn(2, 4) # Example input
        mock_target = torch.randint(0, 10, (2,)) # Example target
        mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
        mock_dataloader.__len__.return_value = 1 # Ensure length is correct

        # Simulate OOM during backward pass
        mock_scaled_loss = MagicMock(spec=torch.Tensor)
        mock_scaler.scale = MagicMock(return_value=mock_scaled_loss)
        oom_error_message = "CUDA out of memory. Tried to allocate..."
        mock_scaled_loss.backward = MagicMock(side_effect=RuntimeError(oom_error_message))

        # --- Setup Loop ---
        minimal_config = TrainingConfig(batch_size=2, use_amp=False, gradient_accumulation_steps=1, log_interval=10, num_epochs=1, learning_rate=1e-4)
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config=minimal_config
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)
        mock_trainer = MagicMock(spec=Trainer)

        # --- Run Epoch & Assert OOM Error Logged ---
        start_global_step = 0
        loop.train_epoch(trainer=mock_trainer, current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)
        # Assert OOM error was logged, not raised
        mock_logger_fixture.error.assert_called_once()
        assert oom_error_message in mock_logger_fixture.error.call_args[0][0]
        assert "CUDA out of memory" in mock_logger_fixture.error.call_args[0][0] 