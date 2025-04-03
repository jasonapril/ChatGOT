import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader # Need this

# Import the class to test
from craft.training.training_loop import TrainingLoop

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
        mock_tqdm.return_value = enumerate(mock_dataloader)

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={}
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch & Assert Exception ---
        start_global_step = 0
        with pytest.raises(RuntimeError, match="Simulated error"):
            loop.train_epoch(
                current_epoch=0,
                global_step=start_global_step,
                progress=mock_progress_tracker_instance # Add progress
            )

        # Assertions
        mock_model.train.assert_called_once()
        mock_logger_fixture.error.assert_called_once()
        args, kwargs = mock_logger_fixture.error.call_args
        assert "Simulated error" in args[0]
        assert kwargs.get('exc_info', False) is True

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    def test_train_epoch_tqdm_exception(
        self,
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

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={}
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch & Assert Warning --- #
        start_global_step = 0
        tqdm_error_message = "TQDM error"
        with patch('craft.training.training_loop.tqdm', side_effect=Exception(tqdm_error_message)) as mock_tqdm_local:
            epoch_metrics = loop.train_epoch(
                current_epoch=0,
                global_step=start_global_step,
                progress=mock_progress_tracker_instance # Add progress
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
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=bad_batch_dataloader,
            device=mock_device,
            config={}
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch & Assert Error Log --- #
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance # Add progress
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
        # --- Setup Mocks ---\
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss_tensor = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss_tensor
        mock_tqdm.return_value = enumerate(mock_dataloader)

        # Simulate OOM during backward pass
        mock_scaled_loss = MagicMock()
        mock_scaler.scale.return_value = mock_scaled_loss
        oom_error_message = "CUDA out of memory. Tried to allocate..."
        mock_scaled_loss.backward.side_effect = RuntimeError(oom_error_message)

        # --- Setup Loop ---\
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={}
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False) # Keep AMP off for simplicity

        # --- Run Epoch & Assert Exception ---\
        start_global_step = 0
        with pytest.raises(RuntimeError, match=oom_error_message):
            loop.train_epoch(
                current_epoch=0,\
                global_step=start_global_step,\
                progress=mock_progress_tracker_instance
            )

        # Assertions
        mock_model.train.assert_called_once()
        mock_scaled_loss.backward.assert_called_once() # Ensure backward was attempted
        mock_logger_fixture.error.assert_called_once()
        args, kwargs = mock_logger_fixture.error.call_args
        assert "CUDA OOM encountered" in args[0]
        assert oom_error_message not in args[0] # Should have custom OOM message
        assert not kwargs # kwargs should be empty for the specific OOM log 