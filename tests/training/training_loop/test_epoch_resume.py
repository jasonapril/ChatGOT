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
    class ProgressTracker: pass

class TestEpochResume:
    """Tests for TrainingLoop train_epoch resume functionality."""

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_resume(self,
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
        """Test resuming training from a specific global step within an epoch."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_scaler.scale.return_value = mock_loss

        num_batches = 5
        batches = [(torch.randn(2, 4), torch.randint(0, 10, (2,))) for _ in range(num_batches)]
        multi_batch_dataloader = MagicMock(spec=DataLoader)
        multi_batch_dataloader.__iter__.return_value = iter(batches)
        multi_batch_dataloader.__len__.return_value = num_batches
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)

        # --- Setup Loop (AMP Disabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler

        # Resume logic setup ...
        current_epoch = 2
        steps_per_epoch = num_batches
        loaded_global_step = (current_epoch * steps_per_epoch) + 1
        expected_batches_to_process = num_batches - (loaded_global_step % steps_per_epoch) - 1
        start_global_step_for_epoch = current_epoch * steps_per_epoch

        # --- Run Epoch ---
        epoch_metrics = loop.train_epoch(
            current_epoch=current_epoch,
            global_step=start_global_step_for_epoch,
            progress=mock_progress_tracker_instance,
            loaded_global_step=loaded_global_step
        )

        # --- Assertions ---
        resume_msg_args = (
             f"Resuming epoch {current_epoch+1} from batch offset {(loaded_global_step % steps_per_epoch) + 1}/{steps_per_epoch} (global step {loaded_global_step + 1})",
         )
        mock_logger_fixture.info.assert_any_call(*resume_msg_args)
        assert mock_logger_fixture.info.call_count > 1
        assert mock_model.call_count == expected_batches_to_process
        assert mock_cross_entropy.call_count == expected_batches_to_process
        assert mock_optimizer.step.call_count == expected_batches_to_process
        assert mock_scaler.update.call_count == expected_batches_to_process
        assert mock_progress_tracker_instance.update.call_count == expected_batches_to_process
        # Correct the expected final global step calculation
        expected_final_global_step = start_global_step_for_epoch + expected_batches_to_process
        assert epoch_metrics.get('final_global_step') == expected_final_global_step 