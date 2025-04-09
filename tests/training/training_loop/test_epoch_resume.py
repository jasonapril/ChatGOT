import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader # Need this
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
        mock_optimizer.reset_mock() # Reset mock at start of test
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
        mock_progress_tracker_instance.start_time = None # Add start_time

        # --- Setup Config ---
        config = TrainingConfig(
            batch_size=2,
            log_interval=10,
            num_epochs=1,
            learning_rate=1e-4,
            use_amp=False,
            gradient_accumulation_steps=1
        )

        # --- Setup Loop (AMP Disabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config=config # Pass config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch with resume ---
        start_epoch = 0
        initial_global_step = 0 # Assume training starts at 0
        loaded_global_step = 2 # Resume from global step 2 (which means skip first 2 batches)

        loop.train_epoch(
            trainer=mock_trainer,
            current_epoch=start_epoch,
            global_step=initial_global_step,
            progress=mock_progress_tracker_instance,
            loaded_global_step=loaded_global_step # Pass the resume step
        )

        # --- Assertions ---
        # Total model calls should be total batches - skipped batches
        expected_calls = num_batches - (loaded_global_step + 1)
        assert mock_model.call_count == expected_calls
        assert mock_cross_entropy.call_count == expected_calls
        # With grad_accum=1 and AMP off, optimizer steps directly after each processed batch
        expected_steps = expected_calls
        assert mock_optimizer.step.call_count == expected_steps
        # Called at start + after each actual step
        assert mock_optimizer.zero_grad.call_count == 1 + expected_steps
        # Scaler methods should NOT be called
        mock_scaler.scale.assert_not_called()
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        # update should be called once per optimizer step
        assert mock_progress_tracker_instance.update.call_count == expected_steps

        resume_msg_args = (
             f"Resuming epoch {start_epoch+1} from batch offset {(loaded_global_step % num_batches) + 1}/{num_batches} (global step {loaded_global_step + 1})",
         )
        mock_logger_fixture.info.assert_any_call(*resume_msg_args)
        assert mock_logger_fixture.info.call_count > 1
        expected_final_global_step = initial_global_step + expected_calls
        # assert mock_progress_tracker_instance.get('final_global_step') == expected_final_global_step # ProgressTracker doesn't store final step 

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_resume_gradient_accumulation(self,
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
        """Test resuming training with gradient accumulation."""
        # --- Setup Mocks ---
        mock_optimizer.reset_mock() # Reset mock at start of test
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_scaler.scale.return_value = mock_loss

        num_batches = 5
        accumulation_steps = 2
        batches = [(torch.randn(2, 4), torch.randint(0, 10, (2,))) for _ in range(num_batches)]
        multi_batch_dataloader = MagicMock(spec=DataLoader)
        multi_batch_dataloader.__iter__.return_value = iter(batches)
        multi_batch_dataloader.__len__.return_value = num_batches
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)
        mock_progress_tracker_instance.start_time = None # Add start_time

        # --- Setup Config ---
        config = TrainingConfig(
            batch_size=2,
            log_interval=10,
            num_epochs=1,
            learning_rate=1e-4,
            use_amp=False,
            gradient_accumulation_steps=accumulation_steps
        )

        # --- Setup Loop (AMP Disabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config=config # Pass config object
        )
        loop.scaler = mock_scaler # Inject mock scaler
        mock_scaler.update = MagicMock() # Prevent unexpected side-effects
        mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

        # --- Run Epoch with resume ---
        start_epoch = 0
        initial_global_step = 0 # Assume training starts at 0
        loaded_global_step = 2 # Resume from global step 2 (batch index 2)

        loop.train_epoch(
            trainer=mock_trainer,
            current_epoch=start_epoch,
            global_step=initial_global_step,
            progress=mock_progress_tracker_instance,
            loaded_global_step=loaded_global_step # Pass the resume step
        )

        # --- Assertions ---
        # Total model calls should be total batches - batches skipped
        # Batches skipped = index 0, 1, 2 (loaded_global_step)
        expected_calls = num_batches - (loaded_global_step + 1)
        assert mock_model.call_count == expected_calls
        assert mock_cross_entropy.call_count == expected_calls
        # Calculate expected optimizer steps considering accumulation and resume point
        # Batches processed: index 3, 4
        # Step happens on batch 3 (idx=3, (3+1)%2==0)
        # Step happens on batch 4 (idx=4, last batch)
        expected_steps = 2
        assert mock_optimizer.step.call_count == expected_steps
        # Called at start + after each actual step
        assert mock_optimizer.zero_grad.call_count == 1 + expected_steps
        # Scaler methods should NOT be called
        mock_scaler.scale.assert_not_called()
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        # update should be called once per optimizer step
        assert mock_progress_tracker_instance.update.call_count == expected_steps

        # Correct resume message check
        resume_batch_offset = loaded_global_step % num_batches # Batch index to start processing *after*
        resume_msg_args = (
             f"Resuming epoch {start_epoch+1} from batch offset {resume_batch_offset + 1}/{num_batches} (global step {loaded_global_step + 1})",
         )
        mock_logger_fixture.info.assert_any_call(*resume_msg_args)
        mock_logger_fixture.info.assert_any_call(f"Fast-forwarded to batch {resume_batch_offset + 1}. Resuming training.")
        # expected_final_global_step = initial_global_step + expected_calls # Removed assertion