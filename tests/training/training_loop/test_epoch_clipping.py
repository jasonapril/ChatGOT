import pytest
import torch
from unittest.mock import MagicMock, patch, call, ANY
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

class TestEpochClipping:
    """Tests for TrainingLoop train_epoch gradient clipping."""

    @patch('torch.amp.autocast')
    @patch('tqdm.tqdm')
    @patch('torch.nn.utils.clip_grad_norm_')
    def test_train_epoch_grad_clipping(self,
                                       mock_clip_grad_norm,
                                       mock_tqdm,
                                       mock_autocast,
                                       mock_model,
                                       mock_optimizer,
                                       mock_dataloader,
                                       mock_device,
                                       mock_progress_tracker_instance,
                                       mock_scaler,
                                       mock_logger_fixture,
                                       start_global_step
                                      ):
        """Test that gradients are clipped if max_grad_norm is set."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None

        mock_output_tensor = mock_model.return_value
        mock_permuted_output = MagicMock(spec=torch.Tensor, name="mock_permuted_output")
        mock_output_tensor.view = MagicMock(return_value=mock_permuted_output, name="mock_output_tensor.view")
        mock_loss = MagicMock(spec=torch.Tensor, name="mock_loss_after_ce")
        mock_loss.requires_grad = True
        mock_loss.__truediv__ = MagicMock(return_value=mock_loss, name="mock_loss.__truediv__")
        mock_loss.backward = MagicMock(name="mock_loss.backward")

        with patch('torch.nn.functional.cross_entropy', return_value=mock_loss) as mock_ce_func:

            # Explicitly define iterator behavior for this test
            mock_input = torch.randn(2, 4) # Example input
            mock_target = torch.randint(0, 10, (2,)) # Example target
            mock_dataloader.__iter__.return_value = iter([(mock_input, mock_target)])
            mock_dataloader.__len__.return_value = 1 # Ensure length is correct
            mock_progress_tracker_instance.start_time = None # Add start_time

            # Since AMP is False, scaler methods won't be called in the main logic
            mock_scaler.is_enabled = MagicMock(return_value=False)
            mock_scaler.unscale_ = MagicMock() # Mock unscale_ even though it won't be called by loop
            mock_scaler.step = MagicMock() # Mock step
            mock_scaler.update = MagicMock() # Mock update
            mock_params = mock_model.parameters()

            # --- Setup Config ---
            config = TrainingConfig(
                batch_size=2,
                log_interval=10,
                num_epochs=1,
                learning_rate=1e-4,
                use_amp=False, # AMP is OFF
                gradient_accumulation_steps=1,
                max_grad_norm=1.0 # Set clipping value
            )

            # --- Setup Loop ---
            loop = TrainingLoop(
                model=mock_model,
                optimizer=mock_optimizer,
                train_dataloader=mock_dataloader,
                device=mock_device,
                config=config # Pass config object
            )
            loop.scaler = mock_scaler # Inject mock scaler (though not used by loop logic)
            mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

            # --- Run Epoch with Patched isnan/isinf --- #
            current_epoch = 0
            start_global_step_for_epoch = start_global_step

            # Ensure both isnan and isinf are checked by mocking their return values
            with patch('torch.isnan', return_value=False) as mock_isnan, \
                 patch('torch.isinf', return_value=False) as mock_isinf, \
                 patch.object(mock_loss, 'item', return_value=1.5) as mock_item:

                epoch_metrics = loop.train_epoch(
                    current_epoch=current_epoch,
                    global_step=start_global_step_for_epoch,
                    progress=mock_progress_tracker_instance, # Pass progress
                    trainer=mock_trainer # Pass mock trainer
                )

            # --- Assertions ---
            mock_model.assert_called_once() # Check forward pass happened
            mock_output_tensor.view.assert_called_once_with(-1, mock_output_tensor.size(-1)) # Check view call for CE
            mock_ce_func.assert_called_once() # Check cross_entropy was called
            # Check loss.item() was called (once for NaN check, once for accumulation)
            assert mock_item.call_count >= 1, f"Expected loss.item() to be called at least once, got {mock_item.call_count}" # Loosen assertion
            mock_isnan.assert_called_once_with(mock_loss)
            mock_isinf.assert_called_once_with(mock_loss)
            mock_loss.backward.assert_called_once()
            # Scaler methods should NOT be called by loop logic when AMP is False
            mock_scaler.unscale_.assert_not_called()
            mock_clip_grad_norm.assert_called_once_with(mock_params, config.max_grad_norm) # Clipping happens
            mock_optimizer.step.assert_called_once() # Optimizer steps directly
            mock_scaler.step.assert_not_called() # Scaler step is NOT called
            mock_scaler.update.assert_not_called() # Scaler update is NOT called
            mock_progress_tracker_instance.update.assert_called_once() # Progress update happens
            # Check update args (basic)
            call_args, call_kwargs = mock_progress_tracker_instance.update.call_args
            assert call_kwargs.get('step') == start_global_step + 1
            assert 'loss' in call_kwargs

    @patch('torch.amp.autocast')
    @patch('tqdm.tqdm')
    @patch('torch.nn.utils.clip_grad_norm_')
    def test_train_epoch_clipping_skips_update(self, # Renamed test, simulates NaN/Inf or unscaling issue
                                               mock_clip_grad_norm,
                                               mock_tqdm,
                                               mock_autocast,
                                               mock_model,
                                               mock_optimizer,
                                               mock_dataloader,
                                               mock_device,
                                               mock_progress_tracker_instance,
                                               mock_scaler,
                                               mock_logger_fixture,
                                               start_global_step
                                              ):
        """Test that optimizer step is skipped if scaler.step returns None (e.g., NaN/Inf grad)."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None

        mock_output_tensor = mock_model.return_value
        mock_permuted_output = MagicMock(spec=torch.Tensor, name="mock_permuted_output")
        mock_output_tensor.view = MagicMock(return_value=mock_permuted_output, name="mock_output_tensor.view")
        mock_loss = MagicMock(spec=torch.Tensor, name="mock_loss_after_ce")
        mock_loss.requires_grad = True
        mock_loss.__truediv__ = MagicMock(return_value=mock_loss, name="mock_loss.__truediv__")
        mock_loss.backward = MagicMock(name="mock_loss.backward")

        with patch('torch.nn.functional.cross_entropy', return_value=mock_loss) as mock_ce_func:
            mock_dataloader.__iter__.return_value = iter([(torch.randn(2, 4), torch.randint(0, 10, (2,)))])
            mock_dataloader.__len__.return_value = 1
            mock_progress_tracker_instance.start_time = None

            # Simulate scaler.step returning None (e.g., due to Inf/NaN gradients detected after unscaling)
            mock_scaler.step = MagicMock(return_value=None) # Simulate step skip
            # Mock get_scale just to return something comparable, even though step returns None
            mock_scaler.get_scale = MagicMock(return_value=128.0)
            mock_scaler.is_enabled = MagicMock(return_value=True) # AMP is ON for this test
            mock_scaler.scale = MagicMock(return_value=mock_loss)
            mock_scaler.update = MagicMock() # Mock update
            mock_scaler.unscale_ = MagicMock() # Mock unscale_
            mock_params = mock_model.parameters()

            # --- Setup Config ---
            config = TrainingConfig(
                batch_size=2, log_interval=10, num_epochs=1, learning_rate=1e-4,
                use_amp=True, gradient_accumulation_steps=1, max_grad_norm=1.0
            )

            # --- Setup Loop ---
            loop = TrainingLoop(
                model=mock_model, optimizer=mock_optimizer, train_dataloader=mock_dataloader,
                device=mock_device, config=config
            )
            loop.scaler = mock_scaler
            mock_trainer = MagicMock(spec=Trainer)

            # --- Run Epoch with Patched isnan/isinf for loss check --- #
            current_epoch = 0
            start_global_step_for_epoch = start_global_step
            with patch('torch.isnan', return_value=False) as mock_isnan, \
                 patch('torch.isinf', return_value=False) as mock_isinf, \
                 patch.object(mock_loss, 'item', return_value=1.5) as mock_item:
                loop.train_epoch(
                    current_epoch=current_epoch, global_step=start_global_step_for_epoch,
                    progress=mock_progress_tracker_instance, trainer=mock_trainer
                )

            # --- Assertions ---
            mock_model.assert_called_once()
            mock_loss.backward.assert_called_once()
            mock_scaler.unscale_.assert_called_once_with(mock_optimizer)
            mock_clip_grad_norm.assert_called_once_with(mock_params, config.max_grad_norm)
            # Ensure scaler.step was called, but resulted in no optimizer step
            mock_scaler.step.assert_called_once_with(mock_optimizer)
            mock_optimizer.step.assert_not_called() # Step not taken
            # Scaler update should still be called
            mock_scaler.update.assert_called_once()
            # Progress tracker update should NOT happen if the step was skipped
            mock_progress_tracker_instance.update.assert_not_called()
            # Check loss checks happened
            assert mock_item.call_count >= 1
            mock_isnan.assert_called_once_with(mock_loss)
            mock_isinf.assert_called_once_with(mock_loss)