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

            mock_scaler.is_enabled = MagicMock(return_value=False)
            mock_scaler.scale = MagicMock(return_value=mock_loss)
            mock_params = mock_model.parameters()

            # --- Setup Config ---
            config = TrainingConfig(
                batch_size=2,
                log_interval=10,
                num_epochs=1,
                learning_rate=1e-4,
                use_amp=False,
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
            loop.scaler = mock_scaler # Inject mock scaler
            mock_trainer = MagicMock(spec=Trainer) # Add mock trainer

            # --- Run Epoch with Patched isnan/isinf --- #
            current_epoch = 0
            start_global_step_for_epoch = start_global_step

            with patch('torch.isnan', return_value=MagicMock(any=MagicMock(return_value=False))) as mock_isnan, \
                 patch('torch.isinf', return_value=MagicMock(any=MagicMock(return_value=False))) as mock_isinf, \
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
            mock_isnan.assert_called_once_with(mock_loss)
            mock_isinf.assert_called_once_with(mock_loss)
            mock_item.assert_called_once() # Check loss.item() was called
            mock_loss.backward.assert_called_once()
            mock_scaler.unscale_.assert_called_once_with(mock_optimizer)
            mock_clip_grad_norm.assert_called_once_with(mock_params, config.max_grad_norm)
            mock_optimizer.step.assert_called_once()
            mock_progress_tracker_instance.update.assert_called_once()
            # Get the call args and assert specific parts if necessary
            call_args, call_kwargs = mock_progress_tracker_instance.update.call_args
            assert call_kwargs.get('step') == start_global_step + 1
            assert call_kwargs.get('loss') == 1.5
            assert 'learning_rate' in call_kwargs
            assert 'tokens_per_second' in call_kwargs
            assert call_kwargs.get('additional_metrics') is None 