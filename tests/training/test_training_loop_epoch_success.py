import pytest
import torch
from unittest.mock import MagicMock, patch, call, ANY
from torch.utils.data import DataLoader, TensorDataset # Need these

# Import the class to test
from craft.training.training_loop import TrainingLoop

# Mock ProgressTracker if not available or for isolation
try:
    from craft.training.progress import ProgressTracker
except ImportError:
    # If ProgressTracker can't be imported, create a dummy class for spec=True
    class ProgressTracker: pass

class TestTrainingLoopTrainEpoch:
    """Tests for the TrainingLoop train_epoch method (successful runs)."""

    # --- train_epoch Tests ---
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
        mock_tqdm.return_value = enumerate(mock_dataloader) # Make tqdm return the enumerator

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
        accumulation_steps = 1
        mock_scaler.scale.assert_called_once_with(mock_loss / accumulation_steps)
        mock_optimizer.step.assert_called_once()
        mock_scaler.update.assert_called_once()
        assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
        last_step = start_global_step_for_epoch + len(mock_dataloader)
        mock_progress_tracker_instance.update.assert_called_with(step=last_step, loss=mock_loss_value)
        assert 'loss' in epoch_metrics
        assert epoch_metrics.get('num_steps', -1) == len(mock_dataloader)

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
    def test_train_epoch_amp_enabled(self,
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
        """Test train_epoch with AMP enabled."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.5, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(mock_dataloader)
        mock_scaler.is_enabled = MagicMock(return_value=True) # AMP is ON

        # --- Setup Loop (AMP Enabled) ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            use_amp=True,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler
        # --- Run Epoch ---
        start_global_step = 0
        with patch('torch.amp.autocast') as mock_autocast_inner:
            epoch_metrics = loop.train_epoch(
                current_epoch=0,
                global_step=start_global_step,
                progress=mock_progress_tracker_instance
            )
            # --- Assertions ---
            mock_autocast_inner.assert_called_with(device_type=mock_device.type, enabled=True)
            accumulation_steps = 1
            mock_scaler.scale.assert_called_once_with(mock_loss / accumulation_steps)
            mock_scaler.mock_scaled_loss.backward.assert_called_once()
            mock_scaler.unscale_.assert_called_once_with(mock_optimizer)
            mock_scaler.step.assert_called_once_with(mock_optimizer)
            mock_scaler.update.assert_called_once()
            assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
            mock_progress_tracker_instance.update.assert_called_with(step=start_global_step + 1, loss=mock_loss.item())

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
        mock_loss = torch.tensor(1.5, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(mock_dataloader)
        mock_scaler.is_enabled = MagicMock(return_value=False)

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            scheduler=mock_scheduler,
            use_amp=False,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler
        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance
        )
        # --- Assertions ---
        mock_optimizer.step.assert_called_once()
        mock_scheduler.step.assert_called_once()
        assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
        mock_progress_tracker_instance.update.assert_called_with(step=start_global_step + 1, loss=mock_loss.item())

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

            mock_tqdm.return_value = enumerate(mock_dataloader)
            mock_scaler.is_enabled = MagicMock(return_value=False)
            mock_scaler.scale = MagicMock(return_value=mock_loss)
            mock_params = mock_model.parameters()

            # --- Setup Loop ---
            max_norm = 1.0
            loop = TrainingLoop(
                model=mock_model,
                optimizer=mock_optimizer,
                train_dataloader=mock_dataloader,
                device=mock_device,
                config={},
                use_amp=False,
                gradient_accumulation_steps=1,
                max_grad_norm=max_norm
            )
            loop.scaler = mock_scaler

            # --- Run Epoch with Patched isnan/isinf --- #
            current_epoch = 0
            start_global_step_for_epoch = start_global_step

            with patch('torch.isnan', return_value=MagicMock(any=MagicMock(return_value=False))) as mock_isnan, \
                 patch('torch.isinf', return_value=MagicMock(any=MagicMock(return_value=False))) as mock_isinf, \
                 patch.object(mock_loss, 'item', return_value=1.5) as mock_item:

                epoch_metrics = loop.train_epoch(
                    current_epoch=current_epoch,
                    global_step=start_global_step_for_epoch,
                    progress=mock_progress_tracker_instance # Pass progress
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
            mock_clip_grad_norm.assert_called_once_with(mock_params, max_norm)
            mock_optimizer.step.assert_called_once()
            mock_progress_tracker_instance.update.assert_called_once_with(step=start_global_step + 1, loss=1.5)
            assert 'loss' in epoch_metrics

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_nan_inf_loss(self,
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
        """Test that NaN/Inf loss skips the optimizer step and logs a warning."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(float('nan'), requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(mock_dataloader)
        mock_scaler.is_enabled = MagicMock(return_value=False)

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler
        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance
        )
        # --- Assertions ---
        mock_model.assert_called_once()
        mock_cross_entropy.assert_called_once()
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        mock_optimizer.step.assert_not_called()
        mock_logger_fixture.warning.assert_called_once()
        assert "NaN/Inf loss detected" in mock_logger_fixture.warning.call_args[0][0]
        assert epoch_metrics.get('loss', -1) == 0.0
        assert epoch_metrics.get('num_steps', -1) == 0
        mock_progress_tracker_instance.update.assert_not_called()

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
        assert epoch_metrics.get('num_steps', -1) == expected_batches_to_process

    @patch('torch.amp.autocast')
    @patch('torch.nn.functional.cross_entropy')
    @patch('tqdm.tqdm')
    def test_train_epoch_callbacks(self,
                                   mock_tqdm,
                                   mock_cross_entropy,
                                   mock_autocast,
                                   mock_model,
                                   mock_optimizer,
                                   mock_dataloader,
                                   mock_device,
                                   mock_progress_tracker_instance,
                                   mock_scaler,
                                   mock_logger_fixture,
                                   mock_callback_fixture
                                  ):
        """Test that step begin/end callbacks are called."""
        # --- Setup Mocks ---
        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.5, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(mock_dataloader)
        mock_scaler.is_enabled = MagicMock(return_value=False)
        mock_callback = mock_callback_fixture

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=mock_dataloader,
            device=mock_device,
            config={},
            use_amp=False,
            gradient_accumulation_steps=1,
            callbacks=[mock_callback]
        )
        loop.scaler = mock_scaler
        # --- Run Epoch ---
        start_global_step = 10
        loop.train_epoch(
            current_epoch=0,
            global_step=start_global_step,
            progress=mock_progress_tracker_instance
        )
        # --- Assertions ---
        mock_callback.on_step_begin.assert_called_once_with(start_global_step, logs=ANY)
        mock_callback.on_step_end.assert_called_once_with(start_global_step + 1, logs=ANY)
        assert mock_progress_tracker_instance.update.call_count == len(mock_dataloader)
        mock_progress_tracker_instance.update.assert_called_with(step=start_global_step + 1, loss=mock_loss.item())

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

        mock_autocast.return_value.__enter__.return_value = None
        mock_autocast.return_value.__exit__.return_value = None
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_cross_entropy.return_value = mock_loss
        mock_tqdm.return_value = enumerate(multi_batch_dataloader)

        # --- Setup Loop ---
        loop = TrainingLoop(
            model=mock_model,
            optimizer=mock_optimizer,
            train_dataloader=multi_batch_dataloader,
            device=mock_device,
            config={'max_steps': max_steps_to_run}, # Set max_steps in config
            use_amp=False,
            gradient_accumulation_steps=1
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch ---
        start_global_step = 0
        epoch_metrics = loop.train_epoch(current_epoch=0, global_step=start_global_step, progress=mock_progress_tracker_instance)

        # --- Assertions ---
        assert mock_model.call_count == max_steps_to_run
        assert mock_cross_entropy.call_count == max_steps_to_run
        assert mock_optimizer.step.call_count == max_steps_to_run
        assert mock_progress_tracker_instance.update.call_count == max_steps_to_run
        mock_logger_fixture.info.assert_any_call(f"Reached max_steps ({max_steps_to_run}). Ending epoch early.")
        assert epoch_metrics.get('num_steps') == max_steps_to_run

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
            config={},\
            use_amp=False,\
            gradient_accumulation_steps=accumulation_steps
        )
        loop.scaler = mock_scaler
        loop.scaler.is_enabled = MagicMock(return_value=False)

        # --- Run Epoch ---\
        start_global_step = 0
        epoch_metrics = loop.train_epoch(
            current_epoch=0,\
            global_step=start_global_step,\
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
        # The number of *valid* steps in the epoch (where loss was computed) should be num_batches
        assert epoch_metrics.get('num_steps', -1) == num_batches 