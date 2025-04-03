"""
Tests for the TensorBoardLogger callback.
"""
import pytest
from unittest.mock import MagicMock, call, patch
import logging # Required for caplog

from craft.training.callbacks import TensorBoardLogger

# --- Fixtures --- #

@pytest.fixture
def tb_logger_callback(tmp_path): # Use pytest's tmp_path fixture for log_dir
    """Creates a TensorBoardLogger instance."""
    log_dir = tmp_path / "tb_logs"
    return TensorBoardLogger(log_dir=str(log_dir))

# mock_trainer now provided by conftest.py

# --- Test Class --- #

class TestTensorBoardLogger:

    def test_init(self, tb_logger_callback, tmp_path):
        """Test TensorBoardLogger initialization."""
        expected_log_dir = str(tmp_path / "tb_logs")
        assert tb_logger_callback.log_dir == expected_log_dir
        assert tb_logger_callback.writer is None

    def test_on_train_begin_initializes_writer(self, tb_logger_callback):
        """Test that SummaryWriter is initialized on train begin."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_class.assert_called_once_with(tb_logger_callback.log_dir)
            # Check that the instance is stored
            assert tb_logger_callback.writer == mock_writer_class.return_value

    def test_on_train_begin_handles_exception(self, tb_logger_callback, caplog):
        """Test that an exception during SummaryWriter init is handled."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            mock_writer_class.side_effect = Exception("Initialization failed")
            with caplog.at_level(logging.ERROR):
                tb_logger_callback.on_train_begin()
            assert "Failed to initialize TensorBoard SummaryWriter" in caplog.text
            assert "Initialization failed" in caplog.text # Check exception is logged
            assert tb_logger_callback.writer is None

    def test_on_step_end_logs_metrics(self, tb_logger_callback):
        """Test logging of step-level metrics."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            # Need to initialize the writer first
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            logs = {'loss': 0.123, 'lr': 0.001, 'other_metric': 5}
            tb_logger_callback.on_step_end(step=step, logs=logs)

            # Check calls to add_scalar
            calls = [
                call.add_scalar('Loss/train_step', 0.123, step),
                call.add_scalar('LearningRate/step', 0.001, step)
                # 'other_metric' is not explicitly logged, so no call expected
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            # Ensure only expected metrics were logged
            assert mock_writer_instance.add_scalar.call_count == 2

    def test_on_step_end_no_logs_or_writer(self, tb_logger_callback):
        """Test that nothing happens if writer not initialized or logs are None."""
        # Scenario 1: Writer not initialized (No patch needed)
        tb_logger_callback.on_step_end(step=1, logs={'loss': 0.1})
        # add_scalar should not be available or called (writer is None)

        # Scenario 2: Writer initialized, logs is None
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value
            tb_logger_callback.on_step_end(step=1, logs=None)
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_step_end_missing_keys(self, tb_logger_callback):
        """Test on_step_end when 'loss' or 'lr' are missing in logs."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            # Case 1: Missing 'loss'
            logs1 = {'lr': 0.001}
            tb_logger_callback.on_step_end(step=step, logs=logs1)
            mock_writer_instance.add_scalar.assert_called_once_with('LearningRate/step', 0.001, step)
            mock_writer_instance.reset_mock()

            # Case 2: Missing 'lr'
            logs2 = {'loss': 0.123}
            tb_logger_callback.on_step_end(step=step + 1, logs=logs2)
            mock_writer_instance.add_scalar.assert_called_once_with('Loss/train_step', 0.123, step + 1)
            mock_writer_instance.reset_mock()

            # Case 3: Both None
            logs3 = {'loss': None, 'lr': None}
            tb_logger_callback.on_step_end(step=step + 2, logs=logs3)
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_epoch_end_logs_metrics(self, tb_logger_callback, mock_trainer):
        """Test logging of epoch-level metrics, including LR from trainer."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            # Note: set_trainer is needed only to test that LR *could* be accessed,
            # but the current implementation doesn't automatically log it on epoch_end.
            tb_logger_callback.set_trainer(mock_trainer)
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            epoch = 5
            # Learning rate is not automatically logged on epoch end, only loss/val_loss
            logs = {'loss': 0.5, 'val_loss': 0.6, 'perplexity': 10.0}
            tb_logger_callback.on_epoch_end(epoch=epoch, logs=logs)

            # expected_lr = mock_trainer.optimizer.param_groups[0]['lr'] # Not logged here
            calls = [
                call.add_scalar('Loss/train_epoch', 0.5, epoch + 1),
                call.add_scalar('Loss/validation_epoch', 0.6, epoch + 1)
                # LR is not logged from epoch_end, even if in logs
                # 'perplexity' is not explicitly logged
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 2 # Expect 2 calls now

    def test_on_epoch_end_missing_keys(self, tb_logger_callback, mock_trainer):
        """Test on_epoch_end when various keys are missing/None."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            # No need to set_trainer here as we are testing missing keys / no auto LR logging
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            epoch = 5
            # expected_lr = mock_trainer.optimizer.param_groups[0]['lr'] # Not used

            # Case 1: Missing 'val_loss'
            logs1 = {'loss': 0.5}
            tb_logger_callback.on_epoch_end(epoch=epoch, logs=logs1)
            calls1 = [
                call.add_scalar('Loss/train_epoch', 0.5, epoch + 1)
                # No LR call expected
            ]
            mock_writer_instance.assert_has_calls(calls1, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 1
            mock_writer_instance.reset_mock()

            # Case 2: Missing 'loss' (train loss)
            logs2 = {'val_loss': 0.6}
            tb_logger_callback.on_epoch_end(epoch=epoch + 1, logs=logs2)
            calls2 = [
                call.add_scalar('Loss/validation_epoch', 0.6, epoch + 2)
                # No LR call expected
            ]
            mock_writer_instance.assert_has_calls(calls2, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 1
            mock_writer_instance.reset_mock()

            # Case 3: logs is None
            tb_logger_callback.on_epoch_end(epoch=epoch + 2, logs=None)
            # No LR call expected
            mock_writer_instance.add_scalar.assert_not_called()
            mock_writer_instance.reset_mock()

            # Case 4: Keys present but None
            logs4 = {'loss': None, 'val_loss': None}
            tb_logger_callback.on_epoch_end(epoch=epoch + 3, logs=logs4)
            # No LR call expected
            mock_writer_instance.add_scalar.assert_not_called()
            mock_writer_instance.reset_mock()

            # Case 5: No trainer set (LR should not be logged even if present in logs)
            # This case implicitly tested by not calling set_trainer initially.
            # Let's add an explicit log with LR to show it's ignored without trainer
            tb_logger_callback.trainer = None # Ensure trainer is None
            logs5 = {'loss': 0.7, 'val_loss': 0.8, 'lr': 0.0001}
            tb_logger_callback.on_epoch_end(epoch=epoch + 4, logs=logs5)
            calls5 = [
                call.add_scalar('Loss/train_epoch', 0.7, epoch + 5),
                call.add_scalar('Loss/validation_epoch', 0.8, epoch + 5)
                # No LR call expected
            ]
            mock_writer_instance.assert_has_calls(calls5, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 2

    def test_on_train_end_closes_writer(self, tb_logger_callback):
        """Test that the writer is closed on train end."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            tb_logger_callback.on_train_end()
            mock_writer_instance.close.assert_called_once()

    def test_on_train_end_no_writer(self, tb_logger_callback):
        """Test that close is not called if writer wasn't initialized."""
        # Writer is None initially (No patch needed)
        tb_logger_callback.on_train_end()
        # No error should occur, and close shouldn't be called on None

    # Add simple tests to ensure other required methods exist
    def test_other_methods_exist(self, tb_logger_callback):
        """Check that other abstract methods are implemented (even if empty)."""
        assert hasattr(tb_logger_callback, 'on_epoch_begin')
        assert hasattr(tb_logger_callback, 'on_step_begin')
        assert hasattr(tb_logger_callback, 'set_trainer')
        # Call them to ensure no error (set_trainer needs a mock)
        tb_logger_callback.on_epoch_begin(epoch=0)
        tb_logger_callback.on_step_begin(step=0)
        tb_logger_callback.set_trainer(MagicMock()) 