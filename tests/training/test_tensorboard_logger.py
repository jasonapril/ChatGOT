"""
Tests for the TensorBoardLogger callback.
"""
import pytest
from unittest.mock import MagicMock, call, patch
import logging # Required for caplog
import os

from craft.training.callbacks import TensorBoardLogger

# --- Fixtures --- #

@pytest.fixture
def tb_logger_callback(tmp_path): # Use pytest's tmp_path fixture for log_dir
    """Creates a TensorBoardLogger instance with an experiment ID."""
    log_dir = tmp_path / "tb_logs"
    # Add experiment_id for predictable log path generation
    experiment_id = "test_experiment_123"
    return TensorBoardLogger(log_dir=str(log_dir), experiment_id=experiment_id)

# mock_trainer now provided by conftest.py

# --- Test Class --- #

class TestTensorBoardLogger:

    def test_init(self, tb_logger_callback, tmp_path):
        """Test TensorBoardLogger initialization."""
        expected_log_dir_base = str(tmp_path / "tb_logs")
        assert tb_logger_callback.log_dir_base == expected_log_dir_base # Check the base path stored
        assert tb_logger_callback.experiment_id == "test_experiment_123" # Check experiment ID
        assert tb_logger_callback.writer is None
        assert tb_logger_callback.log_dir_absolute is None # log_dir_absolute is set later

    def test_on_train_begin_initializes_writer(self, tb_logger_callback, mock_trainer):
        """Test that SummaryWriter is initialized on train begin."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class, \
             patch('os.makedirs') as mock_makedirs: # Mock makedirs
            # Pass mock_trainer
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            # The log_dir_absolute is now constructed inside on_train_begin using base and experiment_id
            expected_log_dir = os.path.abspath(os.path.join(tb_logger_callback.log_dir_base, tb_logger_callback.experiment_id))
            mock_makedirs.assert_called_once_with(expected_log_dir, exist_ok=True)
            mock_writer_class.assert_called_once_with(log_dir=expected_log_dir)
            # Check that the instance is stored
            assert tb_logger_callback.writer == mock_writer_class.return_value
            assert tb_logger_callback.log_dir_absolute == expected_log_dir # Verify log_dir_absolute is set

    def test_on_train_begin_handles_exception(self, tb_logger_callback, mock_trainer, caplog):
        """Test that an exception during SummaryWriter init is handled."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            mock_writer_class.side_effect = Exception("Initialization failed")
            with caplog.at_level(logging.ERROR):
                 # Pass mock_trainer
                tb_logger_callback.on_train_begin(trainer=mock_trainer)
            assert "Failed to initialize TensorBoard SummaryWriter" in caplog.text
            assert "Initialization failed" in caplog.text # Check exception is logged
            assert tb_logger_callback.writer is None
            # Ensure log_dir_absolute is set before the error, but writer remains None
            expected_log_dir = os.path.abspath(os.path.join(tb_logger_callback.log_dir_base, tb_logger_callback.experiment_id))
            assert tb_logger_callback.log_dir_absolute == expected_log_dir
            # assert tb_logger_callback.log_dir_absolute is None # This was wrong, path is determined before writer init

    def test_on_step_end_logs_metrics(self, tb_logger_callback, mock_trainer):
        """Test logging of step-level metrics."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            # Need to initialize the writer first
            tb_logger_callback.on_train_begin(trainer=mock_trainer) # Pass mock_trainer
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            # The TrainingLoop passes loss and lr in the logs dict
            logs = {'loss': 0.123, 'lr': 0.001, 'other_metric': 5}
            # on_step_end does NOT take trainer
            tb_logger_callback.on_step_end(step=step, logs=logs)

            # Check calls to add_scalar based on current implementation
            calls = [
                call.add_scalar('Loss/train_step', 0.123, step),
                call.add_scalar('LearningRate', 0.001, step),
                # 'other_metric' is not explicitly logged by the callback anymore
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            # Ensure only expected metrics were logged
            assert mock_writer_instance.add_scalar.call_count == 2 # Only loss and lr

    def test_on_step_end_no_logs_or_writer(self, tb_logger_callback, mock_trainer):
        """Test that nothing happens if writer not initialized or logs are None."""
        # Scenario 1: Writer not initialized (No patch needed)
        # on_step_end does NOT take trainer
        tb_logger_callback.on_step_end(step=1, logs={'loss': 0.1})
        # add_scalar should not be available or called (writer is None)
        # No assertion needed, just checking it doesn't crash

        # Scenario 2: Writer initialized, logs is None
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer) # Pass mock_trainer
            mock_writer_instance = mock_writer_class.return_value
             # on_step_end does NOT take trainer
            tb_logger_callback.on_step_end(step=1, logs=None)
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_step_end_missing_keys(self, tb_logger_callback, mock_trainer):
        """Test on_step_end when 'loss' or 'lr' are missing in logs."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer) # Pass mock_trainer
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            # Case 1: Missing 'loss'
            logs1 = {'lr': 0.001, 'another': 10}
            # on_step_end does NOT take trainer
            tb_logger_callback.on_step_end(step=step, logs=logs1)
            mock_writer_instance.add_scalar.assert_called_once_with('LearningRate', 0.001, step)
            mock_writer_instance.reset_mock()

            # Case 2: Missing 'lr'
            logs2 = {'loss': 0.123, 'another': 10}
             # on_step_end does NOT take trainer
            tb_logger_callback.on_step_end(step=step + 1, logs=logs2)
            # Only loss should be logged
            mock_writer_instance.add_scalar.assert_called_once_with('Loss/train_step', 0.123, step + 1)
            mock_writer_instance.reset_mock()

            # Case 3: Both None
            logs3 = {'loss': None, 'lr': None}
             # on_step_end does NOT take trainer
            tb_logger_callback.on_step_end(step=step + 2, logs=logs3)
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_epoch_end_logs_metrics(self, tb_logger_callback, mock_trainer):
        """Test logging of epoch-level metrics."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer) # Initialize writer
            mock_writer_instance = mock_writer_class.return_value

            epoch_arg = 5 # Epoch argument passed to the method (unused by logger)
            trainer_epoch = 5 # Epoch value expected by logger from trainer state
            # Ensure mock_trainer has the state.epoch attribute the logger uses
            mock_trainer.state = MagicMock()
            mock_trainer.state.epoch = trainer_epoch

            # Prepare separate train and val metrics dictionaries
            train_metrics = {'loss': 0.5, 'perplexity': 10.0, 'epoch_time_sec': 120.5}
            val_metrics = {'loss': 0.6, 'perplexity': 12.0} # Example validation metrics
            logs = {} # The logs dict passed to on_epoch_end is currently unused by logger

            # Call with the new signature
            tb_logger_callback.on_epoch_end(
                trainer=mock_trainer,
                epoch=epoch_arg, # Pass the argument, even though logger ignores it
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                logs=logs
            )

            # Check calls based on implementation key format: Epoch/{key}_train|val
            # Assert using the epoch value from trainer.state.epoch
            calls = [
                # Train metrics
                call.add_scalar('Epoch/loss_train', 0.5, trainer_epoch),
                call.add_scalar('Epoch/perplexity_train', 10.0, trainer_epoch),
                call.add_scalar('Epoch/epoch_time_sec_train', 120.5, trainer_epoch),
                # Validation metrics
                call.add_scalar('Epoch/loss_val', 0.6, trainer_epoch),
                call.add_scalar('Epoch/perplexity_val', 12.0, trainer_epoch),
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 5

    def test_on_epoch_end_missing_keys(self, tb_logger_callback, mock_trainer):
        """Test on_epoch_end when various keys are missing/None."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer) # Initialize writer
            mock_writer_instance = mock_writer_class.return_value

            # Initialize mock trainer state
            mock_trainer.state = MagicMock()

            epoch_base = 5

            # Case 1: Missing 'loss' in train_metrics, missing val_metrics entirely
            epoch1 = epoch_base
            mock_trainer.state.epoch = epoch1 # Set trainer state epoch
            train_metrics1 = {'perplexity': 10.0}
            val_metrics1 = {}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch1, train_metrics=train_metrics1, val_metrics=val_metrics1)
            calls1 = [
                call.add_scalar('Epoch/perplexity_train', 10.0, epoch1) # Use correct epoch value
            ]
            mock_writer_instance.assert_has_calls(calls1, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 1
            mock_writer_instance.reset_mock()

            # Case 2: Missing 'perplexity' in val_metrics, train_metrics is None
            epoch2 = epoch_base + 1
            mock_trainer.state.epoch = epoch2 # Set trainer state epoch
            train_metrics2 = None
            val_metrics2 = {'loss': 0.6}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch2, train_metrics=train_metrics2, val_metrics=val_metrics2)
            calls2 = [
                call.add_scalar('Epoch/loss_val', 0.6, epoch2) # Use correct epoch value
            ]
            mock_writer_instance.assert_has_calls(calls2, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 1
            mock_writer_instance.reset_mock()

            # Case 3: Both train_metrics and val_metrics are None
            epoch3 = epoch_base + 2
            mock_trainer.state.epoch = epoch3 # Set trainer state epoch
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch3, train_metrics=None, val_metrics=None)
            mock_writer_instance.add_scalar.assert_not_called()
            mock_writer_instance.reset_mock()

            # Case 4: Keys present but None
            epoch4 = epoch_base + 3
            mock_trainer.state.epoch = epoch4 # Set trainer state epoch
            train_metrics4 = {'loss': None}
            val_metrics4 = {'loss': None}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch4, train_metrics=train_metrics4, val_metrics=val_metrics4)
            mock_writer_instance.add_scalar.assert_not_called() # None values should be skipped
            mock_writer_instance.reset_mock()

    def test_on_train_end_closes_writer(self, tb_logger_callback, mock_trainer):
        """Test that the writer is closed on train end."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer) # Pass mock_trainer
            mock_writer_instance = mock_writer_class.return_value

             # Pass mock_trainer
            tb_logger_callback.on_train_end(trainer=mock_trainer)
            mock_writer_instance.close.assert_called_once()

    def test_on_train_end_no_writer(self, tb_logger_callback, mock_trainer):
        """Test that close is not called if writer wasn't initialized."""
        # Writer is None initially (No patch needed)
         # Pass mock_trainer
        tb_logger_callback.on_train_end(trainer=mock_trainer)
        # No error should occur, and close shouldn't be called on None

    # Add simple tests to ensure other required methods exist
    def test_other_methods_exist(self, tb_logger_callback, mock_trainer):
        """Check that other abstract methods are implemented (even if empty)."""
        assert hasattr(tb_logger_callback, 'on_epoch_begin')
        assert hasattr(tb_logger_callback, 'on_step_begin')
        assert hasattr(tb_logger_callback, 'set_trainer')
        # Call them to ensure no error (set_trainer needs a mock)
        # Pass mock_trainer where needed
        tb_logger_callback.on_epoch_begin(trainer=mock_trainer, epoch=0)
        # on_step_begin does NOT take trainer
        tb_logger_callback.on_step_begin(step=0)
        tb_logger_callback.set_trainer(mock_trainer) # Test set_trainer explicitly 