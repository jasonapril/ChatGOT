"""
Tests for the TensorBoardLogger callback.
"""
import pytest
from unittest.mock import MagicMock, call, patch
import logging # Required for caplog
import os
import torch # Keep torch import if needed elsewhere, or remove if not
from craft.training.callbacks.tensorboard import TensorBoardLogger
from hydra.core.hydra_config import HydraConfig # ADDED

from craft.training.callbacks import TensorBoardLogger

# --- Fixtures --- #

@pytest.fixture
def tb_logger_callback(tmp_path): # Use pytest's tmp_path fixture for log_dir
    """Creates a TensorBoardLogger instance with an experiment ID."""
    log_dir = tmp_path / "tb_logs"
    # experiment_id is not an argument, use 'comment' for similar effect
    comment = "_test_experiment_123"
    return TensorBoardLogger(log_dir=str(log_dir), comment=comment)

# mock_trainer now provided by conftest.py

# --- Test Class --- #

class TestTensorBoardLogger:

    # Helper to set up HydraConfig mock
    def _setup_hydra_mock_patch(self):
        mock_hydra_conf = MagicMock()
        mock_hydra_conf.hydra.run.dir = 'mock/output/dir/run'
        patch_hydra = patch('hydra.core.hydra_config.HydraConfig.get', return_value=mock_hydra_conf)
        patch_exists = patch('craft.training.callbacks.tensorboard.os.path.exists', return_value=False)
        return patch_hydra, patch_exists

    def test_init(self, tb_logger_callback, tmp_path):
        """Test TensorBoardLogger initialization."""
        expected_log_dir_base = str(tmp_path / "tb_logs")
        assert tb_logger_callback.log_dir_config == expected_log_dir_base # CORRECTED: Check the original configured path
        # experiment_id is removed, check comment instead
        assert tb_logger_callback.comment == "_test_experiment_123"
        assert tb_logger_callback.writer is None
        assert tb_logger_callback.resolved_log_dir is None # resolved_log_dir is set later in _initialize_writer

    def test_on_train_begin_initializes_writer(self, tb_logger_callback, mock_trainer):
        """Test that SummaryWriter is initialized on train begin."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            # After on_train_begin, log_dir_absolute should be set.
            # The exact path depends on mocks (Hydra CWD, os.path.abspath), so we check it was set and used.
            assert tb_logger_callback.resolved_log_dir is not None
            resolved_log_dir = tb_logger_callback.resolved_log_dir

            # Check that the directory was actually created (since we removed the patch)
            assert os.path.isdir(resolved_log_dir)

            # Assert SummaryWriter was called with the path determined by the callback
            mock_writer_class.assert_called_once_with(log_dir=resolved_log_dir)
            assert tb_logger_callback.writer is mock_writer_class.return_value

    def test_on_train_begin_handles_exception(self, tb_logger_callback, mock_trainer, caplog):
        """Test that an exception during SummaryWriter init is handled."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            mock_writer_class.side_effect = Exception("Initialization failed")
            with caplog.at_level(logging.ERROR):
                tb_logger_callback.on_train_begin(trainer=mock_trainer)
            assert tb_logger_callback.writer is None
            assert "Failed to initialize SummaryWriter" in caplog.text
            assert "Initialization failed" in caplog.text

    def test_on_step_end_logs_metrics(self, tb_logger_callback, mock_trainer):
        """Test logging of step-level metrics."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            logs = {'loss': 0.123, 'lr': 0.001, 'other_metric': 5}
            tb_logger_callback.on_step_end(step=step, logs=logs)

            calls = [
                call.add_scalar('Loss/train', 0.123, step),
                call.add_scalar('LearningRate', 0.001, step),
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            for c in mock_writer_instance.add_scalar.call_args_list:
                assert c.args[0] != 'Metrics/other_metric'

    def test_on_step_end_no_logs_or_writer(self, tb_logger_callback, mock_trainer):
        """Test that nothing happens if writer not initialized or logs are None."""
        # Scenario 1: No changes needed
        tb_logger_callback.on_step_end(step=1, logs={'loss': 0.1})

        # Scenario 2: Apply patches individually
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            mock_writer_instance = mock_writer_class.return_value
            tb_logger_callback.on_step_end(step=2, logs=None)
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_step_end_missing_keys(self, tb_logger_callback, mock_trainer):
        """Test on_step_end when 'loss' or 'lr' are missing in logs."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            # Case 1: Missing 'loss'
            logs1 = {'lr': 0.001, 'another': 10}
            tb_logger_callback.on_step_end(step=step, logs=logs1)
            mock_writer_instance.add_scalar.assert_called_once_with('LearningRate', 0.001, step)
            mock_writer_instance.reset_mock()

            # Case 2: Missing 'lr'
            logs2 = {'loss': 0.123, 'another': 10}
            tb_logger_callback.on_step_end(step=step + 1, logs=logs2)
            mock_writer_instance.add_scalar.assert_called_once_with('Loss/train', 0.123, step + 1)

    def test_on_epoch_end_logs_metrics(self, tb_logger_callback, mock_trainer):
        """Test logging of epoch-level metrics."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            mock_writer_instance = mock_writer_class.return_value

            epoch = 5
            mock_trainer.global_step = 500
            train_metrics = {'loss': 0.4, 'perplexity': 8.0, 'epoch_time_sec': 120}
            val_metrics = {'loss': 0.3, 'perplexity': 10.5}
            logs = {'ignored': 1}

            tb_logger_callback.on_epoch_end(
                trainer=mock_trainer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                logs=logs
            )

            expected_calls = [
                call('Loss/validation', 0.3, 500),
                call('Metrics/val_perplexity', 10.5, 500),
                call('Metrics/train_perplexity', 8.0, 500),
            ]
            mock_writer_instance.add_scalar.assert_has_calls(expected_calls, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == len(expected_calls)
            for call_args in mock_writer_instance.add_scalar.call_args_list:
                assert call_args[0][0] != 'Loss/train'
                assert 'time' not in call_args[0][0]

    def test_on_epoch_end_missing_keys(self, tb_logger_callback, mock_trainer):
        """Test on_epoch_end when various keys are missing/None."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            mock_writer_instance = mock_writer_class.return_value
            mock_trainer.global_step = 600
            epoch = 6

            # Case 1: Metrics are None
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics=None, val_metrics=None, logs=None)
            mock_writer_instance.add_scalar.assert_not_called()
            mock_writer_instance.reset_mock()

            # Case 2: Metrics are empty dicts
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics={}, val_metrics={}, logs=None)
            mock_writer_instance.add_scalar.assert_not_called()
            mock_writer_instance.reset_mock()

            # Case 3: Only train loss and time (should be ignored)
            train_metrics_3 = {'loss': 0.1, 'epoch_time_sec': 100}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics=train_metrics_3, val_metrics={})
            mock_writer_instance.add_scalar.assert_not_called() 
            mock_writer_instance.reset_mock()

            # Case 4: Only val_loss
            val_metrics_4 = {'loss': 0.2}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics={}, val_metrics=val_metrics_4)
            mock_writer_instance.add_scalar.assert_called_once_with('Loss/validation', 0.2, 600)
            mock_writer_instance.reset_mock()

            # Case 5: Only other train metric
            train_metrics_5 = {'custom_metric': 99}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics=train_metrics_5, val_metrics={})
            mock_writer_instance.add_scalar.assert_called_once_with('Metrics/train_custom_metric', 99, 600)
            mock_writer_instance.reset_mock()

            # Case 6: Only other val metric
            val_metrics_6 = {'custom_val_metric': 88}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics={}, val_metrics=val_metrics_6)
            mock_writer_instance.add_scalar.assert_called_once_with('Metrics/val_custom_val_metric', 88, 600)
            mock_writer_instance.reset_mock()

            # Case 7: Mix with None values
            train_metrics_7 = {'loss': 0.1, 'acc': None}
            val_metrics_7 = {'loss': None, 'f1': 0.7}
            tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics=train_metrics_7, val_metrics=val_metrics_7)
            # Should only log val_f1
            mock_writer_instance.add_scalar.assert_called_once_with('Metrics/val_f1', 0.7, 600)
            mock_writer_instance.reset_mock()

            # Case 8: Global step missing on trainer (should warn and not log)
            mock_trainer.global_step = None
            with patch.object(tb_logger_callback, 'logger') as mock_cb_logger:
                tb_logger_callback.on_epoch_end(trainer=mock_trainer, epoch=epoch, train_metrics={'loss': 0.1}, val_metrics={'loss': 0.2})
                mock_cb_logger.warning.assert_called_once_with("Cannot log epoch metrics: trainer.global_step not found.")
                mock_writer_instance.add_scalar.assert_not_called()

    def test_on_train_end_closes_writer(self, tb_logger_callback, mock_trainer):
        """Test that the writer is closed on train end."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin(trainer=mock_trainer)
            mock_writer_instance = mock_writer_class.return_value
            tb_logger_callback.on_train_end(trainer=mock_trainer, logs={})
            mock_writer_instance.close.assert_called_once()
            assert tb_logger_callback.writer is None

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