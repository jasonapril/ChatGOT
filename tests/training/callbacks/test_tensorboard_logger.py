"""
Tests for the TensorBoardLogger callback.
"""
import pytest
from unittest.mock import MagicMock, call, patch, ANY
import logging # Required for caplog
import os
import torch # Keep torch import if needed elsewhere, or remove if not
from craft.training.callbacks.tensorboard import TensorBoardLogger
from hydra.core.hydra_config import HydraConfig # ADDED
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

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

    def test_on_train_begin_initializes_writer(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test that SummaryWriter is initialized on train begin."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Explicitly set trainer and a resolved path before calling on_train_begin
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)

            tb_logger_callback.on_train_begin()
            # After on_train_begin, log_dir_absolute should be set.
            # The exact path depends on mocks (Hydra CWD, os.path.abspath), so we check it was set and used.
            assert tb_logger_callback.resolved_log_dir is not None
            resolved_log_dir = tb_logger_callback.resolved_log_dir

            # Check that the directory was actually created (since we removed the patch)
            # assert os.path.isdir(resolved_log_dir) # REMOVED: Not created when set_log_dir_absolute is used

            # Assert SummaryWriter was called with the path determined by the callback
            mock_writer_class.assert_called_once_with(log_dir=resolved_log_dir)
            assert tb_logger_callback.writer is mock_writer_class.return_value

    def test_on_train_begin_handles_exception(self, tb_logger_callback, mock_trainer, caplog, tmp_path):
        """Test that an exception during SummaryWriter init is handled."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            mock_writer_class.side_effect = Exception("Initialization failed")
            with caplog.at_level(logging.ERROR):
                # Explicitly set trainer and a resolved path before calling on_train_begin
                tb_logger_callback.set_trainer(mock_trainer)
                resolved_path = str(tmp_path / "resolved_logs_exception") # Use tmp_path
                tb_logger_callback.set_log_dir_absolute(resolved_path)

                tb_logger_callback.on_train_begin()
            assert tb_logger_callback.writer is None
            assert "Failed to initialize SummaryWriter" in caplog.text
            assert "Initialization failed" in caplog.text

    def test_on_step_end_logs_metrics(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test logging of step-level metrics."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Explicitly set trainer and a resolved path before calling on_train_begin
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs_step") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)
            tb_logger_callback.on_train_begin() # No trainer arg needed
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            global_step_val = 150 # Example global step
            # Pass metrics dictionary
            metrics_dict = {'loss': 0.123, 'lr': 0.001, 'other_metric': 5}
            tb_logger_callback.on_step_end(step=step, global_step=global_step_val, metrics=metrics_dict)

            calls = [
                call.add_scalar('Loss/train_step', 0.123, global_step_val),
                call.add_scalar('LearningRate/step', 0.001, global_step_val),
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            for c in mock_writer_instance.add_scalar.call_args_list:
                assert c.args[0] != 'Metrics/other_metric'

    def test_on_step_end_no_logs_or_writer(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test that nothing happens if writer not initialized or logs are None."""
        # Scenario 1: Writer not initialized (no on_train_begin call)
        # Pass global_step as it's now required, use metrics instead of logs
        tb_logger_callback.on_step_end(step=1, global_step=10, metrics={'loss': 0.1})

        # Scenario 2: Writer IS initialized, but logs are None
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Explicitly set trainer and a resolved path before calling on_train_begin
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs_step_missing") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)
            tb_logger_callback.on_train_begin() # No trainer arg needed
            mock_writer_instance = mock_writer_class.return_value
            # Pass empty metrics dictionary instead of None
            tb_logger_callback.on_step_end(step=2, global_step=20, metrics={})
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_step_end_missing_keys(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test on_step_end when 'loss' or 'lr' are missing in logs."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Explicitly set trainer and a resolved path before calling on_train_begin
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs_step_missing") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)
            tb_logger_callback.on_train_begin() # No trainer arg needed
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            global_step_val = 150
            # Case 1: Missing 'loss'
            metrics1_dict = {'lr': 0.001, 'another': 10}
            tb_logger_callback.on_step_end(step=step, global_step=global_step_val, metrics=metrics1_dict)

            # Check that add_scalar was called for both lr and the other metric
            expected_calls = [
                call('LearningRate/step', 0.001, global_step=global_step_val),
                call('Train/Another_step', 10, global_step=global_step_val) # Default tag includes Train/
            ]
            # Use assert_has_calls with any_order=True as the order might not be guaranteed
            mock_writer_instance.add_scalar.assert_has_calls(expected_calls, any_order=True)
            # Assert total number of calls
            assert mock_writer_instance.add_scalar.call_count == 2

            mock_writer_instance.add_scalar.reset_mock()

            # Case 2: Missing 'lr'
            metrics2_dict = {'loss': 0.5, 'another': 20}
            tb_logger_callback.on_step_end(step=step + 1, global_step=global_step_val + 1, metrics=metrics2_dict)

            # Check that add_scalar was called for loss, the other metric, AND the LR fallback
            expected_calls_2 = [
                call('Loss/train_step', 0.5, global_step=global_step_val + 1), # Use the actual tag from logs
                call('Train/Another_step', 20, global_step=global_step_val + 1),
                call('LearningRate/step', ANY, global_step=global_step_val + 1) # Check fallback call
            ]
            mock_writer_instance.add_scalar.assert_has_calls(expected_calls_2, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 3 # Expect 3 calls now

            # Case 3: Empty metrics
            mock_writer_instance.add_scalar.reset_mock()
            tb_logger_callback.on_step_end(step=step + 2, global_step=global_step_val + 2, metrics={})
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_epoch_end_logs_metrics(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test logging of epoch-level metrics."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Explicitly set trainer and a resolved path before calling on_train_begin
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs_epoch") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)
            tb_logger_callback.on_train_begin() # No trainer arg needed
            mock_writer_instance = mock_writer_class.return_value

            epoch = 5
            mock_trainer.global_step = 500
            # Combine metrics into a single metrics dictionary
            metrics_dict = {
                "loss": 0.4, # Assume this is train loss if not prefixed
                "perplexity": 8.0,
                "epoch_time_sec": 120,
                "val_loss": 0.3,
                "val_perplexity": 10.5,
                "final_global_step": mock_trainer.global_step # Include final step
            }

            # Call with the unified metrics dictionary
            tb_logger_callback.on_epoch_end(
                epoch=epoch,
                global_step=mock_trainer.global_step, # Pass global_step
                metrics=metrics_dict # Use 'metrics' keyword
            )

            # Expected calls
            expected_calls = [
                call('Train/Loss_epoch', 0.4, global_step=500),
                call('Train/Perplexity_epoch', 8.0, global_step=500),
                # epoch_time_sec IS logged if it's a scalar
                call('Train/Epoch_time_sec_epoch', 120, global_step=500),
                call('Validation/Loss_epoch', 0.3, global_step=500),
                call('Validation/Perplexity_epoch', 10.5, global_step=500),
                # final_global_step IS logged if it's a scalar
                call('Train/Final_global_step_epoch', 500, global_step=500),
                # Use the LR from the mock trainer's optimizer
                call("LR/Learning_Rate_epoch", mock_trainer.optimizer.param_groups[0]['lr'], global_step=500)
            ]
            # Filter out potential None LR if optimizer mock doesn't have param_groups
            expected_calls = [c for c in expected_calls if c[1][1] is not None]

            mock_writer_instance.add_scalar.assert_has_calls(expected_calls, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == len(expected_calls)
            mock_writer_instance.flush.assert_called_once()

    def test_on_epoch_end_missing_keys(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test on_epoch_end when various keys are missing/None."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Explicitly set trainer and a resolved path before calling on_train_begin
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs_epoch_missing") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)
            tb_logger_callback.on_train_begin() # No trainer arg needed
            mock_writer_instance = mock_writer_class.return_value
            mock_trainer.global_step = 600
            epoch = 6

            # Case 1: Metrics are None (represented by empty metrics dict)
            tb_logger_callback.on_epoch_end(epoch=epoch, global_step=mock_trainer.global_step, metrics={}) # Use 'metrics', pass global_step

            # Case 2: Only some metrics present
            metrics2 = {"val_loss": 0.7}
            tb_logger_callback.on_epoch_end(epoch=epoch, global_step=mock_trainer.global_step, metrics=metrics2) # Use 'metrics', pass global_step

            mock_writer_instance.reset_mock()

            # Case 3: Logs contain some valid metrics
            metrics_partial = {'loss': 0.1, "val_accuracy": 0.95, "final_global_step": 600}
            tb_logger_callback.on_epoch_end(epoch=epoch, global_step=mock_trainer.global_step, metrics=metrics_partial) # Use 'metrics', pass global_step

            expected_calls_partial = [
                call('Train/Loss_epoch', 0.1, global_step=600),
                call('Validation/Accuracy_epoch', 0.95, global_step=600),
                call('Train/Final_global_step_epoch', 600, global_step=600), # Expect final_global_step
            ]
            if mock_trainer.optimizer:
                 lr = mock_trainer.optimizer.param_groups[0]['lr']
                 if lr is not None:
                     expected_calls_partial.append(call("LR/Learning_Rate_epoch", lr, global_step=600))

            mock_writer_instance.add_scalar.assert_has_calls(expected_calls_partial, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == len(expected_calls_partial)

    def test_on_train_end_closes_writer(self, tb_logger_callback, mock_trainer, tmp_path):
        """Test that the writer is closed on train end."""
        patch_hydra, patch_exists = self._setup_hydra_mock_patch()
        with patch_hydra as mock_hydra_get, \
             patch_exists as mock_exists, \
             patch('craft.training.callbacks.tensorboard.SummaryWriter', autospec=True) as mock_writer_class:
            # Ensure trainer is set *before* on_train_begin to resolve log dir
            # Also explicitly set resolved path to ensure writer creation
            tb_logger_callback.set_trainer(mock_trainer)
            resolved_path = str(tmp_path / "resolved_logs_close") # Use tmp_path
            tb_logger_callback.set_log_dir_absolute(resolved_path)
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value
            # Pass metrics= instead of logs=
            tb_logger_callback.on_train_end(metrics={}) # Removed trainer=, use metrics=
            mock_writer_instance.close.assert_called_once()

    # Add simple tests to ensure other required methods exist
    def test_other_methods_exist(self, tb_logger_callback, mock_trainer):
        """Test that other unused callback methods exist and are callable."""
        # Set trainer before calling methods that might need it internally
        tb_logger_callback.set_trainer(mock_trainer)
        
        # These should just run without error, matching BaseCallback signatures
        tb_logger_callback.on_init_end()
        tb_logger_callback.on_epoch_begin(epoch=0)
        tb_logger_callback.on_step_begin(step=0)
        tb_logger_callback.on_evaluation_begin()
        tb_logger_callback.on_evaluation_end(metrics={'eval_loss': 0.5})
        tb_logger_callback.on_train_end()
        # Save/Load require state and filename, use dummy values
        mock_state = MagicMock()
        tb_logger_callback.on_save_checkpoint(state=mock_state, filename="dummy.pt")
        tb_logger_callback.on_load_checkpoint(state=mock_state, filename="dummy.pt")
        tb_logger_callback.on_exception(exception=Exception("Test")) 