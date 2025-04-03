"""
Tests for the EarlyStopping callback.
"""
import pytest
import logging
import numpy as np
from unittest.mock import MagicMock, patch

from craft.training.callbacks import EarlyStopping, Callback

# --- Fixtures --- #

# mock_trainer now provided by conftest.py
# Note: conftest version includes state.stopped_training, state.best_metric_value, etc.

@pytest.fixture
def early_stopping_callback():
    """Creates an EarlyStopping callback instance."""
    return EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0.01,
        verbose=False
    )

# Fixture for a basic EarlyStopping instance
@pytest.fixture
def callback(request):
    # Allows tests to override default parameters
    params = getattr(request, "param", {})
    default_params = {
        'monitor': 'eval_loss',
        'patience': 2,
        'mode': 'min'
    }
    default_params.update(params)
    return EarlyStopping(**default_params)

# Mock Trainer fixture (already provided by conftest.py, but needed for type hinting/clarity)
# Let's keep the import from conftest implicit for now, just use mock_trainer directly.

# --- Test Class --- #

class TestEarlyStopping:

    def test_init_min_mode(self, callback):
        """Test initialization sets attributes correctly for min mode."""
        assert callback.monitor == 'eval_loss'
        assert callback.patience == 2
        assert callback.mode == 'min'
        assert callback.delta == 0
        assert callback.wait == 0
        assert callback.stopped_epoch == 0
        assert callback.monitor_op is not None
        assert callback.best == np.inf

    def test_init_max_mode(self):
        """Test initialization with mode='max'."""
        callback = EarlyStopping(monitor='acc', patience=3, mode='max')
        assert callback.mode == 'max'
        assert callback.best == -np.inf

    def test_init_mode_auto_acc(self):
        """Test initialization with mode='auto' for accuracy."""
        callback = EarlyStopping(monitor='val_acc', patience=3, mode='auto')
        assert callback.mode == 'max'
        assert callback.best == -np.inf

    def test_init_mode_auto_loss(self):
        """Test initialization with mode='auto' for loss."""
        callback = EarlyStopping(monitor='loss', patience=3, mode='auto')
        assert callback.mode == 'min'
        assert callback.best == np.inf

    def test_init_invalid_mode(self, caplog):
        """Test warning for invalid mode."""
        with caplog.at_level(logging.WARNING):
            callback = EarlyStopping(monitor='loss', mode='invalid')
        assert "EarlyStopping mode invalid is unknown, fallback to auto mode." in caplog.text
        assert callback.mode == 'min'
        assert callback.best == np.inf

    def test_on_train_begin_resets_state(self, mock_trainer):
        """Test that on_train_begin resets the callback state."""
        callback = EarlyStopping(monitor='eval_loss', patience=2, mode='min', delta=0.1)
        callback.wait = 1
        callback.stopped_epoch = 5
        callback.best = 0.5

        callback.on_train_begin(trainer=mock_trainer)

        assert callback.wait == 0
        assert callback.stopped_epoch == 0
        assert callback.best == np.inf
        assert callback.best_weights is None
        assert callback.best_epoch == 0

        callback_max = EarlyStopping(monitor='val_acc', patience=3, mode='max')
        callback_max.wait = 2
        callback_max.stopped_epoch = 3
        callback_max.best = 0.8

        callback_max.on_train_begin(trainer=mock_trainer)

        assert callback_max.wait == 0
        assert callback_max.stopped_epoch == 0
        assert callback_max.best == -np.inf

    def test_on_epoch_end_monitor_missing(self, callback, mock_trainer, caplog):
        """Test warning when monitored metric is missing from logs."""
        logs = {'other_metric': 0.5}
        with caplog.at_level(logging.WARNING):
            callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs=logs)
        assert f"Early stopping conditioned on metric `{callback.monitor}` which is not available" in caplog.text
        assert callback.wait == 0

    def test_on_epoch_end_improvement_min_mode(self, callback, mock_trainer):
        """Test behavior on improvement in min mode."""
        logs = {'eval_loss': 0.5}
        initial_best = callback.best
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs=logs)
        assert callback.best == 0.5
        assert callback.wait == 0
        assert callback.best_epoch == 0
        assert initial_best > callback.best

    def test_on_epoch_end_no_improvement_min_mode(self, callback, mock_trainer):
        """Test behavior on no improvement in min mode."""
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs={'eval_loss': 0.6})
        assert callback.wait == 0
        assert callback.best == 0.6

        callback.on_epoch_end(trainer=mock_trainer, epoch=1, logs={'eval_loss': 0.65})
        assert callback.wait == 1
        assert callback.best == 0.6

    def test_on_epoch_end_stops_training_min_mode(self, callback, mock_trainer):
        """Test training stops after patience runs out in min mode."""
        callback.patience = 1 # Set patience to 1 for quicker test
        mock_trainer.stop_training = False # Initialize stop_training
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs={'eval_loss': 0.6}) # Best = 0.6
        assert not mock_trainer.stop_training # Check it's still False

        callback.on_epoch_end(trainer=mock_trainer, epoch=1, logs={'eval_loss': 0.7}) # wait = 1
        assert callback.wait == 1
        assert mock_trainer.stop_training # Should stop now
        assert callback.stopped_epoch == 1

    @pytest.mark.parametrize("callback", [{'monitor': 'val_acc', 'mode': 'max', 'patience': 1}], indirect=True)
    def test_on_epoch_end_improvement_max_mode(self, callback, mock_trainer):
        """Test behavior on improvement in max mode."""
        initial_best = callback.best # Should be -inf
        mock_trainer.stop_training = False # Initialize stop_training
        logs = {'val_acc': 0.8}
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs=logs)
        assert callback.best == 0.8
        assert callback.wait == 0
        assert callback.best_epoch == 0
        assert callback.best > initial_best

    @pytest.mark.parametrize("callback", [{'monitor': 'val_acc', 'mode': 'max', 'patience': 1}], indirect=True)
    def test_on_epoch_end_stops_training_max_mode(self, callback, mock_trainer):
        """Test training stops after patience runs out in max mode."""
        mock_trainer.stop_training = False # Initialize stop_training
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs={'val_acc': 0.8}) # Best = 0.8
        assert not mock_trainer.stop_training # Check it's still False

        callback.on_epoch_end(trainer=mock_trainer, epoch=1, logs={'val_acc': 0.7}) # wait = 1
        assert callback.wait == 1
        assert mock_trainer.stop_training # Should stop now
        assert callback.stopped_epoch == 1

    @pytest.mark.parametrize("callback", [{'monitor': 'eval_loss', 'mode': 'min', 'patience': 1, 'delta': 0.1}], indirect=True)
    def test_on_epoch_end_min_delta_improvement(self, callback, mock_trainer):
        """Test min_delta requires significant improvement."""
        mock_trainer.stop_training = False # Initialize stop_training
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs={'eval_loss': 0.6}) # Best = 0.6
        assert callback.wait == 0

        # Improvement less than delta (0.6 -> 0.55 is 0.05 improvement < 0.1 delta)
        callback.on_epoch_end(trainer=mock_trainer, epoch=1, logs={'eval_loss': 0.55})
        assert callback.wait == 1 # Counts as no improvement
        assert callback.best == 0.6 # Best doesn't update
        assert mock_trainer.stop_training # Stops because wait >= patience

        # Reset wait and test significant improvement
        callback.wait = 0
        mock_trainer.stop_training = False
        callback.on_epoch_end(trainer=mock_trainer, epoch=2, logs={'eval_loss': 0.4}) # Improvement of 0.2 > 0.1 delta
        assert callback.wait == 0
        assert callback.best == 0.4
        assert not mock_trainer.stop_training

    @pytest.mark.parametrize("callback", [{'monitor': 'val_acc', 'mode': 'max', 'patience': 1, 'delta': 0.1}], indirect=True)
    def test_on_epoch_end_min_delta_not_improved(self, callback, mock_trainer):
        """Test delta with max mode."""
        callback.on_epoch_end(trainer=mock_trainer, epoch=0, logs={'val_acc': 0.7})
        assert callback.wait == 0

        callback.on_epoch_end(trainer=mock_trainer, epoch=1, logs={'val_acc': 0.75})
        assert callback.wait == 1
        assert callback.best == 0.7
        assert mock_trainer.stop_training

        callback.wait = 0
        mock_trainer.stop_training = False
        callback.on_epoch_end(trainer=mock_trainer, epoch=2, logs={'val_acc': 0.85})
        assert callback.wait == 0
        assert callback.best == 0.85
        assert not mock_trainer.stop_training 