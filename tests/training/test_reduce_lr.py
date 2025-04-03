"""
Tests for the ReduceLROnPlateauOrInstability callback.
"""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, call, patch
import logging # Needed for logger assertion

from craft.training.callbacks import ReduceLROnPlateauOrInstability

# --- Mocks --- #

# mock_trainer now provided by conftest.py

# --- Test Class --- #

class TestReduceLROnPlateauOrInstability:

    @pytest.fixture
    def lr_callback_params(self):
        """Provides default parameters for the callback."""
        return {
            'monitor': 'loss',
            'factor': 0.5,
            'patience': 2,
            'min_lr': 1e-6,
            'threshold': 1.5,
            'cooldown': 1,
            'window_size': 5,
            'verbose': False
        }

    @pytest.fixture
    def lr_callback(self, lr_callback_params):
        """Creates a ReduceLROnPlateauOrInstability callback instance."""
        return ReduceLROnPlateauOrInstability(**lr_callback_params)

    def test_init(self, lr_callback):
        """Test initialization with default parameters."""
        assert lr_callback.monitor == 'loss'
        assert lr_callback.factor == 0.5
        assert lr_callback.patience == 2
        assert lr_callback.min_lr == 1e-6
        assert lr_callback.threshold == 1.5
        assert lr_callback.cooldown == 1
        assert lr_callback.window_size == 5
        assert lr_callback.verbose is False
        assert lr_callback.wait == 0
        assert lr_callback.cooldown_counter == 0
        assert lr_callback.best_loss == float('inf')
        assert lr_callback.recent_losses == []

    def test_set_trainer(self, lr_callback, mock_trainer):
        """Test setting the trainer and initializing state."""
        lr_callback.set_trainer(mock_trainer)
        lr_callback.on_train_begin()  # This will initialize initial_lr
        assert lr_callback.trainer == mock_trainer
        assert lr_callback.optimizer == mock_trainer.optimizer
        assert lr_callback.initial_lr == 0.01  # From mock_trainer fixture

    # --- Edge Cases for Initialization and Setup ---

    def test_init_invalid_factor(self):
        """Test ValueError if factor >= 1.0."""
        with pytest.raises(ValueError, match="Factor should be < 1.0."):
            ReduceLROnPlateauOrInstability(factor=1.0)
        with pytest.raises(ValueError, match="Factor should be < 1.0."):
            ReduceLROnPlateauOrInstability(factor=1.5)

    def test_set_trainer_no_optimizer(self):
        """Test error log if trainer lacks optimizer."""
        callback = ReduceLROnPlateauOrInstability()
        mock_trainer_no_opt = MagicMock(spec=[]) # Trainer with no attributes specified
        with patch.object(callback, 'logger') as mock_logger:
            callback.set_trainer(mock_trainer_no_opt)
            mock_logger.error.assert_called_once_with("Trainer does not have an optimizer attribute")
            assert callback._get_lr() is None

    def test_get_lr_no_optimizer(self):
        """Test _get_lr returns None if optimizer not set."""
        callback = ReduceLROnPlateauOrInstability()
        assert callback._get_lr() is None # Trainer is None initially
        mock_trainer_no_opt = MagicMock(spec=[])
        callback.set_trainer(mock_trainer_no_opt)
        assert callback._get_lr() is None # Trainer has no optimizer

    def test_set_lr_no_optimizer(self):
        """Test _set_lr logs error if optimizer not set."""
        callback = ReduceLROnPlateauOrInstability()
        with patch.object(callback, 'logger') as mock_logger:
            callback._set_lr(0.001)
            mock_logger.error.assert_called_once_with(
                "Optimizer not set in ReduceLROnPlateauOrInstability callback."
            )

    # --- Tests for on_step_end Logic ---

    def test_on_step_end_instability_refined(self, lr_callback, mock_trainer):
        """Test learning rate reduction on instability using refined logic."""
        lr_callback.threshold = 1.5 # Explicitly set threshold for clarity
        lr_callback.patience = 2
        lr_callback.set_trainer(mock_trainer)
        initial_lr = mock_trainer.optimizer.param_groups[0]['lr']

        # Fill window with good loss, establish best_loss
        for step in range(5): lr_callback.on_step_end(step=step, logs={'loss': 0.1})
        assert lr_callback.best_loss == pytest.approx(0.1)

        # Simulate instability spike above threshold * best_loss
        # Step 1: Spike (wait=1)
        lr_callback.on_step_end(step=5, logs={'loss': 0.1 * 1.6}) # 0.16 > 0.1 * 1.5
        assert lr_callback.wait == 1
        # Step 2: Spike again (wait=2 >= patience)
        lr_callback.on_step_end(step=6, logs={'loss': 0.1 * 1.7})
        assert lr_callback.wait == 0 # Reset after reduction
        # Check reduction
        assert mock_trainer.optimizer.param_groups[0]['lr'] == initial_lr * 0.5

    def test_on_step_end_cooldown(self, lr_callback_params, mock_trainer):
        """Test cooldown period after learning rate reduction."""
        # Create callback instance within the test with specific params
        params = {**lr_callback_params, 'patience': 1, 'cooldown': 2, 'verbose': False}
        lr_callback = ReduceLROnPlateauOrInstability(**params)

        lr_callback.set_trainer(mock_trainer)
        initial_lr = mock_trainer.optimizer.param_groups[0]['lr']

        # Fill window and trigger reduction (patience=1)
        for step in range(5): lr_callback.on_step_end(step=step, logs={'loss': 0.1}) # Assuming window_size=5 from params
        lr_callback.on_step_end(step=5, logs={'loss': 0.5}) # Spike triggers wait=1 (threshold=1.5, best_loss=0.1 => 0.15)
        lr_callback.on_step_end(step=6, logs={'loss': 0.6}) # Spike again, wait=2 >= patience=1 -> Reduce

        assert lr_callback.cooldown_counter == 1 # Now using cooldown=2 from init, decremented after step 6
        current_lr = initial_lr * 0.5
        assert mock_trainer.optimizer.param_groups[0]['lr'] == current_lr

        # During cooldown, LR should not change even if conditions met
        lr_callback.on_step_end(step=7, logs={'loss': 0.7}) # Spike (wait would increment, cooldown decrements)
        assert lr_callback.cooldown_counter == 0 # Decremented from 1 to 0 after step 7
        assert lr_callback.wait == 0 # Wait reset during cooldown
        assert mock_trainer.optimizer.param_groups[0]['lr'] == current_lr

        lr_callback.on_step_end(step=8, logs={'loss': 0.8}) # Spike (cooldown decrements to 0)
        assert lr_callback.cooldown_counter == 0 # Stays 0
        assert lr_callback.wait == 0
        assert mock_trainer.optimizer.param_groups[0]['lr'] == current_lr

        # --- Cleaner "After cooldown" logic --- #
        # Cooldown finished after step 8. Asssume recent_losses = [0.7, 0.8]

        # Simulate steps to refill window with a stable loss
        stable_loss = 0.1 # Use a low stable loss for refill
        refill_steps = params['window_size'] - len(lr_callback.recent_losses)
        initial_window = list(lr_callback.recent_losses) # Capture state before refill
        for i in range(refill_steps):
            lr_callback.on_step_end(step=8+i+1, logs={'loss': stable_loss})
        # Window should now be full [0.7, 0.8, 0.1, 0.1, 0.1]
        assert len(lr_callback.recent_losses) == params['window_size'], f"Window size incorrect after refill. Expected {params['window_size']}, got {len(lr_callback.recent_losses)}"

        # Callback should have updated best_loss in the last iteration of the loop
        expected_best_loss = np.mean(lr_callback.recent_losses) # Average of the final window
        assert lr_callback.best_loss == pytest.approx(expected_best_loss), (
               f"best_loss not updated correctly after refill. Expected ~{expected_best_loss}, got {lr_callback.best_loss}"
        )

        # Ensure wait is 0 before introducing new spikes (refill shouldn't trigger instability)
        assert lr_callback.wait == 0, f"Wait counter non-zero after stable refill: {lr_callback.wait}"

        # Now introduce instability again
        # Step 'A' (e.g., step 12 if window_size=5)
        spike_loss_1 = 1.0 # Spike clearly above threshold (expected_best_loss * 1.5)
        lr_callback.on_step_end(step=8+refill_steps+1, logs={'loss': spike_loss_1})
        # Instability check: 1.0 > ~0.36 * 1.5 is true. Wait increments to 1.
        # Patience check: wait (1) >= patience (1) is true. Reduce LR immediately.
        assert lr_callback.wait == 0, f"Wait counter not reset after immediate LR reduction. Value: {lr_callback.wait}"
        assert lr_callback.cooldown_counter == 2, f"Cooldown counter incorrect after reduction. Expected 2 (based on runner), got {lr_callback.cooldown_counter}"
        assert mock_trainer.optimizer.param_groups[0]['lr'] == pytest.approx(current_lr * 0.5), (
               f"LR not reduced correctly at step A. Expected {current_lr * 0.5}, got {mock_trainer.optimizer.param_groups[0]['lr']}"
        )

    def test_on_step_end_loss_improves(self, lr_callback_params, mock_trainer):
        """Test best_loss update when moving average improves."""
        params = {**lr_callback_params, 'window_size': 2, 'verbose': False}
        callback = ReduceLROnPlateauOrInstability(**params)
        callback.set_trainer(mock_trainer)
        callback.best_loss = 1.0
        with patch.object(callback, 'logger') as mock_logger:
            callback.on_step_end(1, logs={'loss': 0.6})
            callback.on_step_end(2, logs={'loss': 0.4})
            moving_avg = np.mean([0.6, 0.4])
            assert callback.best_loss == pytest.approx(moving_avg)
            mock_logger.debug.assert_any_call(f"New best average loss: {moving_avg:.4f}")

    def test_on_step_end_stabilization_log(self, lr_callback_params, mock_trainer):
        """Test log message when loss stabilizes after a spike."""
        params = {**lr_callback_params, 'window_size': 2, 'patience': 3, 'threshold': 1.5, 'verbose': False}
        callback = ReduceLROnPlateauOrInstability(**params)
        callback.set_trainer(mock_trainer)
        callback.best_loss = 1.0
        with patch.object(callback, 'logger') as mock_logger:
            # Step 0: Add baseline loss to partially fill window
            callback.on_step_end(0, logs={'loss': 1.0})
            assert callback.wait == 0 # Window not full yet
            # Step 1: Spike (window is now full: [1.0, 1.6])
            callback.on_step_end(1, logs={'loss': 1.6})
            assert callback.wait == 1 # Spike detected (1.6 > 1.0 * 1.5)
            mock_logger.reset_mock()
            # Step 2: Stabilizes (window: [1.6, 1.4])
            callback.on_step_end(2, logs={'loss': 1.4})
            assert callback.wait == 0 # Stabilized (1.4 <= 1.0 * 1.5)
            mock_logger.debug.assert_called_once_with(f"Loss stabilized (1.4000 <= 1.0000 * 1.50). Resetting wait counter.")

    def test_on_step_end_lr_reduction(self, lr_callback_params, mock_trainer):
        """Test successful LR reduction when patience exceeded."""
        params = {**lr_callback_params, 'window_size': 1, 'patience': 2, 'threshold': 1.1, 'factor': 0.5, 'cooldown': 3, 'verbose': True}
        callback = ReduceLROnPlateauOrInstability(**params)
        initial_lr = 0.01
        mock_trainer.optimizer.param_groups[0]['lr'] = initial_lr
        callback.set_trainer(mock_trainer)
        callback.best_loss = 1.0
        with patch.object(callback, 'logger') as mock_logger:
            callback.on_step_end(1, logs={'loss': 1.2}); assert callback.wait == 1
            callback.on_step_end(2, logs={'loss': 1.3}); # wait=2 >= patience
            assert callback.wait == 0
            assert callback.cooldown_counter == 3
            new_lr = initial_lr * 0.5
            assert mock_trainer.optimizer.param_groups[0]['lr'] == pytest.approx(new_lr)
            mock_logger.warning.assert_called_once_with(f"Learning rate reduced from {initial_lr:.2e} to {new_lr:.2e}")
            assert callback.recent_losses == []
            assert callback.best_loss == float('inf')

    def test_on_step_end_min_lr_hit(self, lr_callback_params, mock_trainer):
        """Test behavior when patience exceeded but LR is at minimum."""
        params = {**lr_callback_params, 'window_size': 1, 'patience': 2, 'threshold': 1.1, 'min_lr': 1e-6, 'verbose': False}
        callback = ReduceLROnPlateauOrInstability(**params)
        initial_lr = 1e-6
        mock_trainer.optimizer.param_groups[0]['lr'] = initial_lr
        callback.set_trainer(mock_trainer)
        callback.best_loss = 1.0
        with patch.object(callback, 'logger') as mock_logger:
            callback.on_step_end(1, logs={'loss': 1.2}); assert callback.wait == 1
            callback.on_step_end(2, logs={'loss': 1.3}); # wait=2 >= patience
            assert callback.wait == 0
            assert callback.cooldown_counter == 0
            assert mock_trainer.optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr)
            mock_logger.info.assert_called_once_with(f"Patience exceeded (0/2), but LR already at minimum ({initial_lr:.2e}).")
            assert callback.recent_losses != []
            assert callback.best_loss != float('inf')

    def test_on_step_end_missing_log_data(self, lr_callback, mock_trainer):
        """Test on_step_end returns early if monitor key or LR is missing."""
        lr_callback.set_trainer(mock_trainer)
        lr_callback.on_train_begin()
        initial_wait = lr_callback.wait
        initial_losses = list(lr_callback.recent_losses)
        lr_callback.on_step_end(step=0, logs={'other_metric': 0.5}); assert lr_callback.wait == initial_wait; assert lr_callback.recent_losses == initial_losses
        lr_callback.on_step_end(step=1, logs={'loss': None}); assert lr_callback.wait == initial_wait; assert lr_callback.recent_losses == initial_losses
        with patch.object(lr_callback, '_get_lr', return_value=None):
             lr_callback.on_step_end(step=2, logs={'loss': 0.5}); assert lr_callback.wait == initial_wait; assert lr_callback.recent_losses == initial_losses

    # --- Tests for other Callback methods --- #

    def test_on_train_begin_resets_state(self, lr_callback_params, mock_trainer):
        """Test on_train_begin resets internal state."""
        callback = ReduceLROnPlateauOrInstability(**lr_callback_params)
        callback.set_trainer(mock_trainer)
        callback.wait = 5; callback.cooldown_counter = 2; callback.best_loss = 0.5
        callback.recent_losses = [0.5, 0.6]; callback.initial_lr = 0.0
        callback.on_train_begin()
        assert callback.wait == 0
        assert callback.cooldown_counter == 0
        assert callback.best_loss == float('inf')
        assert callback.recent_losses == []
        assert callback.initial_lr == mock_trainer.optimizer.param_groups[0]['lr']

    def test_empty_methods_exist(self, lr_callback):
        """Check that other abstract methods are implemented (even if empty)."""
        assert hasattr(lr_callback, 'on_train_end')
        assert hasattr(lr_callback, 'on_epoch_begin')
        assert hasattr(lr_callback, 'on_epoch_end')
        assert hasattr(lr_callback, 'on_step_begin')
        # Call them to ensure no error
        lr_callback.on_train_end()
        lr_callback.on_epoch_begin(epoch=0)
        lr_callback.on_epoch_end(epoch=0)
        lr_callback.on_step_begin(step=0) 