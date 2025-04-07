import pytest
import time
import logging
import numpy as np
from unittest.mock import MagicMock, patch, ANY

from craft.training.progress import ProgressTracker

TOTAL_STEPS = 100
LOG_INTERVAL = 10

@pytest.fixture
def mock_logger():
    """Fixture for patching the logger used by ProgressTracker."""
    with patch('craft.training.progress.logging.getLogger') as mock_get_logger:
        logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = logger_instance
        yield logger_instance

@pytest.fixture
def mock_tqdm_bar():
    """Fixture for a mock tqdm progress bar instance."""
    bar = MagicMock()
    bar.update = MagicMock()
    bar.set_postfix = MagicMock()
    bar.close = MagicMock()
    return bar

@pytest.fixture
def patch_tqdm_cls(mock_tqdm_bar):
    """Fixture for patching the tqdm class."""
    with patch('craft.training.progress.tqdm') as mock_tqdm_class:
        mock_tqdm_class.return_value = mock_tqdm_bar
        yield mock_tqdm_class

# --- Test Class ---

class TestProgressTracker:

    def test_init_success(self, mock_logger, patch_tqdm_cls, mock_tqdm_bar):
        """Test successful initialization including tqdm setup."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS, log_interval=LOG_INTERVAL)
        patch_tqdm_cls.assert_called_once_with(
            total=TOTAL_STEPS,
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0,
            initial=0
        )
        assert tracker.progress_bar == mock_tqdm_bar
        assert tracker.logger == mock_logger
        assert tracker.total_steps == TOTAL_STEPS
        assert tracker.log_interval == LOG_INTERVAL
        assert tracker.start_time is None

    def test_init_tqdm_fails(self, mock_logger):
        """Test initialization when tqdm fails."""
        error_message = "tqdm import failed"
        with patch('craft.training.progress.tqdm', side_effect=Exception(error_message)) as mock_tqdm_class:
            tracker = ProgressTracker(total_steps=TOTAL_STEPS)

        mock_tqdm_class.assert_called_once()
        mock_logger.warning.assert_called_once()
        assert "Failed to initialize tqdm progress bar" in mock_logger.warning.call_args[0][0]
        assert error_message in mock_logger.warning.call_args[0][0]
        assert tracker.progress_bar is None # Lines 55-57 covered

    def test_start(self, patch_tqdm_cls):
        """Test the start method."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS)
        assert tracker.start_time is None
        tracker.start()
        assert tracker.start_time is not None
        assert tracker.last_log_time == tracker.start_time

    def test_update_with_tqdm(self, mock_logger, patch_tqdm_cls, mock_tqdm_bar):
        """Test the update method when tqdm is available."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS, log_interval=LOG_INTERVAL)
        tracker.start()
        time.sleep(0.01) # Ensure a small time delta

        step = 1
        loss = 1.2345
        lr = 1e-4
        tps = 5000.0
        add_metrics = {'extra': 'value'}

        tracker.update(step=step, loss=loss, learning_rate=lr, tokens_per_second=tps, additional_metrics=add_metrics)

        assert len(tracker.step_times) == 1
        assert tracker.step_times[0] > 0
        assert tracker.losses == [loss]
        assert tracker.learning_rates == [lr]
        assert tracker.tokens_per_second == [tps]

        expected_postfix = {
            'loss': f'{loss:.4f}',
            'lr': f'{lr:.2e}',
            'T/s': f'{tps:.0f}',
            'extra': 'value' # Additional metrics included
        }
        mock_tqdm_bar.set_postfix.assert_called_once_with(expected_postfix)
        mock_tqdm_bar.update.assert_called_once_with(1) # Line 89 covered

        # Check no logging before interval
        mock_logger.info.assert_not_called()

    def test_update_without_tqdm(self, mock_logger):
        """Test the update method when tqdm initialization failed."""
        with patch('craft.training.progress.tqdm', side_effect=Exception("tqdm failed")):
            tracker = ProgressTracker(total_steps=TOTAL_STEPS, log_interval=LOG_INTERVAL)
        tracker.start()
        time.sleep(0.01)

        step = 1
        loss = 1.23
        tracker.update(step=step, loss=loss)

        assert len(tracker.step_times) == 1
        assert tracker.losses == [loss]
        # Assert that tqdm methods were NOT called
        assert tracker.progress_bar is None
        # No calls expected as progress_bar is None (Covers if check on line 85)
        mock_logger.info.assert_not_called() # Check no logging before interval

    def test_update_triggers_log(self, mock_logger, patch_tqdm_cls):
        """Test that update calls _log_metrics at the specified interval."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS, log_interval=LOG_INTERVAL)
        tracker.start()

        # Mock _log_metrics to check if it's called
        with patch.object(tracker, '_log_metrics') as mock_log_method:
            for step in range(1, LOG_INTERVAL * 2 + 1):
                tracker.update(step=step, loss=1.0)

        assert mock_log_method.call_count == 2
        # Called at step=LOG_INTERVAL and step=LOG_INTERVAL*2
        mock_log_method.assert_any_call(LOG_INTERVAL, 1.0, None, None, None) # Line 93 covered
        mock_log_method.assert_any_call(LOG_INTERVAL * 2, 1.0, None, None, None)

    def test_log_metrics_format(self, mock_logger, patch_tqdm_cls):
        """Test the format and content of the _log_metrics message."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS, log_interval=LOG_INTERVAL)
        tracker.start()
        time.sleep(0.01)

        step = LOG_INTERVAL
        loss = 1.9876
        lr = 5e-5
        tps = 12345.0
        add_metrics = {'accuracy': 0.75}

        # Manually call _log_metrics (update normally calls this)
        tracker._log_metrics(step, loss, lr, tps, add_metrics)

        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]

        assert f"Step: {step}/{TOTAL_STEPS}" in log_msg
        assert f"Loss: {loss:.4f}" in log_msg
        assert f"LR: {lr:.2e}" in log_msg
        assert f"T/s: {tps:.0f}" in log_msg
        assert "Time:" in log_msg
        assert "Step Time:" in log_msg
        assert "accuracy: 0.75" in log_msg

    def test_log_metrics_optional_none(self, mock_logger, patch_tqdm_cls):
        """Test _log_metrics when optional values are None."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS, log_interval=LOG_INTERVAL)
        tracker.start()
        step = LOG_INTERVAL
        loss = 0.5

        # Call with None for optional args
        tracker._log_metrics(step, loss, None, None, None)

        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]

        assert f"Step: {step}/{TOTAL_STEPS}" in log_msg
        assert f"Loss: {loss:.4f}" in log_msg
        assert "LR:" not in log_msg
        assert "T/s:" not in log_msg
        assert "Time:" in log_msg
        assert "Step Time:" in log_msg # Should still be present even if avg is 0

    def test_get_summary(self, patch_tqdm_cls):
        """Test the get_summary method."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS)
        tracker.start()

        # Simulate some updates
        losses = [1.0, 0.8, 0.6]
        lrs = [1e-3, 9e-4, 8e-4]
        tps_vals = [1000, 1200, 1100]
        step_times = [0.1, 0.12, 0.11] # Mocked step times
        tracker.losses = losses
        tracker.learning_rates = lrs
        tracker.tokens_per_second = tps_vals
        tracker.step_times = step_times

        # Fake end time
        start_time = tracker.start_time
        fake_end_time = start_time + 0.5
        with patch('time.time', return_value=fake_end_time):
            summary = tracker.get_summary()

        assert summary['total_steps'] == TOTAL_STEPS
        assert np.isclose(summary['total_time'], 0.5)
        assert np.isclose(summary['avg_loss'], np.mean(losses))
        assert np.isclose(summary['avg_learning_rate'], np.mean(lrs))
        assert np.isclose(summary['avg_tokens_per_second'], np.mean(tps_vals))
        assert np.isclose(summary['avg_step_time'], np.mean(step_times))

    def test_get_summary_no_updates(self, patch_tqdm_cls):
        """Test get_summary when no updates have occurred."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS)
        tracker.start()
        start_time = tracker.start_time
        fake_end_time = start_time + 1.0
        with patch('time.time', return_value=fake_end_time):
             summary = tracker.get_summary() # Line 130 covered

        assert summary['total_steps'] == TOTAL_STEPS
        assert np.isclose(summary['total_time'], 1.0)
        assert summary['avg_loss'] == 0
        assert summary['avg_learning_rate'] == 0
        assert summary['avg_tokens_per_second'] == 0
        assert summary['avg_step_time'] == 0

    def test_close_with_tqdm(self, patch_tqdm_cls, mock_tqdm_bar):
        """Test the close method when tqdm was initialized."""
        tracker = ProgressTracker(total_steps=TOTAL_STEPS)
        assert tracker.progress_bar is not None # Make sure it was created
        tracker.close()
        mock_tqdm_bar.close.assert_called_once() # Line 141 covered
        assert tracker.progress_bar is None # Line 142 covered

    def test_close_without_tqdm(self):
        """Test the close method when tqdm failed to initialize."""
        with patch('craft.training.progress.tqdm', side_effect=Exception("tqdm failed")):
            tracker = ProgressTracker(total_steps=TOTAL_STEPS)

        assert tracker.progress_bar is None
        # Closing should not raise an error if progress_bar is None
        try:
            tracker.close() # Line 143 covered (None check)
        except Exception as e:
            pytest.fail(f"tracker.close() raised an unexpected exception: {e}")
        assert tracker.progress_bar is None

    # Test for close will be added next 