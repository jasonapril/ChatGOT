import pytest
from unittest.mock import MagicMock, patch, call, ANY
import time
import logging
import numpy as np
from tqdm import tqdm
import types # Needed for SimpleNamespace potentially if mocking loaded state

# Assuming ProgressTracker is in src/craft/training/progress.py
from craft.training.progress import ProgressTracker


@pytest.fixture
def mock_tqdm():
    # Fixture to mock the tqdm class
    with patch('craft.training.progress.tqdm', spec=tqdm) as mock:
        # Mock the instance returned by tqdm()
        mock_instance = MagicMock(spec=tqdm)
        mock_instance.n = 0 # Initialize internal counter for testing updates
        mock_instance.postfix = {} # Initialize postfix
        mock.return_value = mock_instance
        yield mock, mock_instance # Return both the class mock and instance mock

# FIX: Remove mock_logger_get fixture, use caplog directly
# @pytest.fixture
# def mock_logger_get(): # Rename to avoid conflict with potentially used 'mock_logger' var name
#     with patch('logging.getLogger') as mock_get_logger:
#         logger_instance = MagicMock(spec=logging.Logger)
#         # Configure the mock getLogger to return our instance when called with the specific name
#         mock_get_logger.return_value = logger_instance
#         yield logger_instance


class TestProgressTracker:

    def test_init_success(self, mock_tqdm, caplog): # Use caplog fixture
        """Test successful initialization with tqdm enabled."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        total_steps = 100
        desc = "TestRun"
        log_interval = 5

        # Check logs during init if needed
        with caplog.at_level(logging.INFO):
            tracker = ProgressTracker(total_steps=total_steps, desc=desc, log_interval=log_interval)

        assert tracker.total_steps == total_steps
        assert tracker.desc == desc
        assert tracker.log_interval == log_interval
        assert tracker.disable_progress_bar is False
        assert tracker.current_step == 0
        assert tracker.start_time is None
        # Check if tqdm was called correctly
        mock_tqdm_class.assert_called_once_with(
            total=total_steps,
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            disable=False # disable should be False
        )
        # Check internal state
        assert tracker._pbar is mock_tqdm_instance # Check internal _pbar reference
        assert tracker.pbar is mock_tqdm_instance # Check property getter
        # Check logger was obtained (via side effect if needed, but not necessary for func)

    def test_init_disabled_progress_bar(self, mock_tqdm, caplog): # Use caplog
        """Test initialization with progress bar disabled."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        with caplog.at_level(logging.INFO):
            tracker = ProgressTracker(total_steps=100, disable_progress_bar=True)

        assert tracker.disable_progress_bar is True
        mock_tqdm_class.assert_not_called() # tqdm should not be initialized
        assert tracker._pbar is None # Internal _pbar should be None
        assert tracker.pbar is None # Property getter should return None
        assert "Progress bar disabled" in caplog.text

    def test_init_tqdm_fails(self, mock_tqdm, caplog): # Use caplog
        """Test initialization when tqdm instantiation fails."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        mock_tqdm_class.side_effect = Exception("TQDM Test Error")

        with caplog.at_level(logging.WARNING):
            tracker = ProgressTracker(total_steps=50)

        assert "Failed to initialize tqdm progress bar: TQDM Test Error" in caplog.text
        assert tracker._pbar is None # Should be None after failure
        assert tracker.pbar is None # Property getter should return None

    def test_start(self, mock_tqdm, caplog): # Use caplog
        """Test the start method initializes time and resets the bar."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        tracker = ProgressTracker(total_steps=10, desc="StartTest")
        with caplog.at_level(logging.INFO):
            tracker.start()

        assert tracker.start_time is not None
        assert tracker.last_log_time == tracker.start_time
        assert tracker.current_step == 0
        assert tracker.losses == []
        assert tracker.steps_per_second == [] # Check renamed list
        mock_tqdm_instance.reset.assert_called_once_with(total=10)
        assert "Starting StartTest..." in caplog.text # Check start log

    def test_update_with_tqdm(self, mock_tqdm, caplog): # Use caplog
        """Test update method correctly updates metrics and tqdm bar."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        tracker = ProgressTracker(total_steps=50, log_interval=10)
        tracker.start()

        # Simulate 5 steps
        # Mock the pbar's internal counter update for accurate delta calculation
        call_ns = []
        def update_side_effect(n):
            call_ns.append(mock_tqdm_instance.n)
            mock_tqdm_instance.n += n # Manually update the mock's internal state
        mock_tqdm_instance.update.side_effect = update_side_effect

        for i in range(1, 6):
            tracker.update(
                step=i,
                loss=0.5 / i,
                learning_rate=1e-3 / i,
                steps_per_second=10.0 * i, # Use renamed argument
                additional_metrics={"acc": 0.9 / i}
            )

        assert tracker.current_step == 5
        assert len(tracker.losses) == 5
        assert len(tracker.learning_rates) == 5
        assert len(tracker.steps_per_second) == 5 # Check renamed list
        assert len(tracker.step_times) == 5

        # Check tqdm update calls
        # FIX 3: Check the actual calls made based on the update logic
        # The update method calculates the delta based on the step and mock_tqdm_instance.n
        # step=1, n=0 -> delta=1 -> update(1), n becomes 1
        # step=2, n=1 -> delta=1 -> update(1), n becomes 2 ... etc.
        assert mock_tqdm_instance.update.call_count == 5
        mock_tqdm_instance.update.assert_has_calls([call(1)] * 5)

        # Check tqdm postfix calls
        assert mock_tqdm_instance.set_postfix.call_count == 5
        # Check the arguments of the last call
        expected_postfix_last_call = {
            'loss': f'{0.5 / 5:.4f}',
            'lr': f'{1e-3 / 5:.2e}',
            'S/s': f'{10.0 * 5:.1f}', # Check renamed metric key
            'acc': f'{0.9 / 5:.3f}'
        }
        mock_tqdm_instance.set_postfix.assert_called_with(expected_postfix_last_call, refresh=False)

    def test_update_without_tqdm(self, caplog): # Use caplog
        """Test update method functions without a tqdm bar (disabled or failed)."""
        tracker = ProgressTracker(total_steps=50, log_interval=5, disable_progress_bar=True)
        tracker.start()

        tracker.update(step=1, loss=0.5, learning_rate=1e-3, steps_per_second=10.0) # Use renamed argument

        assert tracker.current_step == 1
        assert len(tracker.losses) == 1
        assert tracker.pbar is None # Ensure no bar exists

        # Check logs if needed (e.g., for the log interval)
        with caplog.at_level(logging.INFO):
             # Trigger log
            for i in range(2, 6):
                tracker.update(step=i, loss=0.4, learning_rate=1e-3, steps_per_second=12.0) # Use renamed argument

        assert tracker.current_step == 5
        # Check log was triggered at step 5
        assert any("Step: 5/50" in rec.message for rec in caplog.records if rec.levelname == 'INFO')

    def test_update_triggers_log(self, mock_tqdm, caplog): # Use caplog
        """Test that update triggers logging at the specified interval."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        log_interval = 3
        tracker = ProgressTracker(total_steps=10, log_interval=log_interval)
        tracker.start()

        with caplog.at_level(logging.INFO):
            for i in range(1, log_interval + 2): # Go one step past the interval (steps 1, 2, 3, 4)
                tracker.update(step=i, loss=0.1 * i, learning_rate=1e-4, steps_per_second=5.0) # Use renamed arg

        # Log should have been triggered once at step=log_interval (step 3)
        log_records = [rec for rec in caplog.records if "Step: 3/10" in rec.message and rec.levelname == 'INFO']
        assert len(log_records) == 1, f"Expected 1 INFO log record containing 'Step: 3/10', found {len(log_records)}"
        # Check averaged loss in the message
        expected_avg_loss = np.mean([0.1, 0.2, 0.3])
        assert f"Loss: {expected_avg_loss:.4f}" in log_records[0].message # Avg of [0.1, 0.2, 0.3]
        assert "S/s: 5.0" in log_records[0].message

        # Ensure tqdm was updated correctly
        # We need to mock the side effect again if we want to assert calls precisely
        call_ns = []
        def update_side_effect(n):
            call_ns.append(mock_tqdm_instance.n)
            mock_tqdm_instance.n += n
        mock_tqdm_instance.update.side_effect = update_side_effect
        # Re-run updates to check call counts with side effect (bit redundant)
        tracker_re = ProgressTracker(total_steps=10, log_interval=log_interval)
        tracker_re.start()
        tracker_re._pbar = mock_tqdm_instance # Re-assign mock bar
        mock_tqdm_instance.n = 0 # Reset mock state
        mock_tqdm_instance.update.reset_mock()
        for i in range(1, log_interval + 2):
            tracker_re.update(step=i, loss=0.1 * i, learning_rate=1e-4, steps_per_second=5.0)

        assert mock_tqdm_instance.update.call_count == log_interval + 1 # Called for steps 1, 2, 3, 4

    @patch('time.monotonic', side_effect=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    def test_log_metrics_format(self, mock_monotonic, caplog): # Use caplog
        """Test the format of the log message produced by _log_metrics."""
        # Use caplog fixture for capturing logs
        tracker = ProgressTracker(total_steps=100, desc="TestLogFmt", log_interval=1)
        tracker.start() # time.monotonic returns 100.0

        # Simulate one step update to populate metrics
        with caplog.at_level(logging.INFO):
            tracker.update(
                step=1,
                loss=0.5555,
                learning_rate=1.23e-4,
                steps_per_second=25.67,
                additional_metrics={'val_acc': 0.9876}
            ) # time.monotonic returns 101.0 for update step time, then 102.0 for _log_metrics elapsed time

        assert len(caplog.records) > 0, "No log messages captured"
        # Find the log message (update calls _log_metrics internally)
        log_msg = ""
        for record in caplog.records:
            # Check log level and message content
            if record.levelname == 'INFO' and "TestLogFmt Step: 1/100" in record.message:
                 log_msg = record.message
                 break
        assert log_msg, "_log_metrics message not found in captured logs"

        # Check components of the log message
        assert "TestLogFmt Step: 1/100" in log_msg
        assert "Loss: 0.5555" in log_msg # Avg loss after 1 step is just the loss
        assert "LR: 1.23e-04" in log_msg
        assert "S/s: 25.7" in log_msg # Check renamed metric key and formatting
        assert "Step Time: 1.000s" in log_msg # 101.0 (update time) - 100.0 (start time)
        assert "Elapsed: 2.0s" in log_msg # 102.0 (log_metrics time) - 100.0 (start time)
        assert "val_acc: 0.988" in log_msg # Check additional metric formatting

    def test_log_metrics_averaging(self, caplog): # Use caplog
        """Test that _log_metrics correctly averages over the interval."""
        log_interval = 3
        tracker = ProgressTracker(total_steps=10, log_interval=log_interval)
        tracker.start()
        tracker.update(step=1, loss=0.1, learning_rate=1e-4, steps_per_second=10)
        tracker.update(step=2, loss=0.2, learning_rate=1e-4, steps_per_second=12)
        # Next update triggers log
        with caplog.at_level(logging.INFO):
            tracker.update(step=3, loss=0.3, learning_rate=1e-4, steps_per_second=14)
        tracker.update(step=4, loss=0.4, learning_rate=1e-4, steps_per_second=16)

        # Check log triggered at step 3 by searching message content and level
        log_records = [rec for rec in caplog.records if "Step: 3/10" in rec.message and rec.levelname == 'INFO']
        assert len(log_records) == 1, f"Expected 1 INFO log containing 'Step: 3/10', found {len(log_records)}"
        log_msg = log_records[0].message

        # Check averaged values in the log message (avg over steps 1, 2, 3)
        expected_avg_loss = np.mean([0.1, 0.2, 0.3])
        assert f"Loss: {expected_avg_loss:.4f}" in log_msg
        expected_avg_sps = np.mean([10, 12, 14])
        assert f"S/s: {expected_avg_sps:.1f}" in log_msg # Check renamed metric key

    def test_log_metrics_optional_none(self, caplog): # Use caplog
        """Test _log_metrics handles optional metrics being None."""
        tracker = ProgressTracker(total_steps=10, log_interval=1)
        tracker.start()
        with caplog.at_level(logging.INFO):
            tracker.update(step=1, loss=0.5) # No LR or S/s or additional

        # Check log triggered at step 1 by searching message content and level
        log_records = [rec for rec in caplog.records if "Step: 1/10" in rec.message and rec.levelname == 'INFO']
        assert len(log_records) == 1, f"Expected 1 INFO log containing 'Step: 1/10', found {len(log_records)}"
        log_msg = log_records[0].message

        assert "Step: 1/10" in log_msg
        assert "Loss: 0.5000" in log_msg
        assert "LR:" not in log_msg
        assert "S/s:" not in log_msg

    def test_get_average_metrics(self):
        """Test get_average_metrics calculation."""
        # Renamed from test_get_summary
        tracker = ProgressTracker(total_steps=5)
        tracker.start()
        tracker.update(step=1, loss=0.1, learning_rate=1e-4, steps_per_second=10)
        tracker.update(step=2, loss=0.2, learning_rate=2e-4, steps_per_second=12)
        tracker.update(step=3, loss=0.3, learning_rate=3e-4, steps_per_second=14)

        avg_metrics = tracker.get_average_metrics()

        assert isinstance(avg_metrics, dict)
        np.testing.assert_almost_equal(avg_metrics['avg_loss'], np.mean([0.1, 0.2, 0.3]))
        np.testing.assert_almost_equal(avg_metrics['avg_learning_rate'], np.mean([1e-4, 2e-4, 3e-4]))
        np.testing.assert_almost_equal(avg_metrics['avg_steps_per_second'], np.mean([10, 12, 14])) # Check renamed key
        assert 'avg_step_time' in avg_metrics # Step time depends on execution speed, just check presence
        # Ensure results are floats, not numpy types
        assert isinstance(avg_metrics['avg_loss'], float)
        assert isinstance(avg_metrics['avg_steps_per_second'], float)

    def test_get_average_metrics_no_updates(self):
        """Test get_average_metrics returns zeros when no updates occurred."""
        # Renamed from test_get_summary_no_updates
        tracker = ProgressTracker(total_steps=5)
        tracker.start() # Start but don't update

        avg_metrics = tracker.get_average_metrics()

        expected_zeros = {
            'avg_loss': 0.0,
            'avg_learning_rate': 0.0,
            'avg_steps_per_second': 0.0, # Check renamed key
            'avg_step_time': 0.0
        }
        assert avg_metrics == expected_zeros

    @patch('time.monotonic', side_effect=[100.0, 100.5, 110.0, 110.1, 110.2, 110.3])
    def test_close_with_tqdm(self, mock_monotonic, mock_tqdm, caplog):
        """Test close method closes tqdm and logs summary."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        tracker = ProgressTracker(total_steps=5)
        tracker.start() # time = 100.0
        tracker.update(step=1, loss=0.5, steps_per_second=10) # Use renamed arg, time = 100.5

        with caplog.at_level(logging.INFO):
            tracker.close() # time = 110.0 (for total_time), 110.1 (used inside get_average_metrics if it needed time)

        mock_tqdm_instance.close.assert_called_once()
        assert tracker._pbar is None # Internal reference should be cleared

        # Check summary logs captured by caplog
        finish_log = next((rec.message for rec in caplog.records if "finished in" in rec.message and rec.levelname == 'INFO'), None)
        loss_log = next((rec.message for rec in caplog.records if "Final Average Loss:" in rec.message and rec.levelname == 'INFO'), None)
        sps_log = next((rec.message for rec in caplog.records if "Final Average S/s:" in rec.message and rec.levelname == 'INFO'), None)
        step_time_log = next((rec.message for rec in caplog.records if "Final Average Step Time:" in rec.message and rec.levelname == 'INFO'), None)

        assert finish_log is not None and "finished in 10.00 seconds" in finish_log # 110.0 - 100.0
        assert loss_log is not None and "Final Average Loss: 0.5000" in loss_log
        assert sps_log is not None and "Final Average S/s: 10.0" in sps_log # Check renamed metric key
        assert "Final Average LR:" not in caplog.text # LR wasn't consistently provided
        assert step_time_log is not None # Step time is always calculated

    def test_close_without_tqdm(self, caplog):
        """Test close method logs summary when tqdm was disabled."""
        # Use caplog fixture instead of patching mock_logger
        tracker = ProgressTracker(total_steps=5, disable_progress_bar=True)
        tracker.start()
        tracker.update(step=1, loss=0.6)

        with caplog.at_level(logging.INFO):
            tracker.close()

        # No tqdm close call to assert
        assert tracker._pbar is None # Still None

        # Check summary logs using caplog
        assert any("finished in" in rec.message for rec in caplog.records if rec.levelname == 'INFO') # Check time part exists
        assert any("Final Average Loss: 0.6000" in rec.message for rec in caplog.records if rec.levelname == 'INFO')
        assert not any("Final Average S/s:" in rec.message for rec in caplog.records if rec.levelname == 'INFO') # S/s wasn't provided

    def test_update_step(self, mock_tqdm):
        """Test update_step method directly updates tqdm counter."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        tracker = ProgressTracker(total_steps=100)
        tracker.start()
        # Set the mock tqdm instance's internal counter explicitly for the test
        mock_tqdm_instance.n = 0

        tracker.update_step(5)
        assert tracker.current_step == 5
        # Check that pbar.update was called with the delta
        mock_tqdm_instance.update.assert_called_once_with(5) # 5 - 0

        mock_tqdm_instance.reset_mock() # Reset mock for next call
        mock_tqdm_instance.n = 5 # Update mock's internal counter state

        tracker.update_step(12)
        assert tracker.current_step == 12
        mock_tqdm_instance.update.assert_called_once_with(7) # 12 - 5

    def test_set_epoch(self, mock_tqdm):
        """Test set_epoch updates state and tqdm postfix."""
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm
        tracker = ProgressTracker(total_steps=100)
        tracker.start()
        # Initialize postfix dict on the mock instance, like tqdm would
        mock_tqdm_instance.postfix = {'loss': '0.5000'}

        tracker.set_epoch(current_epoch=2, total_epochs=5)

        assert tracker.current_epoch == 2
        assert tracker.total_epochs == 5
        assert tracker.epoch_steps == 0 # Should reset

        # Check if set_postfix was called to update epoch display
        expected_postfix = {'loss': '0.5000', 'epoch': '3/5'} # Epoch is 1-indexed for display
        mock_tqdm_instance.set_postfix.assert_called_once_with(expected_postfix, refresh=True)

    def test_context_manager(self, mock_tqdm, caplog):
        """Test the ProgressTracker as a context manager."""
        # Use caplog fixture
        mock_tqdm_class, mock_tqdm_instance = mock_tqdm

        with caplog.at_level(logging.INFO):
            with ProgressTracker(total_steps=10, desc="ContextTest") as tracker:
                assert isinstance(tracker, ProgressTracker)
                # Check start was called implicitly
                assert tracker.start_time is not None
                # Simulate some work inside the context
                tracker.update(step=1, loss=0.1)
                assert tracker.current_step == 1

        # Check close was called implicitly (tqdm closed, summary logged)
        mock_tqdm_instance.close.assert_called_once()
        assert tracker._pbar is None # Check internal state after close
        # Check logs for start and finish
        assert any(rec.message == "Starting ContextTest..." for rec in caplog.records if rec.levelname == 'INFO')
        assert any("ContextTest finished in" in rec.message for rec in caplog.records if rec.levelname == 'INFO')

# Removed tests for get_summary as the method no longer exists.
# Removed test for set_start_time_if_needed as the method no longer exists.

    # Removed tests for get_summary and set_start_time_if_needed as they no longer exist. 