import pytest
from unittest.mock import MagicMock, call, patch
from typing import List
import torch
import numpy as np
import logging

# Import the classes to test
from craft.training.callbacks import Callback, CallbackList

# --- Mocks --- #

@pytest.fixture
def mock_callback():
    """Provides a basic MagicMock callback."""
    # Create a mock that adheres to the Callback interface (optional but good practice)
    callback = MagicMock(spec=Callback)
    return callback

# mock_trainer fixture now provided by conftest.py

# --- Tests for CallbackList --- #

class TestCallbackList:

    def test_init_empty(self):
        """Test initialization with no callbacks."""
        cb_list = CallbackList()
        assert cb_list.callbacks == []

    def test_init_with_callbacks(self, mock_callback):
        """Test initialization with a list of callbacks."""
        callbacks = [mock_callback, MagicMock(spec=Callback)]
        cb_list = CallbackList(callbacks)
        assert cb_list.callbacks == callbacks

    def test_append(self, mock_callback):
        """Test appending a single callback."""
        cb_list = CallbackList()
        cb_list.append(mock_callback)
        assert cb_list.callbacks == [mock_callback]

    def test_extend(self, mock_callback):
        """Test extending with multiple callbacks."""
        cb_list = CallbackList([mock_callback])
        new_callbacks = [MagicMock(spec=Callback), MagicMock(spec=Callback)]
        cb_list.extend(new_callbacks)
        assert cb_list.callbacks == [mock_callback] + new_callbacks

    def test_set_trainer(self, mock_trainer):
        """Test that set_trainer calls set_trainer on all contained callbacks."""
        mock_cb1 = MagicMock(spec=Callback)
        mock_cb2 = MagicMock(spec=Callback)
        cb_list = CallbackList([mock_cb1, mock_cb2])

        cb_list.set_trainer(mock_trainer)

        mock_cb1.set_trainer.assert_called_once_with(mock_trainer)
        mock_cb2.set_trainer.assert_called_once_with(mock_trainer)

    # Test dispatching of each event type
    @pytest.mark.parametrize(
        "method_name, method_args",
        [
            ("on_train_begin", {}),
            ("on_train_end", {"logs": {"final_metric": 1}}),
            ("on_epoch_begin", {"epoch": 1, "logs": {"epoch_start": True}}),
            ("on_epoch_end", {"epoch": 1, "logs": {"epoch_loss": 0.5}}),
            ("on_step_begin", {"step": 100, "logs": {}}),
            ("on_step_end", {"step": 100, "logs": {"loss": 0.1}}),
        ]
    )
    def test_event_dispatch(self, method_name, method_args):
        """Test that event methods are correctly dispatched to all callbacks."""
        mock_cb1 = MagicMock(spec=Callback)
        mock_cb2 = MagicMock(spec=Callback)
        cb_list = CallbackList([mock_cb1, mock_cb2])

        # Get the method on CallbackList and call it
        list_method = getattr(cb_list, method_name)
        cb_list.trainer = MagicMock() # Set a mock trainer to satisfy checks
        list_method(**method_args)

        # Assert the corresponding method was called on each mock callback
        # Separate assertions for clarity
        cb1_method = getattr(mock_cb1, method_name)
        cb2_method = getattr(mock_cb2, method_name)

        # Unpack args based on method name for assertion
        pos_args = []
        kw_args = {}
        if "epoch" in method_args: pos_args.append(method_args["epoch"])
        if "step" in method_args: pos_args.append(method_args["step"])
        # Ensure logs dict is passed, even if originally None/empty
        kw_args["logs"] = method_args.get("logs", {})

        # Only expect trainer arg for non-step methods
        expected_args = [cb_list.trainer] + pos_args if "step" not in method_name else pos_args
        cb1_method.assert_called_once_with(*expected_args, **kw_args)
        cb2_method.assert_called_once_with(*expected_args, **kw_args)


# --- Tests for ReduceLROnPlateauOrInstability --- #

# REMOVED - Tests moved to test_reduce_lr.py

# --- Tests for SampleGenerationCallback --- #

# REMOVED - Tests moved to test_sample_generation.py

# --- Tests for TensorBoardLogger --- #

# REMOVED - Tests moved to test_tensorboard_logger.py

# --- Tests for EarlyStopping (Standalone Additions due to edit issues) --- #

# Delete the standalone test function below
# def test_early_stopping_on_train_begin_resets_state():
#     ...


# --- Tests for TensorBoardCallback --- #

# REMOVED - Tests moved to test_tensorboard_logger.py

# --- End of Tests --- #