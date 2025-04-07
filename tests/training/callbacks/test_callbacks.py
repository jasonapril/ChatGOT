import pytest
from unittest.mock import MagicMock, call, patch
from typing import List
import torch
import numpy as np
import logging
import inspect

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
        cb_list.append(new_callbacks[0])
        cb_list.append(new_callbacks[1])
        assert cb_list.callbacks == [mock_callback] + new_callbacks

    def test_set_trainer(self, mock_trainer):
        """Test that set_trainer calls set_trainer on all contained callbacks."""
        mock_cb1 = MagicMock(spec=Callback)
        mock_cb2 = MagicMock(spec=Callback)
        cb_list = CallbackList([mock_cb1, mock_cb2])

        cb_list.set_trainer(mock_trainer)

        mock_cb1.set_trainer.assert_called_once_with(mock_trainer)
        mock_cb2.set_trainer.assert_called_once_with(mock_trainer)

    @pytest.mark.parametrize(
        "method_name, call_args_to_list, expected_args_to_callback",
        [
            # on_train_begin: Called with **kwargs. Individual callback receives **kwargs.
            ("on_train_begin", {"extra_arg": 123}, {"extra_arg": 123}),

            # on_train_end: Called with metrics=..., **kwargs. Individual callback receives metrics=..., **kwargs.
            ("on_train_end", {"metrics": {"final_metric": 1}, "extra_arg": 456}, {"metrics": {"final_metric": 1}, "extra_arg": 456}),

            # on_epoch_begin: Called with trainer, current_epoch, global_step, **kwargs. Individual receives all.
            ("on_epoch_begin", {"current_epoch": 1, "global_step": 10, "extra_arg": 789}, {"trainer": "mock_trainer_placeholder", "current_epoch": 1, "global_step": 10, "extra_arg": 789}),

            # on_epoch_end: Called with epoch, global_step, metrics, **kwargs. Individual receives all.
            ("on_epoch_end", {"epoch": 1, "global_step": 20, "metrics": {"loss": 0.5}, "extra_arg": "abc"}, {"epoch": 1, "global_step": 20, "metrics": {"loss": 0.5}, "extra_arg": "abc"}),

            # on_step_begin: Called with step, logs=... Individual receives step (positional), **logs.
            ("on_step_begin", {"step": 100, "logs": {"lr": 0.01}}, {"step": 100, "lr": 0.01}),

            # on_step_end: Called with step, logs=... Individual receives step (positional), **logs.
            ("on_step_end", {"step": 100, "logs": {"loss": 0.1}}, {"step": 100, "loss": 0.1}),
        ]
    )
    def test_event_dispatch(self, method_name, call_args_to_list, expected_args_to_callback):
        """Test that CallbackList correctly dispatches events to individual callbacks."""
        mock_cb1 = MagicMock(spec=Callback)
        mock_cb2 = MagicMock(spec=Callback)
        # Ensure the mock methods exist for getattr checks later
        for cb in [mock_cb1, mock_cb2]:
            # Add the method if it doesn't exist, crucial for spec'd mocks
            if not hasattr(cb, method_name):
                 setattr(cb, method_name, MagicMock())
            # Ensure the attribute IS callable if it exists
            elif not callable(getattr(cb, method_name)):
                 setattr(cb, method_name, MagicMock())


        cb_list = CallbackList([mock_cb1, mock_cb2])
        mock_trainer = MagicMock(name="MockTrainerInstance") # Give mock a name
        cb_list.set_trainer(mock_trainer) # Explicitly set trainer

        # Get the method on CallbackList to call
        list_method = getattr(cb_list, method_name)

        # Prepare args for calling the CallbackList method
        actual_call_args = call_args_to_list.copy()

        # Dynamically add trainer if the list method expects it explicitly as a keyword
        sig = inspect.signature(list_method)
        if "trainer" in sig.parameters:
            # We call everything with kwargs, so only add if it's a kwarg
            # (This check might be redundant if we always call with kwargs, but safe)
             actual_call_args["trainer"] = mock_trainer


        # Special handling for on_epoch_begin which has specific positional args in CallbackList
        # We will call it using keyword arguments for consistency in the test setup.
        # The signature requires trainer, current_epoch, global_step.
        if method_name == "on_epoch_begin":
             required_args = {"trainer": mock_trainer, "current_epoch": actual_call_args["current_epoch"], "global_step": actual_call_args["global_step"]}
             # Add any extra args from the test parameters
             extra_args = {k: v for k, v in actual_call_args.items() if k not in required_args}
             required_args.update(extra_args)
             actual_call_args = required_args # Use this complete set for the call


        # Call the CallbackList method using keyword arguments
        list_method(**actual_call_args)


        # Get the method on the individual mock callbacks
        cb1_method = getattr(mock_cb1, method_name)
        cb2_method = getattr(mock_cb2, method_name)

        # Prepare the expected arguments dictionary for the individual callback assertion
        final_expected_args_dict = expected_args_to_callback.copy()
        # Replace placeholder trainer with the actual mock trainer object
        if "trainer" in final_expected_args_dict and final_expected_args_dict["trainer"] == "mock_trainer_placeholder":
            final_expected_args_dict["trainer"] = mock_trainer

        # --- Assert based on how CallbackList calls individual callbacks ---
        if method_name in ["on_step_begin", "on_step_end"]:
            # CallbackList calls these with step as positional, logs unpacked as kwargs
            # And trainer is passed via **kwargs
            expected_pos_arg = final_expected_args_dict.pop("step")
            cb1_method.assert_called_once_with(expected_pos_arg, **final_expected_args_dict)
            cb2_method.assert_called_once_with(expected_pos_arg, **final_expected_args_dict)
        else:
            # All other methods are called with keywords / **kwargs by CallbackList
            # including trainer, epoch, global_step, metrics where applicable
            cb1_method.assert_called_once_with(**final_expected_args_dict)
            cb2_method.assert_called_once_with(**final_expected_args_dict)


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