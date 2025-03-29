import unittest
import logging
from unittest.mock import MagicMock, patch
import numpy as np
import torch

# Add project root to path to allow importing src
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.callbacks import ReduceLROnPlateauOrInstability, SampleGenerationCallback, TensorBoardLogger

# Disable verbose logging from the callback during tests unless needed
logging.getLogger("ReduceLROnPlateauOrInstability").setLevel(logging.CRITICAL)

class TestReduceLROnPlateauOrInstability(unittest.TestCase):

    def setUp(self):
        """Set up mock trainer, optimizer, and callback for each test."""
        self.initial_lr = 0.01
        self.factor = 0.5
        self.patience = 3
        self.threshold = 1.5
        self.min_lr = 1e-6
        self.cooldown = 2
        self.window_size = 5

        # Mock Optimizer
        self.mock_optimizer = MagicMock()
        # Simulate the structure optimizers use for learning rate
        self.mock_optimizer.param_groups = [{'lr': self.initial_lr}]

        # Mock Trainer (only need optimizer attribute for this callback)
        self.mock_trainer = MagicMock()
        self.mock_trainer.optimizer = self.mock_optimizer

        # Instantiate the Callback
        self.callback = ReduceLROnPlateauOrInstability(
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            min_lr=self.min_lr,
            cooldown=self.cooldown,
            window_size=self.window_size,
            verbose=False # Keep test output clean
        )

        # Link callback to mock trainer
        self.callback.set_trainer(self.mock_trainer)

    def _run_steps(self, losses):
        """Helper to run multiple steps with given losses."""
        for i, loss in enumerate(losses):
            self.callback.on_step_end(step=i, logs={'loss': loss})

    def assertLR(self, expected_lr, msg="LR mismatch"): # Custom assertion
        self.assertAlmostEqual(self.mock_optimizer.param_groups[0]['lr'], expected_lr, places=8, msg=msg)

    def test_stable_loss_no_reduction(self):
        """Test that stable loss below threshold does not trigger LR reduction."""
        stable_losses = [1.0] * (self.window_size + self.patience + 5) # Enough steps
        self._run_steps(stable_losses)
        self.assertLR(self.initial_lr, "LR should not change with stable loss")

    def test_spike_triggers_reduction(self):
        """Test loss spike above threshold triggers LR reduction after patience."""
        # Initial stable period to fill window
        losses = [1.0] * self.window_size
        self._run_steps(losses)
        self.assertLR(self.initial_lr, "LR should be initial before spike")

        # Introduce spike (loss > moving_avg * threshold)
        moving_avg = np.mean(losses[-self.window_size:])
        spike_loss = moving_avg * (self.threshold + 0.1)

        # Run patience steps with spike
        spike_steps = [spike_loss] * self.patience
        for i, loss in enumerate(spike_steps):
            step_num = self.window_size + i
            self.callback.on_step_end(step=step_num, logs={'loss': loss})
            if i < self.patience - 1:
                self.assertLR(self.initial_lr, f"LR should not change before patience exceeded (step {i+1})")

        # LR should be reduced on the last step of patience
        expected_lr = self.initial_lr * self.factor
        self.assertLR(expected_lr, "LR should be reduced after patience exceeded")

    def test_cooldown_prevents_reduction(self):
        """Test that no LR reduction happens during the cooldown period."""
        # Trigger one reduction first
        losses = [1.0] * self.window_size
        moving_avg = np.mean(losses)
        spike_loss = moving_avg * (self.threshold + 0.1)
        losses.extend([spike_loss] * self.patience)
        self._run_steps(losses)

        expected_lr_after_first_reduction = self.initial_lr * self.factor
        self.assertLR(expected_lr_after_first_reduction, "LR should be reduced once")

        # Simulate more spikes during cooldown
        cooldown_spikes = [spike_loss] * self.cooldown
        current_step = len(losses)
        for i, loss in enumerate(cooldown_spikes):
            self.callback.on_step_end(step=current_step + i, logs={'loss': loss})
            self.assertLR(expected_lr_after_first_reduction, f"LR should not change during cooldown (step {i+1})")

    def test_reduction_after_cooldown(self):
        """Test that LR reduction can happen again after cooldown finishes."""
        # 1. Trigger first reduction
        losses = [1.0] * self.window_size
        moving_avg = np.mean(losses)
        spike_loss = moving_avg * (self.threshold + 0.1)
        losses.extend([spike_loss] * self.patience)
        self._run_steps(losses)
        lr_after_first_reduction = self.initial_lr * self.factor
        self.assertLR(lr_after_first_reduction, "LR should be reduced once")

        # 2. Pass through cooldown (with stable loss to reset moving avg)
        stable_after_spike = [moving_avg * 0.9] * self.cooldown # Loss improves
        current_step = len(losses)
        self._run_steps(stable_after_spike)
        self.assertLR(lr_after_first_reduction, "LR should not change during cooldown")

        # 3. Trigger second reduction after cooldown
        # Need to refill window with stable values first
        new_stable_losses = [moving_avg * 0.9] * self.window_size
        self._run_steps(new_stable_losses)
        new_moving_avg = np.mean(new_stable_losses)
        new_spike_loss = new_moving_avg * (self.threshold + 0.1)
        second_spike_steps = [new_spike_loss] * self.patience
        current_step = len(losses) + len(stable_after_spike) + len(new_stable_losses)

        for i, loss in enumerate(second_spike_steps):
            self.callback.on_step_end(step=current_step + i, logs={'loss': loss})
            if i < self.patience - 1:
                 self.assertLR(lr_after_first_reduction, f"LR should not change before 2nd patience exceeded (step {i+1})")

        # LR should be reduced again
        expected_lr_after_second_reduction = lr_after_first_reduction * self.factor
        self.assertLR(expected_lr_after_second_reduction, "LR should be reduced a second time after cooldown")

    def test_min_lr_boundary(self):
        """Test that LR does not drop below min_lr."""
        # Set initial LR close to min_lr such that one reduction hits the boundary
        self.initial_lr = self.min_lr * (1 / self.factor) * 1.1 # Slightly above min_lr / factor
        self.mock_optimizer.param_groups = [{'lr': self.initial_lr}]
        self.callback.set_trainer(self.mock_trainer) # Re-link with updated optimizer

        # Trigger reduction
        losses = [1.0] * self.window_size
        moving_avg = np.mean(losses)
        spike_loss = moving_avg * (self.threshold + 0.1)
        losses.extend([spike_loss] * self.patience)
        self._run_steps(losses)

        # LR should be reduced, but might be slightly above min_lr
        expected_lr_after_reduction = max(self.initial_lr * self.factor, self.min_lr)
        self.assertLR(expected_lr_after_reduction, "LR should be reduced towards min_lr")

        # Try triggering another reduction - should stay at the previous value (or min_lr if it was hit)
        current_step = len(losses)
        self._run_steps([spike_loss] * (self.cooldown + self.patience + 1))
        # The LR should not decrease further
        self.assertLR(expected_lr_after_reduction, "LR should not decrease further after hitting boundary logic")

@patch('src.training.callbacks.logging')
class TestSampleGenerationCallback(unittest.TestCase):
    """Tests for the SampleGenerationCallback."""

    def setUp(self, mock_logging):
        """Set up mocks for SampleGenerationCallback tests."""
        # Get the logger instance *after* patch is active
        self.logger_instance = mock_logging.getLogger.return_value

        # Mock Tokenizer
        self.mock_tokenizer = MagicMock()
        # Create a mock object to be the return value of the tokenizer call
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.tensor([[101, 102]])
        mock_encoding.attention_mask = torch.tensor([[1, 1]])
        # Configure the tokenizer mock to return this object
        self.mock_tokenizer.return_value = mock_encoding
        self.mock_tokenizer.decode = MagicMock(side_effect=lambda x, **kwargs: f"decoded_{x.tolist()}")

        # Mock Model with a generate method
        self.mock_model = MagicMock()
        # Simulate generate returning sequences of token IDs
        self.mock_model.generate = MagicMock(return_value=torch.tensor([[101, 102, 103, 104]]))
        # Mock eval and train methods
        self.mock_model.eval = MagicMock()
        self.mock_model.train = MagicMock()

        # Mock Trainer
        self.mock_trainer = MagicMock()
        self.mock_trainer.model = self.mock_model
        self.mock_trainer.device = torch.device("cpu") # Use CPU for tests

        # Instantiate the Callback
        self.prompt = "Test prompt"
        self.callback = SampleGenerationCallback(
            tokenizer=self.mock_tokenizer,
            prompt=self.prompt,
            sample_every_n_steps=5, # Example: sample every 5 steps
            sample_on_epoch_end=True,
            max_new_tokens=10
        )

        # Link callback to trainer
        self.callback.set_trainer(self.mock_trainer)

    def test_initialization(self, mock_logging):
        """Test if the callback initializes correctly."""
        self.assertEqual(self.callback.tokenizer, self.mock_tokenizer)
        self.assertEqual(self.callback.prompt, self.prompt)
        self.assertEqual(self.callback.device, self.mock_trainer.device)

    def test_generates_on_step_interval(self, mock_logging):
        """Test that generation is triggered at the correct step interval."""
        # Use the logger instance from setUp
        logger_instance = self.logger_instance

        # Simulate steps
        for step in range(1, 12):
            self.callback.on_step_end(step=step, logs={})
            if step % self.callback.sample_every_n_steps == 0:
                 # Check if logger.info containing "Generating Sample" was called
                 self.assertTrue(any("Generating Sample" in call.args[0] for call in logger_instance.info.call_args_list))
                 # Check if model.generate was called
                 self.mock_model.generate.assert_called()
                 # Reset mocks for the next check
                 logger_instance.info.reset_mock()
                 self.mock_model.generate.reset_mock()
            else:
                self.assertFalse(any("Generating Sample" in call.args[0] for call in logger_instance.info.call_args_list))
                self.mock_model.generate.assert_not_called()

    def test_generates_on_epoch_end(self, mock_logging):
        """Test that generation is triggered on epoch end."""
        # Use the logger instance from setUp
        logger_instance = self.logger_instance

        self.callback.on_epoch_end(epoch=1, logs={})
        
        # Check if logger.info containing "Generating Sample" was called
        self.assertTrue(any("Generating Sample" in call.args[0] for call in logger_instance.info.call_args_list))
        # Check if model.generate was called
        self.mock_model.generate.assert_called_once()
        # Check if the model was put in eval mode and back to train
        self.mock_model.eval.assert_called_once()
        self.mock_model.train.assert_called_once()

    def test_no_generation_if_disabled(self, mock_logging):
        """Test that generation doesn't happen if intervals are zero/false."""
        # Use the logger instance from setUp
        logger_instance = self.logger_instance

        # Disable generation
        self.callback.sample_every_n_steps = 0
        self.callback.sample_on_epoch_end = False

        # Simulate steps and epoch end
        for step in range(1, 10):
            self.callback.on_step_end(step=step, logs={})
        self.callback.on_epoch_end(epoch=1, logs={})

        # Check that logger and generate were NOT called
        self.assertFalse(any("Generating Sample" in call.args[0] for call in logger_instance.info.call_args_list))
        self.mock_model.generate.assert_not_called()
        
    # TODO: Add test for when model doesn't have a generate method
    # TODO: Add test for correct prompt encoding and output decoding
    # TODO: Add test for correct device handling

@patch('torch.utils.tensorboard.SummaryWriter')
class TestTensorBoardLogger(unittest.TestCase):

    def setUp(self):
        """Set up mocks for TensorBoardLogger tests."""
        # Manually create a mock instance for setup purposes
        # Note: The actual instance used by the callback comes from MockSummaryWriter.return_value
        # in the test methods, which is provided by the class decorator.
        # This self.mock_writer is mainly a placeholder if needed *within* setUp itself,
        # but assertions in test_* methods should use the instance from the decorator.
        MockSummaryWriter = MagicMock() # Create a placeholder mock instance
        self.mock_writer = MockSummaryWriter.return_value

        self.log_dir = "test_tb_logs"

        # Mock Trainer
        self.mock_trainer = MagicMock()
        self.mock_trainer.global_step = 0
        self.mock_trainer.current_epoch = 0
        self.mock_trainer.config = {'log_dir': self.log_dir}

        # Instantiate the Callback
        self.callback = TensorBoardLogger(log_dir=self.log_dir)

        # Link callback to trainer
        self.callback.set_trainer(self.mock_trainer)

    def test_initialization(self, MockSummaryWriter):
        """Test if the callback initializes correctly."""
        self.assertEqual(self.callback.log_dir, self.log_dir)
        self.assertIsNone(self.callback.writer) # Writer should be None initially

    def test_writer_created_on_train_begin(self, MockSummaryWriter):
        """Test that SummaryWriter is created on train begin."""
        self.callback.on_train_begin()
        # Assert that the MockSummaryWriter class was called once with the correct log_dir
        MockSummaryWriter.assert_called_once_with(log_dir=self.log_dir)
        # Check that the instance created is assigned to the callback's writer
        # Note: self.mock_writer was created manually in setUp, so we need to compare
        # the callback's writer to the actual return value of the class mock call.
        self.assertEqual(self.callback.writer, MockSummaryWriter.return_value)

    def test_writer_closed_on_train_end(self, MockSummaryWriter):
        """Test that SummaryWriter is closed on train end."""
        # Ensure writer is created first
        self.callback.on_train_begin()
        # Get the actual writer instance created by on_train_begin
        writer_instance = MockSummaryWriter.return_value
        self.callback.on_train_end()
        # Assert close was called on the correct instance
        writer_instance.close.assert_called_once()

    def test_logs_scalar_on_step_end(self, MockSummaryWriter):
        """Test that add_scalar is called on step end with appropriate logs."""
        self.callback.on_train_begin() # Create writer
        writer_instance = MockSummaryWriter.return_value # Get the instance

        step = 10
        # Callback only logs 'loss' and 'lr' on step end
        logs = {'loss': 0.5, 'lr': 0.001}
        self.mock_trainer.global_step = step # Simulate trainer updating step

        self.callback.on_step_end(step=step, logs=logs)

        # Check calls to add_scalar on the correct instance
        # Use the correct tag 'Loss/train_step'
        writer_instance.add_scalar.assert_any_call('Loss/train_step', logs['loss'], global_step=step)
        # Optionally check for lr log
        writer_instance.add_scalar.assert_any_call('LearningRate/step', logs['lr'], global_step=step)
        # Remove check for accuracy as it's not logged on step_end

    def test_logs_scalar_on_epoch_end(self, MockSummaryWriter):
        """Test that add_scalar is called on epoch end for relevant metrics."""
        self.callback.on_train_begin() # Create writer
        writer_instance = MockSummaryWriter.return_value # Get the instance

        epoch = 5
        # Use 'loss' key for train loss as expected by callback
        logs = {'loss': 0.6, 'val_loss': 0.7, 'other_metric': 1.0}
        self.mock_trainer.current_epoch = epoch # Simulate trainer updating epoch

        self.callback.on_epoch_end(epoch=epoch, logs=logs)

        # Use epoch + 1 as global_step
        expected_epoch_step = epoch + 1

        # Check calls to add_scalar on the correct instance
        # Check train loss using 'loss' key and epoch+1
        writer_instance.add_scalar.assert_any_call('Loss/train_epoch', logs['loss'], global_step=expected_epoch_step)
        # Check val loss using 'Loss/validation_epoch' tag and epoch+1
        writer_instance.add_scalar.assert_any_call('Loss/validation_epoch', logs['val_loss'], global_step=expected_epoch_step)

        # Ensure it didn't log the 'other_metric' with the default keys
        for call in writer_instance.add_scalar.call_args_list:
            self.assertNotIn('Other_Metric', call.args[0])
            self.assertNotIn('other_metric', call.args[0])

    def test_no_action_if_writer_not_created(self, MockSummaryWriter):
        """Test that logging methods don't crash if writer wasn't created (e.g., train_begin wasn't called)."""
        # Try calling step/epoch end without calling train_begin first
        try:
            self.callback.on_step_end(step=1, logs={'loss': 0.5})
            self.callback.on_epoch_end(epoch=1, logs={'train_loss': 0.6})
            self.callback.on_train_end()
        except Exception as e:
            self.fail(f"Callback methods raised an exception when writer was None: {e}")

        # Ensure no calls were made to the writer
        self.mock_writer.add_scalar.assert_not_called()
        self.mock_writer.close.assert_not_called()


if __name__ == '__main__':
    unittest.main() 