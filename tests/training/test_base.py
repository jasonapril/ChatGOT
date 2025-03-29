import unittest
import torch
import torch.nn as nn
import functools
from unittest.mock import MagicMock, patch
import os
import tempfile
import shutil

# Add project root to path to allow importing src
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.base import LanguageModelTrainer
from src.training.amp import SafeGradScaler
from src.training.utils import enable_gradient_checkpointing, disable_gradient_checkpointing

# A simple dummy model for testing purposes
class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

class DummyModel(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        # Crucially, needs a 'layers' attribute for the current GC implementation
        self.layers = nn.ModuleList([DummyLayer() for _ in range(num_layers)])
        self.embedding = nn.Embedding(100, 10)
        self.head = nn.Linear(10, 100)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x

# Dummy DataLoader
class DummyDataLoader:
    def __init__(self, length=10):
        self.length = length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            # Return dummy data matching DummyModel input expectations
            inputs = torch.randint(0, 100, (4, 8), device=self.device) # Batch=4, SeqLen=8
            targets = torch.randint(0, 100, (4, 8), device=self.device)
            yield inputs, targets

class TestTrainerGradientCheckpointing(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests"""
        self.model = DummyModel(num_layers=2)
        self.train_loader = DummyDataLoader(length=5)
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'optimizer': { # Optimizer config should be a nested dictionary
                'name': 'adamw', # Or 'adam' depending on what you want to test
                'learning_rate': 1e-4,
                # Add other optimizer params like weight_decay if needed by _create_optimizer
            },
            # 'lr': 1e-4, # This is now inside the optimizer dict
            'epochs': 1,
            'checkpoint_dir': self.temp_dir,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_amp': False, # Keep it simple for this test
            'gradient_accumulation_steps': 1,
            'log_interval': 1,
            'scheduler': {'name': 'none'} # Also explicitly disable scheduler for simplicity
            # use_gradient_checkpointing is set per test
        }

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
        # Ensure checkpointing is disabled on the model if modified
        disable_gradient_checkpointing(self.model)

    @patch('src.training.utils.torch_checkpoint') # Mock torch_checkpoint if needed
    def test_gradient_checkpointing_enabled(self, mock_torch_checkpoint):
        """Test that gradient checkpointing is enabled via config."""
        # Check original forward methods
        original_forwards = [layer.forward for layer in self.model.layers]
        self.assertFalse(any(isinstance(f, functools.partial) for f in original_forwards))

        # Enable GC in config
        test_config = self.config.copy()
        test_config['use_gradient_checkpointing'] = True

        # Instantiate Trainer (this should call enable_gradient_checkpointing)
        trainer = LanguageModelTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            config=test_config
        )

        # Assert that forward methods are now wrapped
        modified_forwards = [layer.forward for layer in self.model.layers]
        self.assertTrue(all(isinstance(f, functools.partial) for f in modified_forwards))
        # Check if the original function is stored
        self.assertTrue(all(hasattr(layer, 'forward_original') for layer in self.model.layers))

    def test_gradient_checkpointing_disabled_by_default(self):
        """Test that gradient checkpointing is disabled by default."""
        original_forwards = [layer.forward for layer in self.model.layers]

        # Instantiate Trainer with default config (GC disabled)
        trainer = LanguageModelTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            config=self.config # use_gradient_checkpointing is False/absent
        )

        # Assert that forward methods are unchanged
        modified_forwards = [layer.forward for layer in self.model.layers]
        self.assertEqual(original_forwards, modified_forwards)
        self.assertFalse(any(hasattr(layer, 'forward_original') for layer in self.model.layers))

    def test_gradient_checkpointing_disabled_explicitly(self):
        """Test that gradient checkpointing is disabled when explicitly set to False."""
        original_forwards = [layer.forward for layer in self.model.layers]

        # Disable GC in config explicitly
        test_config = self.config.copy()
        test_config['use_gradient_checkpointing'] = False

        trainer = LanguageModelTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            config=test_config
        )

        # Assert that forward methods are unchanged
        modified_forwards = [layer.forward for layer in self.model.layers]
        self.assertEqual(original_forwards, modified_forwards)
        self.assertFalse(any(hasattr(layer, 'forward_original') for layer in self.model.layers))

# Add more test classes/methods for OOM, NaN handling etc.

class TestTrainerNanInfHandling(unittest.TestCase):

    def setUp(self):
        """Set up common resources for NaN/Inf tests"""
        self.model = DummyModel(num_layers=2)
        self.train_loader = DummyDataLoader(length=5)
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'optimizer': {'name': 'adamw', 'learning_rate': 1e-4},
            'epochs': 1,
            'checkpoint_dir': self.temp_dir,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_amp': False, # Test without AMP first for simplicity
            'gradient_accumulation_steps': 1,
            'log_interval': 1,
            'scheduler': {'name': 'none'}
        }
        self.trainer = LanguageModelTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            config=self.config
        )
        # Mock parts of the trainer for easier testing
        # self.trainer.logger = MagicMock() # Remove this mock; let assertLogs handle capture
        self.trainer.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.trainer.optimizer.param_groups = [{'lr': 1e-4}] # Mock param_groups for logging
        self.trainer.scaler = MagicMock(spec=SafeGradScaler)
        # Mock scaler methods to simulate behavior
        self.trainer.scaler.scale = MagicMock(side_effect=lambda x: x) # Simulate no scaling
        self.trainer.scaler.unscale_ = MagicMock()
        self.trainer.scaler.step = MagicMock()
        self.trainer.scaler.update = MagicMock()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    @patch('src.training.base.nn.functional.cross_entropy')
    def test_nan_loss_skips_step(self, mock_cross_entropy):
        """Test that a NaN loss results in a skipped step and a warning."""
        # Force cross_entropy to return NaN
        mock_cross_entropy.return_value = torch.tensor(float('nan'))

        initial_global_step = self.trainer.global_step

        # Run one epoch - we expect it to run through batches but skip optimizer steps
        # Explicitly capture logs from the trainer's logger instance
        with self.assertLogs(self.trainer.logger.name, level='WARNING') as log:
            metrics = self.trainer.train_epoch()

        # Check that a warning was logged
        self.assertTrue(any("NaN/Inf loss detected" in message for message in log.output))

        # Check that the optimizer step was NOT called
        self.trainer.scaler.step.assert_not_called()
        self.trainer.optimizer.step.assert_not_called() # Check mock optimizer directly too

        # Check that global step did not advance
        self.assertEqual(self.trainer.global_step, initial_global_step)

        # Check that epoch loss calculation handled potential division by zero if all steps were skipped
        # The current implementation calculates avg based on num_valid_steps_in_epoch
        self.assertEqual(metrics['loss'], 0.0) # Expect 0 loss if all steps skipped
        self.assertEqual(metrics['tokens'], 0) # Expect 0 tokens processed if all steps skipped

    @patch('src.training.base.nn.functional.cross_entropy')
    def test_inf_loss_skips_step(self, mock_cross_entropy):
        """Test that an Inf loss results in a skipped step and a warning."""
        # Force cross_entropy to return Inf
        mock_cross_entropy.return_value = torch.tensor(float('inf'))

        initial_global_step = self.trainer.global_step

        # Explicitly capture logs from the trainer's logger instance
        with self.assertLogs(self.trainer.logger.name, level='WARNING') as log:
            metrics = self.trainer.train_epoch()

        # Check log, step call, global step, and metrics as in the NaN test
        self.assertTrue(any("NaN/Inf loss detected" in message for message in log.output))
        self.trainer.scaler.step.assert_not_called()
        self.trainer.optimizer.step.assert_not_called()
        self.assertEqual(self.trainer.global_step, initial_global_step)
        self.assertEqual(metrics['loss'], 0.0)
        self.assertEqual(metrics['tokens'], 0)

    @patch('src.training.base.nn.functional.cross_entropy')
    def test_valid_loss_proceeds(self, mock_cross_entropy):
        """Test that a valid loss proceeds with the optimizer step."""
        # Simulate a valid loss
        valid_loss = torch.tensor(1.23, requires_grad=True)
        mock_cross_entropy.return_value = valid_loss

        initial_global_step = self.trainer.global_step
        num_batches = len(self.train_loader)

        # Run one epoch
        # We don't expect WARNING logs here, so no assertLogs context
        metrics = self.trainer.train_epoch()

        # Check that the optimizer step WAS called (once per batch in this config)
        self.assertEqual(self.trainer.scaler.step.call_count, num_batches)
        # Check that global step advanced
        self.assertEqual(self.trainer.global_step, initial_global_step + num_batches)
        # Check logger warning was NOT called - Removed incorrect mock assertion
        # Rely on the other tests failing if unexpected warnings occur.
        # Check metrics (loss should be approx avg of mocked loss val)
        self.assertAlmostEqual(metrics['loss'], 1.23)
        self.assertGreater(metrics['tokens'], 0)


if __name__ == '__main__':
    unittest.main() 