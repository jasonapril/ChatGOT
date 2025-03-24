import unittest
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.training_loop import train_epoch

class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)

class MockDataLoader:
    """A mock dataloader for testing."""
    def __init__(self, num_batches=5, batch_size=4, seq_length=10):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        
    def __iter__(self):
        for i in range(self.num_batches):
            inputs = torch.randn(self.batch_size, self.seq_length, 10)
            targets = torch.randint(0, 5, (self.batch_size, self.seq_length))
            yield inputs, targets
            
    def __len__(self):
        return self.num_batches

class MockScheduler:
    """A mock scheduler for testing."""
    def __init__(self):
        self.step_count = 0
        
    def step(self):
        self.step_count += 1

class TestTrainingLoop(unittest.TestCase):
    """Unit tests for the training loop module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.model = SimpleModel()
        
        # Create optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = MockScheduler()
        
        # Create dataloader
        self.dataloader = MockDataLoader()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @patch('time.time')
    @patch('logging.info')
    def test_train_epoch_basic(self, mock_logging, mock_time):
        """Test basic training loop functionality."""
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Run training loop
        avg_loss, tokens_per_sec = train_epoch(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader=self.dataloader,
            device=self.device,
            epoch=1,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,
            use_amp=False
        )
        
        # Check that the function returned the expected values
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(tokens_per_sec, float)
        
        # Check that the scheduler was stepped the correct number of times
        self.assertEqual(self.scheduler.step_count, self.dataloader.num_batches)
        
        # Check that logging.info was called
        mock_logging.assert_called()
    
    @patch('time.time')
    @patch('logging.info')
    def test_train_epoch_gradient_accumulation(self, mock_logging, mock_time):
        """Test training loop with gradient accumulation."""
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Set gradient accumulation steps
        gradient_accumulation_steps = 2
        
        # Run training loop
        avg_loss, tokens_per_sec = train_epoch(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader=self.dataloader,
            device=self.device,
            epoch=1,
            max_grad_norm=1.0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=False
        )
        
        # Check that the scheduler was stepped the correct number of times
        # With 5 batches and accumulation steps of 2, we should have 3 updates
        # (at batch 2, 4, and 5)
        expected_steps = (self.dataloader.num_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
        self.assertEqual(self.scheduler.step_count, expected_steps)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @patch('time.time')
    @patch('logging.info')
    def test_train_epoch_with_amp(self, mock_logging, mock_time):
        """Test training loop with automatic mixed precision."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Create scaler
        scaler = torch.cuda.amp.GradScaler()
        
        # Run training loop
        avg_loss, tokens_per_sec = train_epoch(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader=self.dataloader,
            device=torch.device("cuda"),
            epoch=1,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,
            use_amp=True,
            scaler=scaler
        )
        
        # Check that the function returned the expected values
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(tokens_per_sec, float)
    
    @patch('time.time')
    @patch('logging.info')
    def test_train_epoch_early_stopping(self, mock_logging, mock_time):
        """Test training loop with early stopping."""
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Create a custom model that raises an exception after 2 batches
        class EarlyStoppingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.batch_count = 0
                
            def forward(self, x):
                self.batch_count += 1
                if self.batch_count > 2:
                    raise RuntimeError("Early stopping")
                return self.linear(x)
        
        early_model = EarlyStoppingModel()
        
        # Run training loop and expect an exception
        with self.assertRaises(RuntimeError):
            train_epoch(
                model=early_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                dataloader=self.dataloader,
                device=self.device,
                epoch=1,
                max_grad_norm=1.0,
                gradient_accumulation_steps=1,
                use_amp=False
            )
        
        # Check that the model processed 3 batches before stopping (count starts at 0, increments to 3)
        self.assertEqual(early_model.batch_count, 3)

if __name__ == '__main__':
    unittest.main() 