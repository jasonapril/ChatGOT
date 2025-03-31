import unittest
import os
import sys
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging import force_flush_logs, format_time
from src.training.evaluation import (
    evaluate,
    evaluate_with_metrics,
    evaluate_perplexity
)

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

class TestEvaluation(unittest.TestCase):
    """Unit tests for the evaluation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.model = SimpleModel()
        
        # Create dataloader
        self.dataloader = MockDataLoader()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure model is moved to the correct device
        self.model.to(self.device)
    
    @patch('time.time')
    @patch('logging.info')
    def test_evaluate_basic(self, mock_logging, mock_time):
        """Test basic evaluation functionality."""
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.5, 1.0]
        
        # Run evaluation
        avg_loss = evaluate(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            use_amp=False
        )
        
        # Check that the function returned the expected values
        self.assertIsInstance(avg_loss, float)
        
        # Check that logging.info was called
        mock_logging.assert_called()
    
    @patch('time.time')
    @patch('logging.info')
    def test_evaluate_with_log_interval(self, mock_logging, mock_time):
        """Test evaluation with logging interval."""
        # Mock time.time to return sequential times - Provide enough values
        mock_time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Run evaluation with log interval
        avg_loss = evaluate(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            use_amp=False,
            log_interval=0.3  # Log every 0.3 seconds
        )
        
        # Check that the function returned the expected values
        self.assertIsInstance(avg_loss, float)
        
        # Check that logging.info was called more than just for the final results
        # We expect initial message + progress logs + final results
        self.assertGreater(mock_logging.call_count, 3)
    
    @patch('time.time')
    @patch('logging.info')
    def test_evaluate_with_metrics(self, mock_logging, mock_time):
        """Test evaluation with custom metrics."""
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.5, 1.0]
        
        # Define custom metrics
        def accuracy(outputs, targets):
            # Dummy accuracy metric
            return torch.tensor(0.75)
            
        def f1_score(outputs, targets):
            # Dummy F1 score metric
            return torch.tensor(0.8)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1_score
        }
        
        # Run evaluation with metrics
        results = evaluate_with_metrics(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            use_amp=False,
            metrics=metrics
        )
        
        # Check that the function returned the expected metrics
        self.assertIn('loss', results)
        self.assertIn('accuracy', results)
        self.assertIn('f1_score', results)
        self.assertIn('tokens_per_sec', results)
        self.assertIn('elapsed_time', results)
        
        # Check that metric values were calculated
        self.assertAlmostEqual(results['accuracy'], 0.75, places=2)
        self.assertAlmostEqual(results['f1_score'], 0.8, places=2)
        
        # Check that logging.info was called
        mock_logging.assert_called()
    
    @patch('time.time')
    @patch('logging.info')
    def test_evaluate_perplexity(self, mock_logging, mock_time):
        """Test perplexity evaluation."""
        # Mock time.time to return sequential times
        mock_time.side_effect = [0.0, 0.5, 1.0]
        
        # Run perplexity evaluation
        avg_loss, perplexity = evaluate_perplexity(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            use_amp=False
        )
        
        # Check that the function returned the expected values
        self.assertIsInstance(avg_loss, float)
        self.assertIsInstance(perplexity, float)
        
        # Perplexity should be equal to exp(avg_loss)
        expected_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        self.assertAlmostEqual(perplexity, expected_perplexity, places=5)
        
        # Check that logging.info was called
        mock_logging.assert_called()

if __name__ == '__main__':
    unittest.main() 