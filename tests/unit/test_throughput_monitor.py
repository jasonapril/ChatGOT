import unittest
import os
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.performance.throughput_core import ThroughputMonitor

class TestThroughputMonitor(unittest.TestCase):
    """Unit tests for the ThroughputMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = ThroughputMonitor(window_size=10)
    
    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.window_size, 10)
        self.assertEqual(len(self.monitor.batch_times), 0)
        self.assertEqual(self.monitor.total_tokens, 0)
        self.assertEqual(self.monitor.total_samples, 0)
    
    def test_reset(self):
        """Test monitor reset functionality."""
        # Add some data
        self.monitor.batch_times = [1.0, 2.0]
        self.monitor.total_tokens = 100
        self.monitor.total_samples = 10
        
        # Reset
        self.monitor.reset()
        
        # Check everything is reset
        self.assertEqual(len(self.monitor.batch_times), 0)
        self.assertEqual(self.monitor.total_tokens, 0)
        self.assertEqual(self.monitor.total_samples, 0)
    
    def test_batch_timing(self):
        """Test batch timing recording."""
        batch_size = 16
        seq_length = 64
        
        # Start and end a batch with a small delay
        self.monitor.start_batch(batch_size, seq_length)
        time.sleep(0.01)  # Small delay
        self.monitor.end_batch()
        
        # Check that batch time was recorded
        self.assertEqual(len(self.monitor.batch_times), 1)
        self.assertGreater(self.monitor.batch_times[0], 0)
        
        # Check that tokens and samples were recorded
        self.assertEqual(self.monitor.total_tokens, batch_size * seq_length)
        self.assertEqual(self.monitor.total_samples, batch_size)
    
    def test_component_timing(self):
        """Test component timing recording."""
        batch_size = 16
        seq_length = 64
        
        # Start batch
        self.monitor.start_batch(batch_size, seq_length)
        
        # Data loading
        self.monitor.start_data_loading()
        time.sleep(0.01)
        self.monitor.end_data_loading()
        
        # Forward pass
        self.monitor.start_forward(batch_size, seq_length)
        time.sleep(0.01)
        self.monitor.end_forward()
        
        # Backward pass
        self.monitor.start_backward()
        time.sleep(0.01)
        self.monitor.end_backward()
        
        # Optimizer step
        self.monitor.start_optimizer()
        time.sleep(0.01)
        self.monitor.end_optimizer()
        
        # End batch
        self.monitor.end_batch()
        
        # Get component breakdown
        breakdown = self.monitor.get_component_breakdown()
        
        # Check that all components were recorded
        self.assertIn('data_loading', breakdown)
        self.assertIn('forward', breakdown)
        self.assertIn('backward', breakdown)
        self.assertIn('optimizer', breakdown)
        
        # Check that times are positive
        self.assertGreater(breakdown['data_loading'], 0)
        self.assertGreater(breakdown['forward'], 0)
        self.assertGreater(breakdown['backward'], 0)
        self.assertGreater(breakdown['optimizer'], 0)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 1024) # 1 GB
    @patch('torch.cuda.memory_reserved', return_value=2048 * 1024 * 1024) # 2 GB
    @patch('torch.cuda.max_memory_allocated', return_value=0) # Mock peak initially
    def test_memory_stats_cuda(self, mock_max_alloc, mock_reserved, mock_allocated, mock_is_available):
        """Test memory statistics with CUDA."""
        stats = self.monitor.get_memory_stats()
        self.assertIn('allocated', stats) # Check for 'allocated' in MB
        self.assertIn('reserved', stats)  # Check for 'reserved' in MB
        self.assertIn('peak', stats)      # Check for 'peak' in MB
        self.assertAlmostEqual(stats['allocated'], 1024.0)
        self.assertAlmostEqual(stats['reserved'], 2048.0)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_memory_stats_no_cuda(self, mock_is_available):
        """Test memory statistics without CUDA."""
        # Simulate some peak memory recording (might happen via end_batch if cuda was briefly available)
        self.monitor.peak_memory = 500.0 # 500 MB
        stats = self.monitor.get_memory_stats()
        self.assertIn('allocated', stats) 
        self.assertIn('reserved', stats)  
        self.assertIn('peak', stats)      
        self.assertEqual(stats['allocated'], 0) # Should be 0 if no cuda
        self.assertEqual(stats['reserved'], 0)  # Should be 0 if no cuda
        self.assertAlmostEqual(stats['peak'], 500.0) # Peak should be retained
    
    def test_get_throughput(self):
        """Test throughput calculation."""
        # Simulate some batches
        batch_size = 10
        seq_length = 20
        num_batches = 5
        mock_batch_time = 0.09 # 90 ms

        for _ in range(num_batches):
            # Need to call start_forward to log tokens_per_batch
            self.monitor.start_forward(batch_size, seq_length)
            # Simulate batch time directly for simplicity in this test
            self.monitor.batch_times.append(mock_batch_time)
            # Simulate recent throughput calculation (as done in end_batch)
            tokens_in_batch = batch_size * seq_length
            self.monitor.recent_throughputs.append(tokens_in_batch / mock_batch_time)

        # Ensure the lists are populated correctly for the test
        self.assertEqual(len(self.monitor.batch_times), num_batches)
        self.assertEqual(len(self.monitor.tokens_per_batch), num_batches)
        self.assertEqual(len(self.monitor.recent_throughputs), num_batches)
        
        throughput = self.monitor.get_throughput()
        expected_tokens_per_sec = (batch_size * seq_length) / mock_batch_time
        self.assertAlmostEqual(throughput, expected_tokens_per_sec, delta=expected_tokens_per_sec * 0.01)
    
    def test_get_summary(self):
        """Test summary generation."""
        # Simulate some data
        self.monitor.batch_times = [0.1, 0.12, 0.11]
        self.monitor.forward_times = [0.05, 0.06]
        self.monitor.tokens_per_batch = [100, 100, 100]
        self.monitor.recent_throughputs = [1000.0, 833.3, 909.1]
        self.monitor.total_samples = 30
        self.monitor.total_tokens = 300
        
        summary = self.monitor.get_summary()
        
        self.assertIn('throughput', summary)
        self.assertIsInstance(summary['throughput'], float) # Check it's a float now
        self.assertAlmostEqual(summary['throughput'], np.mean([1000.0, 833.3, 909.1]))
        
        self.assertIn('component_breakdown', summary)
        self.assertIsInstance(summary['component_breakdown'], dict)
        self.assertIn('forward', summary['component_breakdown'])
        
        self.assertIn('memory', summary)
        self.assertIsInstance(summary['memory'], dict)
        self.assertIn('allocated', summary['memory']) # Check for MB key
        
        self.assertIn('total_samples', summary)
        self.assertEqual(summary['total_samples'], 30)
        # ... add other checks as needed ...

if __name__ == '__main__':
    unittest.main() 