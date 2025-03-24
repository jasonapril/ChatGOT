import unittest
import os
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.throughput_core import ThroughputMonitor

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
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.max_memory_allocated')
    def test_memory_stats_cuda(self, mock_max_memory, mock_reserved, mock_allocated, mock_is_available):
        """Test memory statistics with CUDA."""
        # Mock CUDA availability and memory stats
        mock_is_available.return_value = True
        mock_allocated.return_value = 1024 * 1024 * 1024  # 1 GB
        mock_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2 GB
        mock_max_memory.return_value = 1.5 * 1024 * 1024 * 1024  # 1.5 GB
        
        # Get memory stats
        stats = self.monitor.get_memory_stats()
        
        # Check stats
        self.assertIn('allocated_gb', stats)
        self.assertIn('reserved_gb', stats)
        self.assertIn('peak_gb', stats)
        
        self.assertAlmostEqual(stats['allocated_gb'], 1.0, places=1)
        self.assertAlmostEqual(stats['reserved_gb'], 2.0, places=1)
        self.assertAlmostEqual(stats['peak_gb'], 1.5, places=1)
    
    def test_memory_stats_no_cuda(self):
        """Test memory statistics without CUDA."""
        # Get memory stats (assuming no CUDA in test environment)
        stats = self.monitor.get_memory_stats()
        
        # Check stats
        self.assertIn('allocated_gb', stats)
        self.assertIn('reserved_gb', stats)
        self.assertIn('peak_gb', stats)
        
        # All should be 0 if CUDA not available
        if not torch_cuda_available():
            self.assertEqual(stats['allocated_gb'], 0)
            self.assertEqual(stats['reserved_gb'], 0)
            self.assertEqual(stats['peak_gb'], 0)
    
    def test_get_throughput(self):
        """Test throughput calculation."""
        # Add batch times
        self.monitor.batch_times = [0.1, 0.2, 0.15]
        self.monitor.total_tokens = 1000
        self.monitor.total_samples = 100
        
        # Get throughput
        throughput = self.monitor.get_throughput()
        
        # Check throughput calculation
        # Average time is (0.1 + 0.2 + 0.15) / 3 = 0.15
        # Expected tokens per second = 1000 / (3 * 0.15) = 2222.22...
        expected_tokens_per_sec = 1000 / (3 * 0.15)
        self.assertAlmostEqual(throughput, expected_tokens_per_sec, delta=1)
    
    def test_get_summary(self):
        """Test summary generation."""
        # Add some data
        self.monitor.batch_times = [0.1, 0.2, 0.15]
        self.monitor.total_tokens = 1000
        self.monitor.total_samples = 100
        
        # Record a component
        self.monitor.component_times = {
            'data_loading': [0.01, 0.02],
            'forward': [0.05, 0.06]
        }
        
        # Get summary
        summary = self.monitor.get_summary()
        
        # Check summary structure
        self.assertIn('throughput', summary)
        self.assertIn('component_breakdown', summary)
        self.assertIn('memory', summary)
        self.assertIn('samples_per_second', summary['throughput'])
        self.assertIn('tokens_per_second', summary['throughput'])
        
        # Check summary values
        self.assertGreater(summary['throughput']['tokens_per_second'], 0)
        self.assertGreater(summary['throughput']['samples_per_second'], 0)

# Helper function to check if torch.cuda is available
def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False

if __name__ == '__main__':
    unittest.main() 