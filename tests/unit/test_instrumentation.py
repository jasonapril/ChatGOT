import unittest
import os
import sys
import tempfile
import json
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.performance.instrumentation import (
    InstrumentedModel,
    InstrumentedDataLoader,
    measure_batch,
    measure_optimizer_step,
    measure_data_loading,
    create_instrumented_model,
    create_instrumented_dataloader,
    save_monitor_stats
)
from src.performance.throughput_core import ThroughputMonitor

class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)

class TestInstrumentation(unittest.TestCase):
    """Unit tests for the instrumentation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.model = SimpleModel()
        
        # Create a monitor
        self.monitor = ThroughputMonitor()
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Simple test tensors
        self.inputs = torch.randn(32, 10)
        self.targets = torch.randn(32, 5)
        
        # Create a simple dataset and dataloader
        self.dataset = [(self.inputs[i], self.targets[i]) for i in range(len(self.inputs))]
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=False
        )
    
    def test_instrumented_model_creation(self):
        """Test creating an instrumented model."""
        # Create instrumented model
        inst_model = create_instrumented_model(self.model, self.monitor)
        
        # Check that it's an instance of InstrumentedModel
        self.assertIsInstance(inst_model, InstrumentedModel)
        
        # Check that it wraps our model
        self.assertEqual(inst_model.model, self.model)
        
        # Check that it has the monitor
        self.assertEqual(inst_model.monitor, self.monitor)
    
    def test_instrumented_model_forward(self):
        """Test forward pass of instrumented model."""
        # Create instrumented model
        inst_model = create_instrumented_model(self.model, self.monitor)
        
        # Forward pass
        with torch.no_grad():
            output = inst_model(self.inputs)
        
        # Check output shape
        self.assertEqual(output.shape, (32, 5))
        
        # Check that forward time was recorded in the correct list
        self.assertEqual(len(self.monitor.forward_times), 1)
    
    def test_instrumented_dataloader_creation(self):
        """Test creating an instrumented dataloader."""
        # Create instrumented dataloader
        inst_loader = create_instrumented_dataloader(self.dataloader, self.monitor)
        
        # Check that it's an instance of InstrumentedDataLoader
        self.assertIsInstance(inst_loader, InstrumentedDataLoader)
        
        # Check that it wraps our dataloader
        self.assertEqual(inst_loader.dataloader, self.dataloader)
        
        # Check that it has the monitor
        self.assertEqual(inst_loader.monitor, self.monitor)
        
        # Check that it preserves dataloader properties
        self.assertEqual(inst_loader.batch_size, self.dataloader.batch_size)
        self.assertEqual(inst_loader.dataset, self.dataloader.dataset)
    
    def test_instrumented_dataloader_iteration(self):
        """Test iteration of instrumented dataloader."""
        # Create instrumented dataloader
        inst_loader = create_instrumented_dataloader(self.dataloader, self.monitor)
        
        # Iterate through the dataloader
        for batch in inst_loader:
            # Just ensure we can iterate without errors
            pass
        
        # Check that data loading time was recorded
        self.assertGreaterEqual(len(self.monitor.data_loading_times), 1)
    
    def test_context_managers(self):
        """Test the context managers for measuring time."""
        # Test measure_optimizer_step
        with measure_optimizer_step(self.monitor):
            time.sleep(0.01)
        # Check that optimizer step was recorded
        self.assertEqual(len(self.monitor.optimizer_times), 1)
        
        # Test measure_data_loading
        with measure_data_loading(self.monitor):
            time.sleep(0.01)
        # Check that data loading was recorded
        self.assertEqual(len(self.monitor.data_loading_times), 1)
    
    @patch('json.dump')
    @patch('os.fsync')
    @patch('os.makedirs')
    def test_save_monitor_stats(self, mock_makedirs, mock_fsync, mock_json_dump):
        """Test saving monitor statistics to a file."""
        # Add some data to the monitor
        self.monitor.batch_times = [0.1, 0.2]
        self.monitor.total_tokens = 1000
        self.monitor.total_samples = 100
        
        # Create output path
        output_path = os.path.join(self.temp_dir, "monitor_stats.json")
        
        # Mock json.dump to avoid any actual file operations
        mock_json_dump.return_value = None
        
        # Use patched open to avoid actual file operations
        with patch('builtins.open') as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Call the function under test - using direct execution (no threading)
            result = save_monitor_stats(
                self.monitor, 
                output_path, 
                timeout=1.0, 
                use_threading=False  # Direct execution
            )
            
            # Check the result is success 
            self.assertTrue(result)
            
            # Verify file operations occurred
            mock_open.assert_called_once_with(output_path, 'w')
            mock_json_dump.assert_called_once()
            
            # Check json.dump called with correct data
            summary_arg = mock_json_dump.call_args[0][0]
            self.assertIn('throughput', summary_arg)
            self.assertIn('component_breakdown', summary_arg)
            self.assertIn('memory', summary_arg)
    
    def test_extract_batch_info(self):
        """Test extracting batch info from different input types."""
        # Create instrumented model
        inst_model = create_instrumented_model(self.model, self.monitor)
        
        # Test with 2D tensor [batch_size, seq_length]
        batch_size, seq_length = inst_model.extract_batch_info(torch.zeros(16, 32))
        self.assertEqual(batch_size, 16)
        self.assertEqual(seq_length, 32)
        
        # Test with 1D tensor [seq_length]
        batch_size, seq_length = inst_model.extract_batch_info(torch.zeros(32))
        self.assertEqual(batch_size, 1)
        self.assertEqual(seq_length, 32)
        
        # Test with list of tensors
        batch_size, seq_length = inst_model.extract_batch_info([torch.zeros(16, 32)])
        self.assertEqual(batch_size, 16)
        self.assertEqual(seq_length, 32)
        
        # Test with invalid input
        batch_size, seq_length = inst_model.extract_batch_info(None)
        self.assertEqual(batch_size, 1)
        self.assertEqual(seq_length, 1)

if __name__ == '__main__':
    unittest.main() 