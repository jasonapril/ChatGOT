"""
Integration tests for the pipeline module.
"""

import unittest
import os
import tempfile
import shutil
import sys
import torch
import pickle
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pipeline import Pipeline, parse_args

class MockArgs:
    """Mock arguments for testing the pipeline."""
    
    def __init__(self):
        self.pipeline_dir = None
        self.resume = False
        self.force_restart = True
        self.stage = None
        
        self.input_file = None
        self.processed_data_path = None
        self.sequence_length = 128
        self.skip_process = False
        
        self.skip_optimization = True
        self.test_batches = 2
        self.min_batch_size = 1
        self.max_batch_size = 8
        
        self.batch_size = 4
        self.epochs = 1
        self.learning_rate = 1e-3
        self.gradient_accumulation_steps = 1
        self.save_every = 1
        self.use_amp = False
        self.force_aggressive_memory = False
        
        self.generate_length = 100
        self.generate_seed = "Test: "
        self.temperature = 0.8
        self.num_samples = 1
        
        self.seed = 42
        self.log_level = "INFO"
        self.force_cpu = True
        
        self.d_model = 128
        self.n_head = 4
        self.n_layers = 2
        self.d_hid = 256

class TestPipeline(unittest.TestCase):
    """Integration tests for pipeline.py."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test dataset
        self.create_test_dataset()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory, handling file permission errors
        try:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        os.chmod(file_path, 0o777)  # Ensure we have permissions
                        os.remove(file_path)
                    except (PermissionError, OSError):
                        pass
            
            # Try removing directory, but don't fail test if it fails
            try:
                shutil.rmtree(self.temp_dir)
            except (PermissionError, OSError):
                pass
        except Exception as e:
            print(f"Warning: Error in tearDown: {e}")
    
    def create_test_dataset(self):
        """Create a small test dataset for pipeline testing."""
        # Create a simple dataset with just a few sequences
        dataset = {
            'train_inputs': torch.tensor([[1, 2, 3, 4] * 32, [5, 6, 7, 8] * 32]),
            'train_targets': torch.tensor([[2, 3, 4, 5] * 32, [6, 7, 8, 9] * 32]),
            'val_inputs': torch.tensor([[9, 10, 11, 12] * 32]),
            'val_targets': torch.tensor([[10, 11, 12, 13] * 32]),
            'char_to_idx': {chr(i+97): i for i in range(26)},  # a-z
            'idx_to_char': {i: chr(i+97) for i in range(26)},  # a-z
            'sequence_length': 128,
            'metadata': {
                'original_file': 'test_file.txt',
                'text_length': 1000,
                'vocab_size': 26,
                'processed_at': 'test'
            }
        }
        
        # Save to temp file
        self.data_path = os.path.join(self.temp_dir, 'test_data.pkl')
        with open(self.data_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_pipeline_initialization(self, mock_args):
        """Test that the pipeline initializes correctly."""
        # Set up mock args
        args = MockArgs()
        args.pipeline_dir = os.path.join(self.temp_dir, 'pipeline')
        args.processed_data_path = self.data_path
        mock_args.return_value = args
        
        # Initialize pipeline
        pipeline = Pipeline(args)
        
        # Check that pipeline was initialized
        self.assertEqual(pipeline.pipeline_dir, args.pipeline_dir)
        self.assertTrue(os.path.exists(args.pipeline_dir))
        
        # Check that state was initialized
        self.assertIsNotNone(pipeline.state)
        self.assertIsNone(pipeline.state["last_completed_stage"])
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_pipeline_process_stage(self, mock_args):
        """Test the process stage of the pipeline."""
        # Create a simple input file
        input_file = os.path.join(self.temp_dir, 'input.txt')
        with open(input_file, 'w') as f:
            f.write("This is a test file with some sample text for processing.")
        
        # Set up mock args
        args = MockArgs()
        args.pipeline_dir = os.path.join(self.temp_dir, 'pipeline')
        args.input_file = input_file
        args.processed_data_path = os.path.join(self.temp_dir, 'processed.pkl')
        args.stage = "process"
        mock_args.return_value = args
        
        # Initialize pipeline and run process stage
        with patch('src.data_processor.process_raw_data') as mock_process:
            # Mock the process_raw_data to just create an empty file
            def mock_process_impl(input_file, output_path, *args, **kwargs):
                with open(output_path, 'wb') as f:
                    pickle.dump({
                        'train_sequences': [],
                        'val_sequences': [],
                        'char_to_idx': {},
                        'idx_to_char': {},
                        'sequence_length': 128,
                        'metadata': {}
                    }, f)
            
            mock_process.side_effect = mock_process_impl
            
            pipeline = Pipeline(args)
            pipeline.run_process()
            
            # Check that process_raw_data was called
            mock_process.assert_called_once()
            
            # Check that processed file was created
            self.assertTrue(os.path.exists(args.processed_data_path))
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_run_optimize_with_skip(self, mock_args):
        """Test the optimize stage with skip_optimization=True."""
        # Set up mock args
        args = MockArgs()
        args.pipeline_dir = os.path.join(self.temp_dir, 'pipeline')
        args.processed_data_path = self.data_path
        args.stage = "optimize"
        args.skip_optimization = True
        args.batch_size = 4
        mock_args.return_value = args
        
        # Initialize pipeline and run optimize stage
        pipeline = Pipeline(args)
        pipeline.run_optimize()
        
        # Check that settings were stored correctly
        settings = pipeline.state["stages"]["optimize"]["settings"]
        self.assertEqual(settings["batch_size"], 4)
        self.assertEqual(settings["gradient_accumulation_steps"], 1)
        self.assertEqual(settings["use_amp"], False)

if __name__ == '__main__':
    unittest.main() 