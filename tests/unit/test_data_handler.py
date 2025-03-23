"""
Unit tests for the data_handler module.
"""

import unittest
import os
import torch
import pickle
import tempfile
import shutil
from src.data_handler import load_data

class TestDataHandler(unittest.TestCase):
    """Tests for data_handler.py functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a small test dataset
        self.test_data = {
            'train_inputs': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'train_targets': torch.tensor([[2, 3, 4], [5, 6, 7]]),
            'val_inputs': torch.tensor([[7, 8, 9]]),
            'val_targets': torch.tensor([[8, 9, 10]]),
            'char_to_idx': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10},
            'idx_to_char': {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j'},
            'sequence_length': 3,
            'metadata': {
                'original_file': 'test_file.txt',
                'text_length': 100,
                'vocab_size': 10,
                'processed_at': 'CPU'
            }
        }
        
        # Save test data to a temporary file
        self.test_data_file = os.path.join(self.temp_dir, 'test_data.pkl')
        with open(self.test_data_file, 'wb') as f:
            pickle.dump(self.test_data, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_load_data(self):
        """Test loading data from pickle file."""
        # Test with default parameters
        train_loader, val_loader, char_to_idx, idx_to_char = load_data(
            data_path=self.test_data_file,
            batch_size=2,
            device_type='cpu'
        )
        
        # Verify the data loaders
        self.assertEqual(len(train_loader), 1)  # should be 1 batch with batch_size=2
        self.assertEqual(len(val_loader), 1)    # should be 1 batch with batch_size=2
        
        # Verify character mappings
        self.assertEqual(len(char_to_idx), 10)
        self.assertEqual(len(idx_to_char), 10)
        self.assertEqual(char_to_idx['a'], 1)
        self.assertEqual(idx_to_char[1], 'a')
    
    def test_load_data_with_different_batch_size(self):
        """Test loading data with different batch sizes."""
        # Test with batch_size=1
        train_loader, val_loader, _, _ = load_data(
            data_path=self.test_data_file,
            batch_size=1,
            device_type='cpu'
        )
        
        # Verify the data loaders
        self.assertEqual(len(train_loader), 2)  # should be 2 batches with batch_size=1
        self.assertEqual(len(val_loader), 1)    # should be 1 batch with batch_size=1
    
    def test_load_data_with_workers(self):
        """Test loading data with workers."""
        # Test with num_workers=2
        train_loader, val_loader, _, _ = load_data(
            data_path=self.test_data_file,
            batch_size=2,
            device_type='cpu',
            num_workers=0  # Use 0 for testing
        )
        
        # Verify the data loaders
        self.assertEqual(len(train_loader), 1)
        self.assertEqual(len(val_loader), 1)

if __name__ == '__main__':
    unittest.main() 