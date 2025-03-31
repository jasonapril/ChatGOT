"""
Unit tests for data loading and processing components.
"""

import unittest
import tempfile
import os
import torch
from omegaconf import OmegaConf
import json
import sys
from unittest.mock import patch, MagicMock
# Import Hydra exception for checking
from hydra.errors import InstantiationException

# Modules to test
from src.data.base import BaseDataset, create_dataset_from_config
from src.data.dataset import CharDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestCharDataset(unittest.TestCase):
    """Tests for the CharDataset class."""

    def setUp(self):
        """Create temporary text and vocab files for testing."""
        self.test_text = "abcdefghijklmnopqrstuvwxyz"
        self.block_size = 5
        
        # Create temp data file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_file.write(self.test_text)
        self.temp_file.close()
        self.temp_file_path = self.temp_file.name
        
        # Create a vocab file CONSISTENT with test_text
        chars = sorted(list(set(self.test_text)))
        vocab_data = {
            'char_to_idx': {ch: i for i, ch in enumerate(chars)},
            'idx_to_char': {i: ch for i, ch in enumerate(chars)},
            'vocab_size': len(chars)
        }
        self.temp_vocab_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_vocab_file.write(json.dumps(vocab_data))
        self.temp_vocab_file.close()
        self.temp_vocab_path = self.temp_vocab_file.name
        
        # Create a separate dummy vocab file for testing load-only scenarios
        self.dummy_vocab_data = {'char_to_idx': {'X': 0}, 'idx_to_char': {'0': 'X'}, 'vocab_size': 1}
        self.temp_dummy_vocab_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_dummy_vocab_file.write(json.dumps(self.dummy_vocab_data))
        self.temp_dummy_vocab_file.close()
        self.temp_dummy_vocab_path = self.temp_dummy_vocab_file.name

    def tearDown(self):
        """Remove the temporary files."""
        os.remove(self.temp_file_path)
        os.remove(self.temp_vocab_path)
        os.remove(self.temp_dummy_vocab_path)

    def test_initialization(self):
        """Test initialization building vocab from file (implicit)."""
        # Provide the consistent vocab_path, CharDataset should load it
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size, vocab_path=self.temp_vocab_path)
        self.assertIsInstance(dataset, BaseDataset)
        # Assert vocab was loaded correctly
        self.assertEqual(dataset.vocab_size, len(set(self.test_text)))
        # Assert data was loaded from file_path
        self.assertEqual(len(dataset.data), len(self.test_text))
        self.assertEqual(dataset.block_size, self.block_size)
        self.assertEqual(dataset.char_to_idx['a'], 0) # Based on consistent vocab
        self.assertEqual(dataset.idx_to_char[0], 'a') # Based on consistent vocab

    def test_len(self):
        """Test the __len__ method."""
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size, vocab_path=self.temp_vocab_path)
        expected_len = len(self.test_text) - self.block_size
        self.assertEqual(len(dataset), expected_len)

    def test_getitem(self):
        """Test the __getitem__ method."""
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size, vocab_path=self.temp_vocab_path)
        idx = 0
        sample = dataset[idx]
        
        # Check type and keys
        self.assertIsInstance(sample, dict)
        self.assertIn('input_ids', sample)
        self.assertIn('labels', sample)
        
        # Check tensor types and shapes
        self.assertIsInstance(sample['input_ids'], torch.Tensor)
        self.assertIsInstance(sample['labels'], torch.Tensor)
        self.assertEqual(sample['input_ids'].shape, (self.block_size,))
        self.assertEqual(sample['labels'].shape, (self.block_size,))
        self.assertEqual(sample['input_ids'].dtype, torch.long)
        self.assertEqual(sample['labels'].dtype, torch.long)
        
        # Check content (labels are input shifted by 1)
        expected_input_text = self.test_text[idx : idx + self.block_size]
        expected_label_text = self.test_text[idx + 1 : idx + self.block_size + 1]
        expected_input_ids = torch.tensor([dataset.char_to_idx[c] for c in expected_input_text], dtype=torch.long)
        expected_label_ids = torch.tensor([dataset.char_to_idx[c] for c in expected_label_text], dtype=torch.long)
        
        self.assertTrue(torch.equal(sample['input_ids'], expected_input_ids))
        self.assertTrue(torch.equal(sample['labels'], expected_label_ids))

    def test_decode(self):
        """Test the decode method."""
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size, vocab_path=self.temp_vocab_path)
        indices = [0, 1, 2, 3, 4] # Corresponds to 'abcde' with consistent vocab
        decoded_text = dataset.decode(indices)
        self.assertEqual(decoded_text, "abcde")
        
        # Test with tensor input
        indices_tensor = torch.tensor(indices)
        decoded_text_tensor = dataset.decode(indices_tensor)
        self.assertEqual(decoded_text_tensor, "abcde")
        
        # Test with unknown index
        indices_unknown = [0, 1, 99, 3] # 99 is out of bounds
        decoded_unknown = dataset.decode(indices_unknown)
        self.assertEqual(decoded_unknown, "ab?d") # Uses get default

    def test_empty_file(self):
        """Test initialization with an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as empty_f:
            empty_f_path = empty_f.name
        # Provide a vocab path (even dummy) as it's required
        dataset = CharDataset(file_path=empty_f_path, block_size=self.block_size, vocab_path=self.temp_dummy_vocab_path)
        # Length should be based on file content, which is 0
        self.assertEqual(len(dataset.data), 0)
        self.assertEqual(len(dataset), 0) 
        # Vocab size should be from the loaded vocab file
        self.assertEqual(dataset.vocab_size, self.dummy_vocab_data['vocab_size'])
        self.assertEqual(dataset.decode([0]), "X") # Decode based on loaded vocab
        os.remove(empty_f_path)

    def test_short_file(self):
        """Test initialization with file content shorter than block_size."""
        short_text = "abc"
        block_size = 5
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as short_f:
            short_f.write(short_text)
            short_f_path = short_f.name
        # Create a vocab consistent with short_text
        chars_short = sorted(list(set(short_text)))
        vocab_data_short = {'char_to_idx': {ch: i for i, ch in enumerate(chars_short)}, 'idx_to_char': {i: ch for i, ch in enumerate(chars_short)}, 'vocab_size': len(chars_short)}
        temp_vocab_short_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        temp_vocab_short_file.write(json.dumps(vocab_data_short))
        temp_vocab_short_file.close()
        temp_vocab_short_path = temp_vocab_short_file.name
        
        dataset = CharDataset(file_path=short_f_path, block_size=block_size, vocab_path=temp_vocab_short_path)
        # Data length is from file
        self.assertEqual(len(dataset.data), len(short_text))
        # Dataset length (__len__) is max(0, data_len - block_size)
        self.assertEqual(len(dataset), 0)
        # Vocab size is from vocab file
        self.assertEqual(dataset.vocab_size, len(set(short_text)))
        # __getitem__ should raise IndexError if len is 0
        with self.assertRaises(IndexError):
            _ = dataset[0]
            
        os.remove(short_f_path)
        os.remove(temp_vocab_short_path)

    def test_invalid_file_path(self):
        """Test initialization with an invalid file path but valid vocab path."""
        with self.assertRaises(FileNotFoundError):
             # Provide valid dummy vocab, but invalid file path
            CharDataset(file_path='non_existent_file.txt', block_size=self.block_size, vocab_path=self.temp_dummy_vocab_path)

    def test_invalid_vocab_path(self):
        """Test initialization with valid file path but invalid vocab path."""
        with self.assertRaises(FileNotFoundError):
             # Provide valid file path, but invalid vocab path
            CharDataset(file_path=self.temp_file_path, block_size=self.block_size, vocab_path='non_existent_vocab.json')

class TestDatasetFactory(unittest.TestCase):
    """Tests for the dataset factory function."""
    
    def setUp(self):
        """Create basic configs for testing."""
        # Create dummy files needed for some tests
        self.temp_data_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_data_file.write("abc")
        self.temp_data_file.close()
        self.temp_data_path = self.temp_data_file.name

        self.temp_vocab_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_vocab_file.write(json.dumps({'char_to_idx': {'a': 0, 'b': 1, 'c': 2}, 'idx_to_char': {'0': 'a', '1': 'b', '2': 'c'}, 'vocab_size': 3}))
        self.temp_vocab_file.close()
        self.temp_vocab_path = self.temp_vocab_file.name
        
        # Basic config structure mimicking Hydra
        self.base_data_config = OmegaConf.create({
            'dataset': {
                'name': 'my_char_dataset' 
            }
        })
        self.split_config = OmegaConf.create({
             # Assuming file_path and block_size are now under split_config
             'file_path': self.temp_data_path,
             'block_size': 5,
             'vocab_path': self.temp_vocab_path
        })
        self.cwd = os.getcwd()

    def tearDown(self):
        """Remove temporary files."""
        os.remove(self.temp_data_path)
        os.remove(self.temp_vocab_path)

    def test_factory_hydra_instantiation(self):
        """Test creating a dataset using Hydra's _target_."""
        # Add _target_ for direct instantiation
        split_config_with_target = OmegaConf.merge(self.split_config, {
            '_target_': 'src.data.dataset.CharDataset'
        })
        
        dataset = create_dataset_from_config(self.base_data_config, split_config_with_target, self.cwd)
        self.assertIsInstance(dataset, CharDataset)
        # Add more assertions based on expected dataset properties
        self.assertEqual(dataset.vocab_size, 3)
        self.assertEqual(len(dataset.data), 3)

    def test_factory_missing_target_fallback(self):
        """Test factory fallback when _target_ is missing (needs investigation)."""
        # This test depends heavily on the internal logic of create_dataset_from_config
        # Currently, the function likely relies on _target_ or might have other lookup logic.
        # For now, let's just call it and expect it *might* raise an error or return None/BaseDataset
        # We will refine this test after reading the function body.
        with self.assertRaises((ValueError, TypeError, NotImplementedError)):
             # Call without _target_ 
             create_dataset_from_config(self.base_data_config, self.split_config, self.cwd)

    def test_factory_invalid_target(self):
        """Test error when _target_ points to a non-existent class."""
        split_config_invalid_target = OmegaConf.merge(self.split_config, {
            '_target_': 'src.data.non_existent.NonExistentDataset'
        })
        with self.assertRaises(Exception): # Expect some form of import or instantiation error
            create_dataset_from_config(self.base_data_config, split_config_invalid_target, self.cwd)

    def test_factory_missing_required_arg(self):
        """Test error when required args for the target are missing."""
        incomplete_split_config = self.split_config.copy()
        del incomplete_split_config.block_size
        
        split_config_with_target = OmegaConf.merge(incomplete_split_config, {
            '_target_': 'src.data.dataset.CharDataset'
        })
        
        # Expect Hydra's InstantiationException which wraps the TypeError
        with self.assertRaises(InstantiationException): 
            create_dataset_from_config(self.base_data_config, split_config_with_target, self.cwd)

if __name__ == '__main__':
    unittest.main() 