"""
Unit tests for data loading and processing components.
"""

import unittest
import tempfile
import os
import torch
from omegaconf import OmegaConf

# Modules to test
from src.data.base import BaseDataset, create_dataset_from_config
from src.data.dataset import CharDataset

class TestCharDataset(unittest.TestCase):
    """Tests for the CharDataset class."""

    def setUp(self):
        """Create a temporary text file for testing."""
        self.test_text = "abcdefghijklmnopqrstuvwxyz"
        self.block_size = 5
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_file.write(self.test_text)
        self.temp_file.close()
        self.temp_file_path = self.temp_file.name
        
        # Keep config for factory tests, but use direct args for direct tests
        self.config_dict = {
            'file_path': self.temp_file_path,
            'block_size': self.block_size,
            'format': 'character' # Needed for fallback factory test
        }

    def tearDown(self):
        """Remove the temporary file."""
        os.remove(self.temp_file_path)

    def test_initialization(self):
        """Test successful dataset initialization."""
        # Use direct args
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size)
        self.assertIsInstance(dataset, BaseDataset)
        self.assertEqual(dataset.vocab_size, len(set(self.test_text)))
        self.assertEqual(len(dataset.data), len(self.test_text))
        self.assertEqual(dataset.block_size, self.block_size)
        self.assertEqual(dataset.char_to_idx['a'], 0)
        self.assertEqual(dataset.idx_to_char[0], 'a')

    def test_len(self):
        """Test the __len__ method."""
        # Use direct args
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size)
        expected_len = len(self.test_text) - self.block_size
        self.assertEqual(len(dataset), expected_len)

    def test_getitem(self):
        """Test the __getitem__ method."""
        # Use direct args
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size)
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
        # Use direct args
        dataset = CharDataset(file_path=self.temp_file_path, block_size=self.block_size)
        indices = [0, 1, 2, 3, 4] # Corresponds to 'abcde'
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
        
        # Use direct args
        dataset = CharDataset(file_path=empty_f_path, block_size=self.block_size)
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.vocab_size, 0)
        self.assertEqual(len(dataset.data), 0)
        self.assertEqual(dataset.decode([0, 1]), "") # Decode empty
        
        os.remove(empty_f_path)

    def test_short_file(self):
        """Test initialization with file content shorter than block_size."""
        short_text = "abc"
        block_size = 5
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as short_f:
            short_f.write(short_text)
            short_f_path = short_f.name
        
        # Use direct args
        dataset = CharDataset(file_path=short_f_path, block_size=block_size)
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.vocab_size, len(set(short_text)))
        self.assertEqual(len(dataset.data), len(short_text))
        
        # __getitem__ should raise IndexError if len is 0
        with self.assertRaises(IndexError):
            _ = dataset[0]
            
        os.remove(short_f_path)

    def test_invalid_path(self):
        """Test initialization with an invalid file path."""
        # Use direct args
        with self.assertRaises(ValueError):
            CharDataset(file_path='non_existent_file.txt', block_size=self.block_size)

class TestDatasetFactory(unittest.TestCase):
    """Tests for the create_dataset_from_config factory function."""

    def setUp(self):
        """Create a temporary file for factory tests."""
        self.test_text = "factory test data"
        self.block_size = 4
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8')
        self.temp_file.write(self.test_text)
        self.temp_file.close()
        self.temp_file_path = self.temp_file.name

    def tearDown(self):
        """Remove the temporary file."""
        os.remove(self.temp_file_path)

    def test_factory_chardataset_fallback(self):
        """Test creating CharDataset using the fallback logic (no _target_)."""
        config = OmegaConf.create({
            'file_path': self.temp_file_path,
            'block_size': self.block_size,
            'format': 'character',
            'extra_param': 'test_value' # Add an extra param
        })
        dataset = create_dataset_from_config(config=config, split='train')
        self.assertIsInstance(dataset, CharDataset)
        self.assertEqual(len(dataset), len(self.test_text) - self.block_size)
        # Check specific params were set correctly
        self.assertEqual(dataset.file_path, self.temp_file_path)
        self.assertEqual(dataset.block_size, self.block_size)
        # Check that extra params ended up in the dataset's config dict
        self.assertIn('extra_param', dataset.config)
        self.assertEqual(dataset.config['extra_param'], 'test_value')
        # Remove broad config comparison: self.assertEqual(dataset.config, config)

    def test_factory_hydra_instantiation(self):
        """Test creating a dataset using Hydra's _target_."""
        config = OmegaConf.create({
            '_target_': 'src.data.dataset.CharDataset',
            'file_path': self.temp_file_path,
            'block_size': self.block_size,
            'extra_param': 'test_value2' # Add extra param
        })
        dataset = create_dataset_from_config(config=config, split='train')
        self.assertIsInstance(dataset, CharDataset)
        self.assertEqual(len(dataset), len(self.test_text) - self.block_size)
        # Check main attributes directly
        self.assertEqual(dataset.file_path, self.temp_file_path)
        self.assertEqual(dataset.block_size, self.block_size)
        # Check extra kwargs stored in config
        self.assertIn('extra_param', dataset.config)
        self.assertEqual(dataset.config['extra_param'], 'test_value2')
        # Check split added by factory (optional, depends on factory implementation)
        # self.assertIn('split', dataset.config)
        # self.assertEqual(dataset.config['split'], 'train')

    def test_factory_unknown_format_no_target(self):
        """Test error when format is unknown and _target_ is missing."""
        config = OmegaConf.create({
            'file_path': self.temp_file_path,
            'block_size': self.block_size,
            'format': 'unknown_format' # Unknown format
        })
        with self.assertRaises(ValueError) as cm:
            create_dataset_from_config(config=config, split='train')
        self.assertIn("Unsupported dataset configuration", str(cm.exception))
        self.assertIn("format 'unknown_format'", str(cm.exception))

    def test_factory_invalid_target(self):
        """Test error when _target_ points to a non-existent class."""
        config = OmegaConf.create({
            '_target_': 'non_existent.module.NonExistentClass',
            'file_path': self.temp_file_path
        })
        with self.assertRaises(ValueError) as cm:
            create_dataset_from_config(config=config, split='train')
        self.assertIn("Could not instantiate dataset from config using _target_", str(cm.exception))

    def test_factory_chardataset_missing_path(self):
        """Test error when using fallback but path is missing."""
        config = OmegaConf.create({
            # 'file_path': self.temp_file_path, # Missing path
            'block_size': self.block_size,
            'format': 'character'
        })
        with self.assertRaises(ValueError) as cm:
            create_dataset_from_config(config=config, split='train')
        # Check factory fallback validation message
        self.assertIn("Fallback for CharDataset requires 'file_path' and 'block_size' in config.", str(cm.exception))

if __name__ == '__main__':
    unittest.main() 