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
import logging
import pickle
import numpy as np
import shutil

# Modules to test
from src.data.base import BaseDataset, create_dataset_from_config
from src.data.dataset import PickledDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging for tests (optional)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestPickledDataset(unittest.TestCase):
    """Tests for the PickledDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.block_size = 5
        self.chars = sorted(list(set("Hello world!")))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.token_ids = [self.char_to_idx[ch] for ch in "Hello world!"]
        
        # Create test data file
        self.file_path = os.path.join(self.temp_dir, "test_data.pkl")
        data_dict = {
            "text": "Hello world!",
            "chars": self.chars,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": len(self.chars),
            "token_ids": np.array(self.token_ids, dtype=np.uint16),
            "tokenizer_type": "character"
        }
        with open(self.file_path, "wb") as f:
            pickle.dump(data_dict, f)
        
        # Create test vocab file
        self.vocab_path = os.path.join(self.temp_dir, "vocab.json")
        vocab_data = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": len(self.chars)
        }
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)

    def test_initialization(self):
        """Test basic initialization."""
        dataset = PickledDataset(file_path=self.file_path, block_size=self.block_size, vocab_path=self.vocab_path)
        self.assertEqual(dataset.block_size, self.block_size)
        self.assertEqual(dataset.vocab_size, len(self.chars))
        self.assertEqual(dataset.char_to_idx, self.char_to_idx)
        self.assertEqual(dataset.idx_to_char, self.idx_to_char)

    def test_len(self):
        """Test __len__ method."""
        dataset = PickledDataset(file_path=self.file_path, block_size=self.block_size, vocab_path=self.vocab_path)
        expected_len = max(0, len(self.token_ids) - self.block_size)
        self.assertEqual(len(dataset), expected_len)

    def test_getitem(self):
        """Test __getitem__ method."""
        dataset = PickledDataset(file_path=self.file_path, block_size=self.block_size, vocab_path=self.vocab_path)
        x, y = dataset[0]
        self.assertEqual(len(x), self.block_size)
        self.assertEqual(len(y), self.block_size)
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertTrue(isinstance(y, torch.Tensor))

    def test_decode(self):
        """Test decode method."""
        dataset = PickledDataset(file_path=self.file_path, block_size=self.block_size, vocab_path=self.vocab_path)
        tokens = torch.tensor([self.char_to_idx[ch] for ch in "Hello"])
        decoded = dataset.decode(tokens)
        self.assertEqual(decoded, "Hello")

    def test_empty_file(self):
        """Test initialization with empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.pkl")
        empty_data = {
            "text": "",
            "chars": [],
            "char_to_idx": {},
            "idx_to_char": {},
            "vocab_size": 0,
            "token_ids": np.array([], dtype=np.uint16),
            "tokenizer_type": "character"
        }
        with open(empty_file, "wb") as f:
            pickle.dump(empty_data, f)
        
        dataset = PickledDataset(file_path=empty_file, block_size=self.block_size, vocab_path=self.vocab_path)
        self.assertEqual(len(dataset), 0)

    def test_short_file(self):
        """Test initialization with file content shorter than block_size."""
        short_text = "abc"
        chars = sorted(list(set(short_text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        token_ids = [char_to_idx[ch] for ch in short_text]
        
        # Create short data file
        short_file = os.path.join(self.temp_dir, "short_data.pkl")
        short_data = {
            "text": short_text,
            "chars": chars,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab_size": len(chars),
            "token_ids": np.array(token_ids, dtype=np.uint16),
            "tokenizer_type": "character"
        }
        with open(short_file, "wb") as f:
            pickle.dump(short_data, f)
        
        # Create short vocab file
        short_vocab = os.path.join(self.temp_dir, "short_vocab.json")
        vocab_data = {
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab_size": len(chars)
        }
        with open(short_vocab, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)
        
        # Test with short data
        dataset = PickledDataset(file_path=short_file, block_size=self.block_size, vocab_path=short_vocab)
        self.assertEqual(len(dataset), 0)  # No complete blocks available
        with self.assertRaises(IndexError):
            dataset[0]  # Should raise IndexError as no complete blocks exist

    def test_invalid_file_path(self):
        """Test initialization with an invalid file path but valid vocab path."""
        with self.assertRaises(FileNotFoundError):
             # Provide valid dummy vocab, but invalid file path
            PickledDataset(file_path='non_existent_file.txt', block_size=self.block_size, vocab_path=self.vocab_path)

    def test_invalid_vocab_path(self):
        """Test initialization with valid file path but invalid vocab path."""
        with self.assertRaises(FileNotFoundError):
             # Provide valid file path, but invalid vocab path
            PickledDataset(file_path=self.file_path, block_size=self.block_size, vocab_path='non_existent_vocab.json')

class TestDatasetFactory(unittest.TestCase):
    """Test dataset factory functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.block_size = 5
        self.chars = sorted(list(set("Hello world!")))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.token_ids = [self.char_to_idx[ch] for ch in "Hello world!"]
        
        # Create test data file
        self.file_path = os.path.join(self.test_dir, "test_data.pkl")
        data_dict = {
            "text": "Hello world!",
            "chars": self.chars,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": len(self.chars),
            "token_ids": np.array(self.token_ids, dtype=np.uint16),
            "tokenizer_type": "character"
        }
        with open(self.file_path, "wb") as f:
            pickle.dump(data_dict, f)
        
        # Create test vocab file
        self.vocab_path = os.path.join(self.test_dir, "vocab.json")
        vocab_data = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": len(self.chars)
        }
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)

        # Set up base config and split config
        self.base_data_config = {
            'data': {
                'format': 'character',
                'split_ratios': [0.7, 0.15, 0.15]
            }
        }
        self.split_config = {
            'block_size': self.block_size,
            'file_path': self.file_path,
            'vocab_path': self.vocab_path
        }
        self.cwd = os.getcwd()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_factory_hydra_instantiation(self):
        """Test creating a dataset using Hydra's _target_."""
        # Add _target_ for direct instantiation
        split_config_with_target = OmegaConf.merge(self.split_config, {
            '_target_': 'src.data.dataset.PickledDataset',
            'file_path': self.file_path,
            'block_size': self.block_size,
            'vocab_path': self.vocab_path
        })
        
        dataset = create_dataset_from_config(self.base_data_config, split_config_with_target, self.cwd)
        self.assertIsInstance(dataset, PickledDataset)
        self.assertEqual(dataset.block_size, self.block_size)
        self.assertEqual(dataset.vocab_size, len(self.chars))

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
        incomplete_split_config.pop('block_size', None)  # Use pop instead of del
        with self.assertRaises(ValueError):
            create_dataset_from_config(self.base_data_config, incomplete_split_config, self.cwd)

if __name__ == '__main__':
    unittest.main() 