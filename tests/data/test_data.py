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
import pytest # Import pytest
# Import Hydra exception for checking
from hydra.errors import InstantiationException
from hydra.utils import instantiate
import hydra # Ensure hydra is imported
import logging
import pickle
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from base and new factory location
from craft.data.base import BaseDataset 
from craft.data.factory import create_dataset, prepare_dataloaders_from_config
from craft.data.dataset import PickledDataset, TextDataset # Add TextDataset if used
from craft.data.tokenizers.base import BaseTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging for tests (optional)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use pytest fixture instead of setUp/tearDown
@pytest.fixture
def pickled_dataset_setup(tmp_path):
    """Sets up temporary files and data for PickledDataset tests."""
    token_ids = list(range(20))
    block_size = 5
    file_path = tmp_path / "test_data.pkl"
    vocab_path = tmp_path / "vocab.json"

    # Save ONLY token list to pickle
    with open(file_path, "wb") as f:
        pickle.dump(token_ids, f)

    # Save external vocab file
    idx_to_char = {i: chr(ord('a')+i) for i in range(26)}
    vocab_data = {
        'vocab_size': 50, 
        'idx_to_char': {str(k): v for k, v in idx_to_char.items()}
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)

    return file_path, vocab_path, block_size, token_ids, idx_to_char

# class TestPickledDataset(unittest.TestCase):
class TestPickledDataset:
    """Tests for the PickledDataset class (pytest style)."""

    # def setUp(self):
    #     """Set up test fixtures."""
    #     # ... removed, handled by fixture ...

    def test_initialization(self, pickled_dataset_setup):
        """Test basic initialization."""
        file_path, vocab_path, block_size, expected_tokens, idx_to_char = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(file_path), 
            block_size=block_size, 
            vocab_path=str(vocab_path)
        )
        assert dataset.block_size == block_size
        assert dataset.file_path == str(file_path)
        assert dataset._vocab_path == str(vocab_path)
        # Compare loaded tensor to expected list
        assert torch.equal(dataset.token_ids, torch.tensor(expected_tokens, dtype=torch.long))

    def test_len(self, pickled_dataset_setup):
        """Test __len__ method."""
        file_path, vocab_path, block_size, expected_tokens, _ = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(file_path), 
            block_size=block_size, 
            vocab_path=str(vocab_path)
        )
        expected_len = max(0, len(expected_tokens) - block_size)
        assert len(dataset) == expected_len

    def test_getitem(self, pickled_dataset_setup):
        """Test __getitem__ method."""
        file_path, vocab_path, block_size, expected_tokens, idx_to_char = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(file_path), 
            block_size=block_size, 
            vocab_path=str(vocab_path)
        )
        idx = 0
        x, y = dataset[idx]
        expected_x = torch.tensor(expected_tokens[idx : idx + block_size], dtype=torch.long)
        expected_y = torch.tensor(expected_tokens[idx + 1 : idx + block_size + 1], dtype=torch.long)
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)
        last_idx = len(dataset) - 1
        if last_idx >= 0:
             x, y = dataset[last_idx]
             expected_x = torch.tensor(expected_tokens[last_idx : last_idx + block_size], dtype=torch.long)
             expected_y = torch.tensor(expected_tokens[last_idx + 1 : last_idx + block_size + 1], dtype=torch.long)
             assert torch.equal(x, expected_x)
             assert torch.equal(y, expected_y)

    def test_decode(self, pickled_dataset_setup):
        """Test decode method."""
        file_path, vocab_path, block_size, _, idx_to_char = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(file_path), 
            block_size=block_size, 
            vocab_path=str(vocab_path)
        )
        ids = [0, 1, 2]
        expected_str = "".join(idx_to_char.get(i, '?') for i in ids)
        assert dataset.decode(ids) == expected_str
        assert dataset.decode(torch.tensor(ids)) == expected_str

    def test_empty_file(self, tmp_path):
        """Test initialization with empty file."""
        file_path = tmp_path / "empty.pkl"
        # Save empty list
        with open(file_path, "wb") as f:
            pickle.dump([], f)
        dataset = PickledDataset(str(file_path), block_size=5)
        assert len(dataset.token_ids) == 0
        assert len(dataset) == 0

    def test_short_file(self, tmp_path):
        """Test initialization with file content shorter than block_size."""
        file_path = tmp_path / "short_data.pkl"
        block_size = 5
        short_data = [0, 1, 2]
        with open(file_path, "wb") as f:
            pickle.dump(short_data, f)
        dataset = PickledDataset(str(file_path), block_size)
        assert len(dataset) == 0  # No complete blocks available
        with pytest.raises(IndexError):
            dataset[0]  # Should raise IndexError as no complete blocks exist

    def test_invalid_file_path(self, tmp_path):
        """Test initialization with an invalid file path but valid vocab path."""
        with pytest.raises(FileNotFoundError):
            PickledDataset(
                file_path=str(tmp_path / "non_existent.pkl"), 
                block_size=5, 
                vocab_path=str(tmp_path / "vocab.json")
            )

    def test_invalid_vocab_path(self, pickled_dataset_setup):
        """Test initialization with valid file path but invalid vocab path."""
        file_path, _, block_size, _, _ = pickled_dataset_setup
        dataset = PickledDataset(str(file_path), block_size, "non_existent.json")
        metadata = dataset.get_metadata()
        assert metadata == {}
        assert dataset.vocab_size is None

# Fixture for TestDatasetFactory
@pytest.fixture(scope="class")
def dataset_factory_setup(tmp_path_factory):
    """Sets up temporary files and configs for TestDatasetFactory."""
    # Use getbasetemp() to get a consistent base temporary directory for the class scope
    tmp_path = tmp_path_factory.getbasetemp()
    
    # Dummy data
    text = "hello world!"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    token_ids = np.array([char_to_idx[ch] for ch in text], dtype=np.uint16)
    
    # Create dummy data files and vocab
    train_file_path = tmp_path / "train.pkl"
    vocab_file_path = tmp_path / "vocab.json"
    
    # Save ONLY the token array to the pickle file
    with open(train_file_path, 'wb') as f:
        pickle.dump(token_ids, f)
        
    # Save vocab data (external)
    vocab_data = {
        'char_to_idx': char_to_idx,
        'idx_to_char': {str(k): v for k, v in idx_to_char.items()}, # JSON needs string keys
        'vocab_size': len(chars)
    }
    with open(vocab_file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=4)

    base_data_config = { 
        "data": {
            "format": "character",
            "split_ratios": [0.7, 0.15, 0.15]
        }
    }
    split_config = {
        "file_path": str(train_file_path),
        "vocab_path": str(vocab_file_path), 
        "block_size": 5,
    }

    return {
        "cwd": str(tmp_path),
        "base_data_config": base_data_config,
        "split_config": split_config, 
        "chars": chars,
        "vocab_size": len(chars)
    }

# class TestDatasetFactory(unittest.TestCase):
class TestDatasetFactory:
    """Tests focused on the dataset factory function(s)."""

    # def setUp(self):
    #     """Set up test fixtures."""
    #     # ... removed, handled by fixture ...

    # def tearDown(self):
    #     """Clean up test fixtures."""
    #     # ... removed, handled by tmp_path ...

    def test_factory_hydra_instantiation(self, dataset_factory_setup):
        """Test creating a dataset using Hydra's _target_."""
        setup_data = dataset_factory_setup
        split_conf = OmegaConf.create(setup_data["split_config"])
        target_conf = OmegaConf.create({
            '_target_': 'craft.data.dataset.PickledDataset'
        })
        split_config_with_target = OmegaConf.merge(split_conf, target_conf)

        # Pass the DictConfig directly
        dataset = create_dataset(split_config_with_target)

        assert isinstance(dataset, PickledDataset)
        assert dataset.block_size == setup_data["split_config"]["block_size"]
        assert dataset._vocab_path == setup_data["split_config"]["vocab_path"]
        assert dataset.file_path == setup_data["split_config"]["file_path"]
        # Check if vocab loaded correctly via vocab_path
        assert dataset.vocab_size == setup_data["vocab_size"]

    def test_factory_missing_target_fallback(self, dataset_factory_setup):
        """Test fallback/error when _target_ is missing."""
        setup_data = dataset_factory_setup
        config_without_target = setup_data["split_config"].copy()

        with pytest.raises(ValueError, match="must be a dictionary or DictConfig with a '_target_' key"):
             # Pass dict directly, factory should handle or raise
            create_dataset(config_without_target)

    def test_factory_invalid_target(self, dataset_factory_setup):
        """Test error when _target_ points to an invalid class."""
        setup_data = dataset_factory_setup
        split_conf = OmegaConf.create(setup_data["split_config"])
        target_conf = OmegaConf.create({
            '_target_': 'non.existent.Class'
        })
        split_config_with_target = OmegaConf.merge(split_conf, target_conf)

        # Expect Hydra's InstantiationException wrapped in ValueError by create_dataset
        with pytest.raises(ValueError, match="Could not create dataset from config"):
            # Pass DictConfig directly
            create_dataset(split_config_with_target)

    def test_factory_missing_required_arg(self, dataset_factory_setup):
        """Test error when required arguments are missing from config."""
        setup_data = dataset_factory_setup
        incomplete_split_config = setup_data["split_config"].copy()
        del incomplete_split_config['block_size']

        split_conf = OmegaConf.create(incomplete_split_config)
        target_conf = OmegaConf.create({
            '_target_': 'craft.data.dataset.PickledDataset'
        })
        split_config_with_target = OmegaConf.merge(split_conf, target_conf)
    
        # Expect InstantiationException (missing arg) wrapped in ValueError by create_dataset
        with pytest.raises(ValueError, match="Could not create dataset from config"):
            # Pass DictConfig directly
            create_dataset(split_config_with_target)

# if __name__ == '__main__':
#     unittest.main() # Remove unittest runner 