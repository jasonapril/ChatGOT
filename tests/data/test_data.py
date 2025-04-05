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
# from hydra.errors import InstantiationException # This was commented out
import hydra # Ensure hydra is imported
import logging
import pickle
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Corrected import
from craft.data.base import BaseDataset, create_dataset_from_config
from craft.data.dataset import PickledDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging for tests (optional)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Use pytest fixture instead of setUp/tearDown
@pytest.fixture
def pickled_dataset_setup(tmp_path):
    """Sets up temporary files and data for PickledDataset tests."""
    block_size = 5
    text = "Hello world!"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    token_ids = [char_to_idx[ch] for ch in text]
    
    # Create test data file
    file_path = tmp_path / "test_data.pkl"
    data_dict = {
        "text": text,
        "chars": chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(chars),
        "token_ids": np.array(token_ids, dtype=np.uint16),
        "tokenizer_type": "character"
    }
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f)
    
    # Create test vocab file
    vocab_path = tmp_path / "vocab.json"
    vocab_data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(chars)
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f)

    # Create empty file for empty test
    empty_file_path = tmp_path / "empty.pkl"
    empty_data = {
        "text": "", "chars": [], "char_to_idx": {}, "idx_to_char": {}, 
        "vocab_size": 0, "token_ids": np.array([], dtype=np.uint16), 
        "tokenizer_type": "character"
    }
    with open(empty_file_path, "wb") as f:
        pickle.dump(empty_data, f)

    # Create short file for short test
    short_text = "abc"
    short_chars = sorted(list(set(short_text)))
    short_char_to_idx = {ch: i for i, ch in enumerate(short_chars)}
    short_idx_to_char = {i: ch for i, ch in enumerate(short_chars)}
    short_token_ids = [short_char_to_idx[ch] for ch in short_text]
    short_file_path = tmp_path / "short_data.pkl"
    short_data_dict = {
        "text": short_text, "chars": short_chars, "char_to_idx": short_char_to_idx, 
        "idx_to_char": short_idx_to_char, "vocab_size": len(short_chars),
        "token_ids": np.array(short_token_ids, dtype=np.uint16), "tokenizer_type": "character"
    }
    with open(short_file_path, "wb") as f:
        pickle.dump(short_data_dict, f)
    short_vocab_path = tmp_path / "short_vocab.json"
    short_vocab_data = {
        "char_to_idx": short_char_to_idx, "idx_to_char": short_idx_to_char, 
        "vocab_size": len(short_chars)
    }
    with open(short_vocab_path, "w", encoding="utf-8") as f:
        json.dump(short_vocab_data, f)
        
    # Return data needed by tests
    return {
        "tmp_path": tmp_path,
        "block_size": block_size,
        "chars": chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "token_ids": token_ids,
        "file_path": file_path,
        "vocab_path": vocab_path,
        "empty_file_path": empty_file_path,
        "short_file_path": short_file_path,
        "short_vocab_path": short_vocab_path
    }

# class TestPickledDataset(unittest.TestCase):
class TestPickledDataset:
    """Tests for the PickledDataset class (pytest style)."""

    # def setUp(self):
    #     """Set up test fixtures."""
    #     # ... removed, handled by fixture ...

    def test_initialization(self, pickled_dataset_setup):
        """Test basic initialization."""
        setup_data = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(setup_data["file_path"]), 
            block_size=setup_data["block_size"], 
            vocab_path=str(setup_data["vocab_path"])
        )
        assert dataset.block_size == setup_data["block_size"]
        assert dataset.vocab_size == len(setup_data["chars"])
        assert dataset.char_to_idx == setup_data["char_to_idx"]
        assert dataset.idx_to_char == setup_data["idx_to_char"]

    def test_len(self, pickled_dataset_setup):
        """Test __len__ method."""
        setup_data = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(setup_data["file_path"]), 
            block_size=setup_data["block_size"], 
            vocab_path=str(setup_data["vocab_path"])
        )
        expected_len = max(0, len(setup_data["token_ids"]) - setup_data["block_size"])
        assert len(dataset) == expected_len

    def test_getitem(self, pickled_dataset_setup):
        """Test __getitem__ method."""
        setup_data = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(setup_data["file_path"]), 
            block_size=setup_data["block_size"], 
            vocab_path=str(setup_data["vocab_path"])
        )
        x, y = dataset[0]
        assert len(x) == setup_data["block_size"]
        assert len(y) == setup_data["block_size"]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_decode(self, pickled_dataset_setup):
        """Test decode method."""
        setup_data = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(setup_data["file_path"]), 
            block_size=setup_data["block_size"], 
            vocab_path=str(setup_data["vocab_path"])
        )
        tokens = torch.tensor([setup_data["char_to_idx"][ch] for ch in "Hello"])
        decoded = dataset.decode(tokens)
        assert decoded == "Hello"

    def test_empty_file(self, pickled_dataset_setup):
        """Test initialization with empty file."""
        setup_data = pickled_dataset_setup
        # Vocab path from main setup is fine, just use the empty data file
        dataset = PickledDataset(
            file_path=str(setup_data["empty_file_path"]), 
            block_size=setup_data["block_size"], 
            vocab_path=str(setup_data["vocab_path"])
        )
        assert len(dataset) == 0

    def test_short_file(self, pickled_dataset_setup):
        """Test initialization with file content shorter than block_size."""
        setup_data = pickled_dataset_setup
        dataset = PickledDataset(
            file_path=str(setup_data["short_file_path"]), 
            block_size=setup_data["block_size"], 
            vocab_path=str(setup_data["short_vocab_path"])
        )
        assert len(dataset) == 0  # No complete blocks available
        with pytest.raises(IndexError):
            dataset[0]  # Should raise IndexError as no complete blocks exist

    def test_invalid_file_path(self, pickled_dataset_setup):
        """Test initialization with an invalid file path but valid vocab path."""
        setup_data = pickled_dataset_setup
        with pytest.raises(FileNotFoundError):
            PickledDataset(
                file_path='non_existent_file.txt', 
                block_size=setup_data["block_size"], 
                vocab_path=str(setup_data["vocab_path"])
            )

    def test_invalid_vocab_path(self, pickled_dataset_setup):
        """Test initialization with valid file path but invalid vocab path."""
        setup_data = pickled_dataset_setup
        with pytest.raises(FileNotFoundError):
            PickledDataset(
                file_path=str(setup_data["file_path"]), 
                block_size=setup_data["block_size"], 
                vocab_path='non_existent_vocab.json'
            )

# Fixture for TestDatasetFactory
@pytest.fixture
def dataset_factory_setup(tmp_path):
    """Sets up temporary files and configs for TestDatasetFactory."""
    block_size = 5
    text = "Hello world!"
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    token_ids = [char_to_idx[ch] for ch in text]
    
    # Create test data file
    file_path = tmp_path / "test_data.pkl"
    data_dict = {
        "text": text, "chars": chars, "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char, "vocab_size": len(chars),
        "token_ids": np.array(token_ids, dtype=np.uint16),
        "tokenizer_type": "character"
    }
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f)
    
    # Create test vocab file
    vocab_path = tmp_path / "vocab.json"
    vocab_data = {
        "char_to_idx": char_to_idx, "idx_to_char": idx_to_char, 
        "vocab_size": len(chars)
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f)

    # Base config and split config (as simple dicts)
    base_data_config = {
        'data': {
            'format': 'character', # This might be used by factory
            'split_ratios': [0.7, 0.15, 0.15]
        }
    }
    split_config = {
        'block_size': block_size,
        # Convert Path objects to strings for config
        'file_path': str(file_path),
        'vocab_path': str(vocab_path)
    }
    
    return {
        "tmp_path": tmp_path, # Though not strictly needed if files are passed
        "block_size": block_size,
        "chars": chars,
        "base_data_config": base_data_config,
        "split_config": split_config,
        "cwd": os.getcwd() # Factory might need CWD
    }

# class TestDatasetFactory(unittest.TestCase):
class TestDatasetFactory:
    """Test dataset factory functionality (pytest style)."""

    # def setUp(self):
    #     """Set up test fixtures."""
    #     # ... removed, handled by fixture ...

    # def tearDown(self):
    #     """Clean up test fixtures."""
    #     # ... removed, handled by tmp_path ...

    def test_factory_hydra_instantiation(self, dataset_factory_setup):
        """Test creating a dataset using Hydra's _target_."""
        setup_data = dataset_factory_setup
        
        # Add _target_ for direct instantiation
        # Merge expects OmegaConf DictConfigs, create them first
        split_conf = OmegaConf.create(setup_data["split_config"])
        target_conf = OmegaConf.create({
            '_target_': 'craft.data.dataset.PickledDataset'
            # Config args are already in split_conf
        })
        # Create the final config for the factory
        split_config_with_target = OmegaConf.merge(split_conf, target_conf)
        
        dataset = create_dataset_from_config(
            setup_data["base_data_config"], 
            split_config_with_target, 
            setup_data["cwd"]
        )
        assert isinstance(dataset, PickledDataset)
        assert dataset.block_size == setup_data["block_size"]
        assert dataset.vocab_size == len(setup_data["chars"])

    def test_factory_missing_target_fallback(self, dataset_factory_setup):
        """Test factory fallback/error when _target_ is missing."""
        setup_data = dataset_factory_setup
        # Assuming the factory *requires* _target_ now
        # Convert dicts to OmegaConf for consistency if factory expects it
        base_conf = OmegaConf.create(setup_data["base_data_config"])
        split_conf_no_target = OmegaConf.create(setup_data["split_config"])
        
        # Expect error because _target_ is missing
        with pytest.raises((ValueError, TypeError, NotImplementedError, KeyError)):
             create_dataset_from_config(base_conf, split_conf_no_target, setup_data["cwd"])

    def test_factory_invalid_target(self, dataset_factory_setup):
        """Test error when _target_ points to a non-existent class."""
        setup_data = dataset_factory_setup
        split_conf = OmegaConf.create(setup_data["split_config"])
        invalid_target_conf = OmegaConf.create({
            '_target_': 'craft.data.non_existent.NonExistentDataset'
        })
        split_config_invalid_target = OmegaConf.merge(split_conf, invalid_target_conf)
        
        # Expect some form of import or instantiation error (Hydra might wrap it)
        with pytest.raises(Exception): 
            create_dataset_from_config(
                setup_data["base_data_config"], 
                split_config_invalid_target, 
                setup_data["cwd"]
            )

    def test_factory_missing_required_arg(self, dataset_factory_setup):
        """Test error when required arguments are missing from config."""
        setup_data = dataset_factory_setup
        # Config missing 'block_size'
        incomplete_split_config = setup_data["split_config"].copy()
        del incomplete_split_config['block_size']
        
        split_conf = OmegaConf.create(incomplete_split_config)
        target_conf = OmegaConf.create({
            '_target_': 'craft.data.dataset.PickledDataset'
        })
        split_config_with_target = OmegaConf.merge(split_conf, target_conf)

        # Expect Hydra's InstantiationException wrapping the TypeError
        with pytest.raises(hydra.errors.InstantiationException):
            create_dataset_from_config(
                setup_data["base_data_config"], 
                split_config_with_target, 
                setup_data["cwd"]
            )

# if __name__ == '__main__':
#     unittest.main() # Remove unittest runner 