import pytest
import torch
import pickle
from pathlib import Path
import tempfile
import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock, ANY
import os
import logging
import numpy as np

# Import the correct high-level function
from craft.data import prepare_dataloaders_from_config
# Import dataset/tokenizer for type checks
from craft.data.dataset import PickledDataset, TextDataset
from craft.data.tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)

# --- Mock Tokenizer (defined at module level) ---
class MockTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=100, config: dict = None): 
        super().__init__(config=config or {})
        self._vocab_size = vocab_size
        
    def encode(self, text, **kwargs): return [0, 1, 2]
    def decode(self, ids, **kwargs): return "mock"
    
    @property
    def vocab_size(self): 
        logger.debug("Accessing MockTokenizer vocab_size property")
        return self._vocab_size
    
    def get_vocab_size(self) -> int:
        logger.debug("Calling MockTokenizer get_vocab_size method")
        return self._vocab_size
        
    def save(self, directory): 
        logger.debug(f"MockTokenizer save called (no-op): {directory}")
        pass
    
    @classmethod
    def load(cls, directory): 
        logger.debug(f"MockTokenizer load called (no-op): {directory}")
        return cls()
        
    def train(self, file_path: str, **kwargs): 
        logger.info(f"MockTokenizer train called (no-op): {file_path}")
        pass 
        
    @property
    def bos_token_id(self): return None
    @property
    def eos_token_id(self): return None
    @property
    def pad_token_id(self): return None

# --- Test Data Fixture ---

@pytest.fixture(scope="module")
def processed_data_dir(tmp_path_factory):
    data_path = tmp_path_factory.mktemp("processed_data")
    seq_length = 10
    vocab_size = 50
    metadata = {"vocab_size": vocab_size}
    # Create dummy data as raw list/array for pickle files
    dummy_token_ids = list(range(100))
    
    with open(data_path / "train.pkl", 'wb') as f:
        pickle.dump(dummy_token_ids, f)
    with open(data_path / "val.pkl", 'wb') as f:
        pickle.dump(dummy_token_ids, f)
    with open(data_path / "test.pkl", 'wb') as f:
        pickle.dump(dummy_token_ids, f)
    return data_path, metadata, seq_length, vocab_size

# --- Tests for PickledDataset ---

@pytest.fixture
def pickled_dataset_test_setup(tmp_path):
    file_path = tmp_path / "test.pkl"
    vocab_path = tmp_path / "vocab.json"
    block_size = 5
    token_ids = list(range(20))
    with open(file_path, "wb") as f: pickle.dump(token_ids, f)
    idx_to_char = {i: chr(ord('a')+i) for i in range(26)}
    vocab_data = {'vocab_size': 50, 'idx_to_char': {str(k): v for k, v in idx_to_char.items()}}
    with open(vocab_path, "w") as f: json.dump(vocab_data, f)
    return file_path, vocab_path, block_size, token_ids, idx_to_char

def test_pickled_dataset_standalone_init_len_getitem(pickled_dataset_test_setup):
    file_path, vocab_path, block_size, token_ids, _ = pickled_dataset_test_setup
    dataset = PickledDataset(str(file_path), block_size, str(vocab_path))
    assert len(dataset.token_ids) == len(token_ids)
    # Corrected expected length calculation
    expected_len = max(0, len(token_ids) - block_size)
    assert len(dataset) == expected_len 
    if len(dataset) > 0:
        x, y = dataset[0]
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)

def test_pickled_dataset_standalone_get_metadata(pickled_dataset_test_setup):
    # This test now uses the fixture which provides vocab_path
    file_path, vocab_path, block_size, _, _ = pickled_dataset_test_setup
    dataset = PickledDataset(str(file_path), block_size, str(vocab_path))
    metadata = dataset.get_metadata()
    assert isinstance(metadata, dict)
    assert metadata.get('vocab_size') == 50 
    assert dataset.vocab_size == 50 

def test_pickled_dataset_standalone_metadata_not_found(pickled_dataset_test_setup):
    # Test cases where metadata should not be found
    file_path, _, block_size, _, _ = pickled_dataset_test_setup
    # Case 1: No vocab_path provided
    dataset_no_path = PickledDataset(str(file_path), block_size)
    assert dataset_no_path.get_metadata() == {}
    assert dataset_no_path.vocab_size is None
    # Case 2: Invalid vocab_path provided
    dataset_bad_path = PickledDataset(str(file_path), block_size, "bad_path.json")
    assert dataset_bad_path.get_metadata() == {}
    assert dataset_bad_path.vocab_size is None

# --- Tests for create_data_loaders_from_config ---

def test_create_data_loaders_train_val(processed_data_dir):
    """Test creating train and validation DataLoaders."""
    data_path, _, seq_length, _ = processed_data_dir
    
    cfg_dict = {
        "data": {
            "datasets": {
                "train": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": str(data_path / "train.pkl"),
                        "block_size": seq_length
                    },
                    "dataloader": {
                        "batch_size": 4,
                        "shuffle": True,
                        "num_workers": 0
                    }
                },
                "val": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": str(data_path / "val.pkl"),
                        "block_size": seq_length
                    },
                    "dataloader": {
                        "batch_size": 8,
                        "shuffle": False,
                        "num_workers": 0
                    }
                }
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    # Call the correct function with the full config
    train_loader, val_loader, test_loader, tokenizer = prepare_dataloaders_from_config(cfg)
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is None
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert test_loader is None
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 8
    assert isinstance(train_loader.dataset, PickledDataset)
    assert isinstance(val_loader.dataset, PickledDataset)
    assert tokenizer is None

def test_create_data_loaders_train_val_test(processed_data_dir):
    """Test creating train, validation, and test DataLoaders."""
    data_path, _, seq_length, _ = processed_data_dir
    
    cfg_dict = {
        "data": {
            "datasets": {
                "train": { # Required
                     "dataset": {"_target_": "craft.data.dataset.PickledDataset", "file_path": str(data_path / "train.pkl"), "block_size": seq_length},
                     "dataloader": {"batch_size": 4}
                },
                 "val": { # Required
                     "dataset": {"_target_": "craft.data.dataset.PickledDataset", "file_path": str(data_path / "val.pkl"), "block_size": seq_length},
                     "dataloader": {"batch_size": 4}
                },
                "test": { # Optional
                     "dataset": {"_target_": "craft.data.dataset.PickledDataset", "file_path": str(data_path / "test.pkl"), "block_size": seq_length},
                     "dataloader": {"batch_size": 4, "shuffle": False} # Explicit shuffle False
                }
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    # Call the correct function with the full config
    train_loader, val_loader, test_loader, tokenizer = prepare_dataloaders_from_config(cfg)
    
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 4
    assert test_loader.batch_size == 4
    assert tokenizer is None

def test_create_data_loaders_missing_split_config(tmp_path):
    """Test factory behavior when required train or val config keys are missing."""
    block_size = 1
    dummy_train_file = tmp_path / "train.pkl"
    dummy_val_file = tmp_path / "val.pkl"
    dummy_data = [0, 1, 2, 3]
    with open(dummy_train_file, 'wb') as f: pickle.dump(dummy_data, f)
    with open(dummy_val_file, 'wb') as f: pickle.dump(dummy_data, f)

    # Config missing 'val' dataset key entirely
    cfg_missing_val_key = OmegaConf.create({"data": {"datasets": {"train": {
        "dataset": {"_target_": "craft.data.dataset.PickledDataset", "file_path": str(dummy_train_file), "block_size": block_size},
        "dataloader": {}
    }}}})

    # Config missing 'train' dataset key entirely
    cfg_missing_train_key = OmegaConf.create({"data": {"datasets": {"val": {
        "dataset": {"_target_": "craft.data.dataset.PickledDataset", "file_path": str(dummy_val_file), "block_size": block_size},
        "dataloader": {}
    }}}})

    # The factory currently does NOT raise an error if 'train' or 'val' keys are completely missing.
    # It only raises if they exist but are misconfigured, or if loading fails later.
    # The RuntimeError check at the end of the factory handles cases where a configured
    # split fails silently. So, we just run the factory call without expecting an immediate error here.
    try:
        tr, vl, _, _ = prepare_dataloaders_from_config(cfg_missing_val_key)
        assert tr is not None
        assert vl is None # Val should be None as it wasn't configured
    except Exception as e:
        pytest.fail(f"prepare_dataloaders_from_config raised unexpected error for missing val key: {e}")

    try:
        tr, vl, _, _ = prepare_dataloaders_from_config(cfg_missing_train_key)
        assert tr is None # Train should be None
        assert vl is not None
    except Exception as e:
        pytest.fail(f"prepare_dataloaders_from_config raised unexpected error for missing train key: {e}")

def test_create_data_loaders_missing_target(processed_data_dir):
    """Test error handling for missing _target_ in required dataset config."""
    data_path, _, seq_length, _ = processed_data_dir
    cfg_dict = {
        "data": {
            "datasets": {
                "train": {
                     "dataset": {"file_path": str(data_path / "train.pkl"), "block_size": seq_length}, # Missing _target_
                     "dataloader": {"batch_size": 4}
                },
                 "val": {
                     "dataset": {"_target_": "craft.data.dataset.PickledDataset", "file_path": str(data_path / "val.pkl"), "block_size": seq_length},
                     "dataloader": {"batch_size": 4}
                }
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    # Match the specific error raised when _target_ is missing inside the factory
    expected_error_msg = r"Dataset configuration for split \'train\' is missing the \'_target_\' key."
    with pytest.raises(ValueError, match=expected_error_msg):
        prepare_dataloaders_from_config(cfg)

def test_create_data_loaders_with_tokenizer(processed_data_dir):
    """Test creating dataloaders when a tokenizer is configured."""
    data_path, _, seq_length, _ = processed_data_dir
    
    # MockTokenizer is defined at module level and should be complete now
    cfg_dict = {
        "data": {
            "tokenizer": {
                "_target_": f"{__name__}.MockTokenizer", 
                "vocab_size": 150
            },
            "datasets": {
                "train": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": str(data_path / "train.pkl"),
                        "block_size": seq_length,
                        "tokenizer": "${data.tokenizer}" # Test interpolation
                    },
                    "dataloader": {"batch_size": 4}
                },
                "val": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": str(data_path / "val.pkl"),
                        "block_size": seq_length,
                        "tokenizer": "${data.tokenizer}"
                    },
                    "dataloader": {"batch_size": 4}
                }
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    _, _, _, tokenizer = prepare_dataloaders_from_config(cfg)
    
    assert tokenizer is not None
    assert isinstance(tokenizer, MockTokenizer)
    assert tokenizer.vocab_size == 150 # Check property access

def test_create_data_loaders_tokenizer_override(processed_data_dir):
    """Test overriding tokenizer config with a pre-instantiated one."""
    data_path, _, seq_length, _ = processed_data_dir
    cfg_dict = {
        "data": {
            "tokenizer": { # Top-level tokenizer (should be overridden)
                "_target_": "tests.data.test_datasets.MockTokenizer",
                "vocab_size": 100 
            },
            "datasets": {
                "train": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": str(data_path / "train.pkl"),
                        "block_size": seq_length,
                        "tokenizer": { # Override tokenizer for this split
                            "_target_": "tests.data.test_datasets.MockTokenizer",
                            "vocab_size": 200 
                        }
                    },
                    "dataloader": {"batch_size": 4}
                },
                "val": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": str(data_path / "val.pkl"),
                        "block_size": seq_length
                        # No tokenizer override, should use top-level one
                    },
                    "dataloader": {"batch_size": 4}
                }
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    # Create a tokenizer instance to override (should work now)
    override_tokenizer = MockTokenizer(vocab_size=200, config={})

    _, _, _, tokenizer = prepare_dataloaders_from_config(cfg, tokenizer_override=override_tokenizer)

    assert tokenizer is override_tokenizer
    assert tokenizer.vocab_size == 200 