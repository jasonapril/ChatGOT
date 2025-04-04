import pytest
import torch
import pickle
from pathlib import Path
import tempfile
import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Assume src is importable, adjust path if necessary
from src.craft.data.dataset import PickledDataset
from src.craft.data.base import create_data_loaders_from_config

# --- Test Data Fixture ---

@pytest.fixture(scope="module")
def processed_data_dir():
    """Creates a temporary directory with dummy processed data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        
        # Sample data parameters
        vocab_size = 50
        seq_length = 10 # This is the block_size for PickledDataset
        num_sequences_train = 100
        num_sequences_val = 20
        num_sequences_test = 10
        
        # Create flat lists of token IDs
        # Total length needs to be > block_size for len(dataset) > 0
        total_tokens_train = num_sequences_train * (seq_length + 1) 
        total_tokens_val = num_sequences_val * (seq_length + 1)
        total_tokens_test = num_sequences_test * (seq_length + 1)

        train_token_ids = torch.randint(0, vocab_size, (total_tokens_train,), dtype=torch.long).tolist()
        val_token_ids = torch.randint(0, vocab_size, (total_tokens_val,), dtype=torch.long).tolist()
        test_token_ids = torch.randint(0, vocab_size, (total_tokens_test,), dtype=torch.long).tolist()
        
        # Metadata
        char_to_idx = {chr(i + 97): i for i in range(26)} # a-z
        idx_to_char = {i: chr(i + 97) for i in range(26)}
        # Store metadata compatible with PickledDataset loading
        metadata_for_pkl = {
             "char_to_idx": char_to_idx,
             "idx_to_char": {str(k): v for k, v in idx_to_char.items()}, # Keys must be strings in JSON
             "vocab_size": vocab_size 
        }
        metadata_for_json = metadata_for_pkl # Same content for json file

        # Save data files
        with open(data_path / "train.pkl", "wb") as f:
            # Save flat token_ids and metadata directly in pkl
            pickle.dump({"token_ids": train_token_ids, **metadata_for_pkl}, f)
        with open(data_path / "val.pkl", "wb") as f:
            pickle.dump({"token_ids": val_token_ids, **metadata_for_pkl}, f)
        with open(data_path / "test.pkl", "wb") as f:
            pickle.dump({"token_ids": test_token_ids, **metadata_for_pkl}, f)
        with open(data_path / "metadata.json", "w") as f:
            json.dump(metadata_for_json, f)
            
        # Yield path, expected metadata (with int keys for idx_to_char), block_size, and token counts
        yield data_path, {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char, "vocab_size": vocab_size}, seq_length, total_tokens_train

# --- Tests for PickledDataset ---

def test_pickled_dataset_init_len_getitem(processed_data_dir):
    """Test initialization, length, and item retrieval for PickledDataset."""
    data_path, _, seq_length, total_tokens_train = processed_data_dir
    train_pkl_path = data_path / "train.pkl"
    
    # Pass block_size (using seq_length from fixture)
    dataset = PickledDataset(str(train_pkl_path), block_size=seq_length)
    
    # Check length 
    expected_len = total_tokens_train - seq_length
    assert len(dataset) == expected_len
    assert len(dataset) > 0 # Ensure length is positive
    
    # Check getitem
    idx = 5
    x, y = dataset[idx]
    
    # Load original flat tokens for comparison
    with open(train_pkl_path, "rb") as f:
        original_tokens = pickle.load(f)["token_ids"]

    expected_x = torch.tensor(original_tokens[idx : idx + seq_length], dtype=torch.long)
    expected_y = torch.tensor(original_tokens[idx + 1: idx + 1 + seq_length], dtype=torch.long)

    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)
    assert x.shape == (seq_length,)
    assert y.shape == (seq_length,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_pickled_dataset_get_metadata(processed_data_dir):
    """Test metadata loading via attributes."""
    data_path, expected_metadata, seq_length = processed_data_dir[:3] # Only need first 3 yields
    train_pkl_path = data_path / "train.pkl"
    
    # Pass block_size
    dataset = PickledDataset(str(train_pkl_path), block_size=seq_length)
    
    # Check attributes
    assert dataset.vocab_size == expected_metadata["vocab_size"]
    assert dataset.char_to_idx == expected_metadata["char_to_idx"]
    # idx_to_char keys are loaded as ints by PickledDataset
    assert dataset.idx_to_char == expected_metadata["idx_to_char"] 

def test_pickled_dataset_metadata_not_found(processed_data_dir):
    """Test metadata loading from pkl when metadata.json is missing."""
    data_path, expected_metadata, seq_length = processed_data_dir[:3]
    train_pkl_path = data_path / "train.pkl"
    metadata_path = data_path / "metadata.json"
    
    # Temporarily remove metadata file
    metadata_path.unlink() 
    
    # Instantiate dataset - it should load metadata from the pkl file now
    dataset = PickledDataset(str(train_pkl_path), block_size=seq_length)

    # Check attributes loaded from pkl
    assert dataset.vocab_size == expected_metadata["vocab_size"]
    assert dataset.char_to_idx == expected_metadata["char_to_idx"]
    assert dataset.idx_to_char == expected_metadata["idx_to_char"]
        
    # Recreate empty metadata for cleanup (fixture expects it)
    metadata_path.touch()

# --- Tests for create_data_loaders_from_config ---

def test_create_data_loaders_train_val(processed_data_dir):
    """Test creating train and validation DataLoaders."""
    data_path, _, seq_length, _ = processed_data_dir
    
    cfg_dict = {
        "data": {
            "train": {
                "dataset": {
                    "_target_": "src.craft.data.dataset.PickledDataset",
                    "file_path": str(data_path / "train.pkl"), # Use file_path
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
                    "_target_": "src.craft.data.dataset.PickledDataset",
                    "file_path": str(data_path / "val.pkl"), # Use file_path
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
    cfg = OmegaConf.create(cfg_dict)
    
    data_loaders = create_data_loaders_from_config(cfg.data)
    
    assert "train" in data_loaders
    assert "val" in data_loaders
    assert "test" not in data_loaders
    
    assert isinstance(data_loaders["train"], DataLoader)
    assert isinstance(data_loaders["val"], DataLoader)
    
    assert data_loaders["train"].batch_size == 4
    assert data_loaders["val"].batch_size == 8
    
    # Check shuffle by inspecting sampler type (might be fragile)
    assert isinstance(data_loaders["train"].sampler, torch.utils.data.RandomSampler)
    assert isinstance(data_loaders["val"].sampler, torch.utils.data.SequentialSampler) 

    # Check dataset instance
    assert isinstance(data_loaders["train"].dataset, PickledDataset)
    assert data_loaders["train"].dataset.file_path == str(data_path / "train.pkl")

def test_create_data_loaders_train_val_test(processed_data_dir):
    """Test creating train, validation, and test DataLoaders."""
    data_path, _, seq_length, _ = processed_data_dir
    
    cfg_dict = {
        "data": {
            "train": { # Required
                 "dataset": {"_target_": "src.craft.data.dataset.PickledDataset", "file_path": str(data_path / "train.pkl"), "block_size": seq_length},
                 "dataloader": {"batch_size": 4}
            },
             "val": { # Required
                 "dataset": {"_target_": "src.craft.data.dataset.PickledDataset", "file_path": str(data_path / "val.pkl"), "block_size": seq_length},
                 "dataloader": {"batch_size": 4}
            },
            "test": { # Optional
                 "dataset": {"_target_": "src.craft.data.dataset.PickledDataset", "file_path": str(data_path / "test.pkl"), "block_size": seq_length},
                 "dataloader": {"batch_size": 4, "shuffle": False} # Explicit shuffle False
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    data_loaders = create_data_loaders_from_config(cfg.data)
    
    assert "train" in data_loaders
    assert "val" in data_loaders
    assert "test" in data_loaders
    
    assert isinstance(data_loaders["test"], DataLoader)
    assert isinstance(data_loaders["test"].sampler, torch.utils.data.SequentialSampler) 

def test_create_data_loaders_missing_split_config():
    """Test error handling for missing train or val config."""
    cfg_missing_val = OmegaConf.create({"data": {"train": { # Missing val
        "dataset": {"_target_": "..."}, # Need dummy dataset config to pass initial checks
        "dataloader": {}
    }}})
    cfg_missing_train = OmegaConf.create({"data": {"val": { # Missing train
        "dataset": {"_target_": "..."},
        "dataloader": {}
    }}})
    
    # Check missing 'val'
    with pytest.raises(ValueError, match="'val' split configuration is required"):
        create_data_loaders_from_config(cfg_missing_val.data)
        
    # Check missing 'train'
    with pytest.raises(ValueError, match="'train' split configuration is required"):
        create_data_loaders_from_config(cfg_missing_train.data)

def test_create_data_loaders_missing_target(processed_data_dir):
    """Test error handling for missing _target_ in dataset config."""
    data_path, _, seq_length, _ = processed_data_dir
    cfg_dict = {
        "data": {
            "train": {
                 "dataset": {"file_path": str(data_path / "train.pkl"), "block_size": seq_length}, # Missing _target_
                 "dataloader": {"batch_size": 4}
            },
             "val": { # Need a valid val config to pass initial checks
                 "dataset": {"_target_": "src.craft.data.dataset.PickledDataset", "file_path": str(data_path / "val.pkl"), "block_size": seq_length},
                 "dataloader": {"batch_size": 4}
            }
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    
    # Expecting our explicit ValueError now, not Hydra's error
    # Updated regex to match the actual error message which might specify DictConfig
    with pytest.raises(ValueError, match="Missing '_target_' key in 'dataset' (DictConfig|configuration) for split 'train'"):
        create_data_loaders_from_config(cfg.data) 