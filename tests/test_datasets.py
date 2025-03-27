"""
Unit tests for dataset base classes.
"""
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader

from src.data.base import BaseDataset, TextDataset, create_dataloaders, create_dataset_from_config


class MockBaseDataset(BaseDataset):
    """Mock implementation of BaseDataset for testing."""
    
    def __init__(self, data=None):
        super().__init__()
        self.data = data or [i for i in range(100)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MockTextDataset(TextDataset):
    """Mock implementation of TextDataset for testing."""
    
    def __init__(self, text="hello world"):
        super().__init__()
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_idx[c] for c in text]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x, x  # Return the same for input and target
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.idx_to_char[i] for i in indices])


class TestBaseDataset(unittest.TestCase):
    """Tests for the BaseDataset class."""
    
    def setUp(self):
        self.dataset = MockBaseDataset()
    
    def test_initialization(self):
        """Test that the dataset initializes correctly."""
        self.assertEqual(self.dataset.data_type, "base")
        self.assertIsInstance(self.dataset, BaseDataset)
    
    def test_len(self):
        """Test the __len__ method."""
        self.assertEqual(len(self.dataset), 100)
    
    def test_getitem(self):
        """Test the __getitem__ method."""
        self.assertEqual(self.dataset[10], 10)
    
    def test_get_config(self):
        """Test the get_config method."""
        config = self.dataset.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config["data_type"], "base")


class TestTextDataset(unittest.TestCase):
    """Tests for the TextDataset class."""
    
    def setUp(self):
        self.dataset = MockTextDataset()
    
    def test_initialization(self):
        """Test that the dataset initializes correctly."""
        self.assertEqual(self.dataset.data_type, "text")
        self.assertIsInstance(self.dataset, BaseDataset)
    
    def test_len(self):
        """Test the __len__ method."""
        self.assertEqual(len(self.dataset), len(self.dataset.text))
    
    def test_getitem(self):
        """Test the __getitem__ method."""
        x, y = self.dataset[5]
        self.assertEqual(x.item(), self.dataset.char_to_idx[self.dataset.text[5]])
        self.assertEqual(y.item(), self.dataset.char_to_idx[self.dataset.text[5]])
    
    def test_decode(self):
        """Test the decode method."""
        indices = [self.dataset.char_to_idx[c] for c in "hello"]
        decoded = self.dataset.decode(indices)
        self.assertEqual(decoded, "hello")
        
        # Test with tensor
        indices_tensor = torch.tensor(indices)
        decoded_tensor = self.dataset.decode(indices_tensor)
        self.assertEqual(decoded_tensor, "hello")


class TestDataLoaderCreation(unittest.TestCase):
    """Tests for dataloader creation functions."""
    
    def setUp(self):
        self.dataset = MockBaseDataset()
    
    def test_create_dataloaders(self):
        """Test creating dataloaders from a dataset."""
        train_dataloader, val_dataloader = create_dataloaders(
            dataset=self.dataset,
            batch_size=16,
            val_split=0.2,
            seed=42,
            num_workers=0,
            pin_memory=False,
        )
        
        # Check that the dataloaders were created
        self.assertIsInstance(train_dataloader, DataLoader)
        self.assertIsInstance(val_dataloader, DataLoader)
        
        # Check the lengths
        self.assertEqual(len(train_dataloader.dataset), 80)  # 80% of 100
        self.assertEqual(len(val_dataloader.dataset), 20)    # 20% of 100
        
        # Check batch size
        self.assertEqual(train_dataloader.batch_size, 16)
        self.assertEqual(val_dataloader.batch_size, 16)
    
    def test_create_dataloaders_no_val(self):
        """Test creating dataloaders with no validation split."""
        train_dataloader, val_dataloader = create_dataloaders(
            dataset=self.dataset,
            batch_size=16,
            val_split=0.0,
            seed=42,
            num_workers=0,
            pin_memory=False,
        )
        
        # Check that only the train dataloader was created
        self.assertIsInstance(train_dataloader, DataLoader)
        self.assertIsNone(val_dataloader)
        
        # Check the length
        self.assertEqual(len(train_dataloader.dataset), 100)  # 100% of 100


class TestDatasetCreation(unittest.TestCase):
    """Tests for dataset creation function."""
    
    @patch("src.data.base.CharDataset")
    def test_create_text_dataset(self, mock_char_dataset):
        """Test creating a text dataset from config."""
        # Set up the mock
        mock_dataset = MagicMock()
        mock_char_dataset.return_value = mock_dataset
        
        # Mock open function
        with patch("builtins.open", unittest.mock.mock_open(read_data="hello world")):
            # Create a simple text dataset config
            config = {
                "data_type": "text",
                "format": "character",
                "data_path": "dummy.txt",
                "block_size": 10
            }
            
            # Call the function
            dataset = create_dataset_from_config(config)
            
            # Check the result
            self.assertIsNotNone(dataset)
            mock_char_dataset.assert_called_once()
    
    def test_unsupported_data_type(self):
        """Test that an error is raised for unsupported data types."""
        config = {
            "data_type": "unsupported"
        }
        
        with self.assertRaises(ValueError):
            create_dataset_from_config(config)


if __name__ == "__main__":
    unittest.main() 