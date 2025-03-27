"""
Unit tests for model base classes.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from src.models.base import BaseModel, GenerativeModel, LanguageModel, create_model_from_config


class MockBaseModel(BaseModel):
    """Mock implementation of BaseModel for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


class MockGenerativeModel(GenerativeModel):
    """Mock implementation of GenerativeModel for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)
    
    def generate(self, prompt, max_length=10):
        return torch.zeros(1, max_length)


class MockLanguageModel(LanguageModel):
    """Mock implementation of LanguageModel for testing."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 10)
        self.linear = nn.Linear(10, 100)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)
    
    def generate(self, input_ids, max_new_tokens=10):
        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, input_ids.shape[1] + max_new_tokens)


class TestBaseModel(unittest.TestCase):
    """Tests for the BaseModel class."""
    
    def setUp(self):
        self.model = MockBaseModel()
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.model_type, "base")
        self.assertIsInstance(self.model, nn.Module)
    
    def test_forward(self):
        """Test the forward method."""
        x = torch.randn(5, 10)
        output = self.model(x)
        self.assertEqual(output.shape, (5, 10))
    
    def test_get_config(self):
        """Test the get_config method."""
        config = self.model.get_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config["model_type"], "base")
    
    def test_save_load(self):
        """Test the save and load methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "model.pt")
            self.model.save(path)
            self.assertTrue(os.path.exists(path))
            
            # Create a new model and load the saved state
            new_model = MockBaseModel()
            new_model.load(path)
            
            # Check that the parameters are the same
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                self.assertTrue(torch.all(torch.eq(p1, p2)))


class TestGenerativeModel(unittest.TestCase):
    """Tests for the GenerativeModel class."""
    
    def setUp(self):
        self.model = MockGenerativeModel()
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.model_type, "generative")
        self.assertIsInstance(self.model, BaseModel)
    
    def test_generate_method(self):
        """Test the generate method."""
        prompt = torch.randn(1, 5)
        output = self.model.generate(prompt)
        self.assertEqual(output.shape, (1, 10))


class TestLanguageModel(unittest.TestCase):
    """Tests for the LanguageModel class."""
    
    def setUp(self):
        self.model = MockLanguageModel()
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.model_type, "language")
        self.assertIsInstance(self.model, GenerativeModel)
    
    def test_forward(self):
        """Test the forward method."""
        x = torch.randint(0, 100, (5, 10))
        output = self.model(x)
        self.assertEqual(output.shape, (5, 10, 100))
    
    def test_generate_method(self):
        """Test the generate method."""
        input_ids = torch.randint(0, 100, (2, 5))
        output = self.model.generate(input_ids, max_new_tokens=15)
        self.assertEqual(output.shape, (2, 20))  # input (5) + new tokens (15)
    
    def test_calculate_perplexity(self):
        """Test the calculate_perplexity method."""
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        perplexity = self.model.calculate_perplexity(logits, targets)
        self.assertGreater(perplexity.item(), 0)


class TestModelCreation(unittest.TestCase):
    """Tests for the model creation function."""
    
    @patch("src.models.base.create_transformer_model")
    def test_create_language_model(self, mock_create_transformer):
        """Test creating a language model from config."""
        # Set up the mock
        mock_model = MagicMock()
        mock_create_transformer.return_value = mock_model
        
        # Create a simple language model config
        config = {
            "model_type": "language",
            "architecture": "transformer",
            "vocab_size": 100
        }
        
        # Call the function
        model = create_model_from_config(config)
        
        # Check the result
        self.assertIsNotNone(model)
        mock_create_transformer.assert_called_once_with(**config)
    
    def test_unsupported_model_type(self):
        """Test that an error is raised for unsupported model types."""
        config = {
            "model_type": "unsupported"
        }
        
        with self.assertRaises(ValueError):
            create_model_from_config(config)


if __name__ == "__main__":
    unittest.main() 