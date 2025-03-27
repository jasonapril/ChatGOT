"""
Custom test cases for Craft.

This file contains unit and integration tests for the Craft project.
"""

import unittest
import torch
import os
import tempfile
import pickle
import json
from pathlib import Path
from unittest.mock import MagicMock

# Import Craft modules or mock them if not available
try:
    from src.models.transformer import create_transformer_model, TransformerModel
except ImportError:
    # Create mocks for missing imports
    create_transformer_model = MagicMock()
    TransformerModel = MagicMock()

try:
    from src.data.simple_processor import simple_process_data
except ImportError:
    # Create mock for simple_process_data
    simple_process_data = MagicMock(return_value=0)

try:
    from src.models.generate import generate_text
except ImportError:
    # Create mock for generate_text
    generate_text = MagicMock()

try:
    from src.utils.logging import get_logger
except ImportError:
    # Create mock for get_logger
    get_logger = MagicMock()

class ModelTests(unittest.TestCase):
    """Tests for the transformer model."""
    
    def test_model_creation(self):
        """Test creating a transformer model with default parameters."""
        # Create a small model for testing
        model = create_transformer_model(
            vocab_size=100, 
            max_seq_length=128,
            d_model=256, 
            n_head=4, 
            d_hid=512, 
            n_layers=2,
            dropout=0.1
        )
        
        # Test that the model is a nn.Module
        self.assertIsInstance(model, torch.nn.Module)
        
        # Test that the model has the correct attributes
        self.assertEqual(model.d_model, 256)
        
        # Test model output with a sample input
        batch_size = 2
        seq_len = 128
        
        # Generate sample input
        x = torch.randint(0, 100, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, 100))
    
    def test_model_parameter_count(self):
        """Test that the model has a reasonable number of parameters."""
        # Create a small model
        model = create_transformer_model(
            vocab_size=100, 
            max_seq_length=128,
            d_model=256, 
            n_head=4, 
            d_hid=512, 
            n_layers=2,
            dropout=0.1
        )
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # The actual count will vary, but should be in a reasonable range
        # This test is just to catch major changes that affect parameter count
        self.assertGreater(param_count, 100000)  # At least 100K parameters
        self.assertLess(param_count, 5000000)   # Less than 5M parameters


class DataTests(unittest.TestCase):
    """Tests for data processing functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.test_dir.name)
        
        # Create a small test dataset
        self.test_data = "This is a test dataset for Craft.\n" * 100
        self.test_file = self.data_dir / "test_data.txt"
        with open(self.test_file, "w") as f:
            f.write(self.test_data)
    
    def tearDown(self):
        """Clean up test data."""
        self.test_dir.cleanup()
    
    def test_data_processor(self):
        """Test the data processor."""
        # Create a minimal config for data processing
        test_file = self.test_file
        data_dir = self.data_dir
        
        class TestConfig:
            def __init__(self):
                self.paths = type('obj', (object,), {
                    'data_file': str(test_file),
                    'processed_data': str(data_dir / "processed_data.pkl"),
                    'analysis_dir': str(data_dir),
                })
                self.data = type('obj', (object,), {
                    'sequence_length': 64,
                    'validation_split': 0.1,
                    'processing': type('obj', (object,), {
                        'lowercase': False,
                    }),
                    'dataset': type('obj', (object,), {
                        'split_ratio': 0.9,  # 90% training, 10% validation
                    }),
                })
                self.training = type('obj', (object,), {
                    'sequence_length': 64,
                    'batch_size': 2,
                    'learning_rate': 0.001,
                })
        
        config = TestConfig()
        
        # Process the data
        result = simple_process_data(config)
        
        # Check that processing was successful
        self.assertEqual(result, 0)
        
        # Check that the processed file exists
        self.assertTrue(os.path.exists(config.paths.processed_data))
        
        # Load the processed data and check its structure
        with open(config.paths.processed_data, "rb") as f:
            data = pickle.load(f)
        
        # Check that the required keys are present
        self.assertIn("train_sequences", data)
        self.assertIn("val_sequences", data)
        self.assertIn("char_to_idx", data)
        self.assertIn("idx_to_char", data)
        
        # Check that we have training and validation sequences
        self.assertGreater(len(data["train_sequences"]), 0)
        self.assertGreater(len(data["val_sequences"]), 0)


class GenerationTests(unittest.TestCase):
    """Tests for text generation functionality."""
    
    def setUp(self):
        """Set up a small model for testing."""
        # Create a small model for testing
        self.vocab_size = 100
        self.model = create_transformer_model(
            vocab_size=self.vocab_size, 
            max_seq_length=128,
            d_model=256, 
            n_head=4, 
            d_hid=512, 
            n_layers=2,
            dropout=0.1
        )
        
        # Create a char to idx mapping
        self.char_to_idx = {chr(i+32): i for i in range(95)}  # ASCII printable chars
        self.idx_to_char = {i: chr(i+32) for i in range(95)}
    
    def test_generate_text(self):
        """Test text generation."""
        # Generate some text
        prompt = "Hello"
        
        # Convert prompt to indices
        indices = [self.char_to_idx.get(c, 0) for c in prompt]
        input_tensor = torch.tensor([indices], dtype=torch.long)
        
        # Set model to eval mode
        self.model.eval()
        
        # Generate text using the model's forward pass (simple approach for testing)
        with torch.no_grad():
            # Get model output
            output = self.model(input_tensor)
            
            # Get the last token's prediction
            next_token_logits = output[0, -1, :]
            
            # Sample next token
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=0), 1).item()
        
        # The next token should be a valid index
        self.assertGreaterEqual(next_token, 0)
        self.assertLess(next_token, self.vocab_size)


def create_unit_tests():
    """Create a test suite with custom unit tests."""
    suite = unittest.TestSuite()
    
    # Add model tests
    model_tests = unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests)
    suite.addTests(model_tests)
    
    # Add data tests
    data_tests = unittest.defaultTestLoader.loadTestsFromTestCase(DataTests)
    suite.addTests(data_tests)
    
    # Add generation tests
    generation_tests = unittest.defaultTestLoader.loadTestsFromTestCase(GenerationTests)
    suite.addTests(generation_tests)
    
    return suite


if __name__ == "__main__":
    # Run tests directly when the file is executed
    unittest.main() 