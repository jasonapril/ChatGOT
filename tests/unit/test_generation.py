import unittest
import os
import sys
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.generation import (
    generate_text,
    sample_text,
    beam_search_generate,
    batch_generate
)

class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self, vocab_size=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)
        self.vocab_size = vocab_size
        
    def forward(self, x):
        # For testing, just return a random distribution
        if len(x.shape) == 1:
            # Single token prediction - [batch_size]
            batch_size = x.shape[0]
            return torch.randn(batch_size, self.vocab_size)
        else:
            # Sequence prediction - [batch_size, seq_len]
            batch_size, seq_len = x.shape
            return torch.randn(batch_size, seq_len, self.vocab_size)

class TestGeneration(unittest.TestCase):
    """Unit tests for the generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set vocabulary size
        self.vocab_size = 10
        
        # Create a simple model for testing
        self.model = SimpleModel(vocab_size=self.vocab_size)
        
        # Create character mappings
        self.char_to_idx = {chr(ord('a') + i): i for i in range(self.vocab_size)}
        self.idx_to_char = {i: chr(ord('a') + i) for i in range(self.vocab_size)}
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_generate_text_basic(self):
        """Test basic text generation functionality."""
        # Generate text
        seed_text = "abc"
        max_length = 10
        
        # Run generation
        generated_text = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            device=self.device
        )
        
        # Check that the function returned a string
        self.assertIsInstance(generated_text, str)
        
        # Check that the generated text starts with the seed text
        self.assertTrue(generated_text.startswith(seed_text))
        
        # Check that the generated text has the correct length
        self.assertEqual(len(generated_text), max_length)
    
    def test_generate_text_with_temperature(self):
        """Test text generation with temperature parameter."""
        # Generate text with different temperatures
        seed_text = "abc"
        max_length = 10
        
        # Run generation with low temperature (more deterministic)
        generated_text_low_temp = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=0.1,
            device=self.device
        )
        
        # Run generation with high temperature (more random)
        generated_text_high_temp = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=2.0,
            device=self.device
        )
        
        # Both generations should be valid
        self.assertIsInstance(generated_text_low_temp, str)
        self.assertIsInstance(generated_text_high_temp, str)
        
        # Both should start with the seed text
        self.assertTrue(generated_text_low_temp.startswith(seed_text))
        self.assertTrue(generated_text_high_temp.startswith(seed_text))
    
    def test_generate_text_with_top_k(self):
        """Test text generation with top-k sampling."""
        # Generate text with top-k sampling
        seed_text = "abc"
        max_length = 10
        
        # Run generation with top-k sampling
        generated_text = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=1.0,
            top_k=3,
            device=self.device
        )
        
        # Check that the function returned a string
        self.assertIsInstance(generated_text, str)
        
        # Check that the generated text starts with the seed text
        self.assertTrue(generated_text.startswith(seed_text))
    
    def test_generate_text_with_top_p(self):
        """Test text generation with top-p sampling."""
        # Generate text with top-p sampling
        seed_text = "abc"
        max_length = 10
        
        # Run generation with top-p sampling
        generated_text = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=1.0,
            top_p=0.9,
            device=self.device
        )
        
        # Check that the function returned a string
        self.assertIsInstance(generated_text, str)
        
        # Check that the generated text starts with the seed text
        self.assertTrue(generated_text.startswith(seed_text))
    
    @patch('logging.info')
    def test_sample_text(self, mock_logging):
        """Test sampling multiple texts."""
        # Sample texts
        seed_text = "abc"
        num_samples = 3
        max_length = 10
        
        # Run sampling
        samples = sample_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            num_samples=num_samples,
            seed_text=seed_text,
            max_length=max_length,
            device=self.device,
            log_samples=True
        )
        
        # Check that the function returned a list
        self.assertIsInstance(samples, list)
        
        # Check that we got the correct number of samples
        self.assertEqual(len(samples), num_samples)
        
        # Check that all samples start with the seed text
        for sample in samples:
            self.assertTrue(sample.startswith(seed_text))
            self.assertEqual(len(sample), max_length)
        
        # Check that logging was called for each sample
        self.assertEqual(mock_logging.call_count, num_samples)
    
    def test_beam_search_generate(self):
        """Test beam search generation."""
        # Generate text with beam search
        seed_text = "abc"
        max_length = 10
        beam_width = 3
        
        # Run beam search generation
        generated_text = beam_search_generate(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            beam_width=beam_width,
            device=self.device
        )
        
        # Check that the function returned a string
        self.assertIsInstance(generated_text, str)
        
        # Check that the generated text starts with the seed text
        self.assertTrue(generated_text.startswith(seed_text))
        
        # Check that the generated text has the correct length
        self.assertEqual(len(generated_text), max_length)
    
    def test_batch_generate(self):
        """Test batch text generation."""
        # Generate text for multiple prompts in parallel
        prompts = ["abc", "def", "ghi"]
        max_length = 10
        
        # Run batch generation
        generated_texts = batch_generate(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            prompts=prompts,
            max_length=max_length,
            device=self.device
        )
        
        # Check that the function returned a list
        self.assertIsInstance(generated_texts, list)
        
        # Check that we got the correct number of samples
        self.assertEqual(len(generated_texts), len(prompts))
        
        # Check that all generated texts start with their respective prompts
        for i, text in enumerate(generated_texts):
            self.assertTrue(text.startswith(prompts[i]))
            self.assertEqual(len(text), max_length)

if __name__ == '__main__':
    unittest.main() 