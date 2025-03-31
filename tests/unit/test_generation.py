import unittest
import os
import sys
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import logging

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
        # Define seed_text used in multiple tests
        self.seed_text = "abc" 
    
    def test_generate_text_basic(self):
        """Test basic text generation functionality."""
        max_length = 10
        generated_text = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=self.seed_text,
            max_length=max_length,
            # Use a very small temperature instead of 0 for numerical stability with multinomial
            temperature=1e-8, # Use 0 for deterministic output (greedy)
            device=self.device
        )
        # Assert total length is seed + max_length
        self.assertEqual(len(generated_text), len(self.seed_text) + max_length)
        # Assert the beginning matches the seed
        self.assertTrue(generated_text.startswith(self.seed_text))
    
    def test_generate_text_with_temperature(self):
        """Test text generation with temperature parameter."""
        # Generate text with different temperatures
        max_length = 10
        
        # Run generation with low temperature (more deterministic)
        generated_text_low_temp = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=self.seed_text,
            max_length=max_length,
            temperature=0.1,
            device=self.device
        )
        
        # Run generation with high temperature (more random)
        generated_text_high_temp = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=self.seed_text,
            max_length=max_length,
            temperature=2.0,
            device=self.device
        )
        
        # Both generations should be valid
        self.assertIsInstance(generated_text_low_temp, str)
        self.assertIsInstance(generated_text_high_temp, str)
        
        # Both should start with the seed text
        self.assertTrue(generated_text_low_temp.startswith(self.seed_text))
        self.assertTrue(generated_text_high_temp.startswith(self.seed_text))
        
        # Assert total length is seed + max_length
        self.assertEqual(len(generated_text_low_temp), len(self.seed_text) + max_length)
        self.assertEqual(len(generated_text_high_temp), len(self.seed_text) + max_length)
    
    def test_generate_text_with_top_k(self):
        """Test text generation with top-k sampling."""
        # Generate text with top-k sampling
        max_length = 10
        
        # Run generation with top-k sampling
        generated_text = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=self.seed_text,
            max_length=max_length,
            temperature=1.0,
            top_k=3,
            device=self.device
        )
        
        # Check that the function returned a string
        self.assertIsInstance(generated_text, str)
        
        # Check that the generated text starts with the seed text
        self.assertTrue(generated_text.startswith(self.seed_text))
        
        # Assert total length is seed + max_length
        self.assertEqual(len(generated_text), len(self.seed_text) + max_length)
    
    def test_generate_text_with_top_p(self):
        """Test text generation with top-p sampling."""
        # Generate text with top-p sampling
        max_length = 10
        
        # Run generation with top-p sampling
        generated_text = generate_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=self.seed_text,
            max_length=max_length,
            temperature=1.0,
            top_p=0.9,
            device=self.device
        )
        
        # Check that the function returned a string
        self.assertIsInstance(generated_text, str)
        
        # Check that the generated text starts with the seed text
        self.assertTrue(generated_text.startswith(self.seed_text))
        
        # Assert total length is seed + max_length
        self.assertEqual(len(generated_text), len(self.seed_text) + max_length)
    
    @patch('src.training.generation.generate_text')
    def test_sample_text(self, mock_generate_text):
        """Test sampling multiple texts."""
        num_samples = 3
        max_length = 10
        # Configure the mock to return strings of the correct expected length
        mock_generate_text.side_effect = [
            self.seed_text + "g" * max_length 
            for _ in range(num_samples)
        ]

        samples = sample_text(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            num_samples=num_samples,
            seed_text=self.seed_text,
            max_length=max_length,
            device=self.device,
            log_samples=False
        )
        
        self.assertEqual(len(samples), num_samples)
        # Check the mock was called correctly
        self.assertEqual(mock_generate_text.call_count, num_samples)
        # Check the returned lengths from the mock
        for sample in samples:
            self.assertEqual(len(sample), len(self.seed_text) + max_length)

    def test_beam_search_generate(self):
        """Test beam search generation."""
        # This function might have different length semantics, skip for now
        # max_length = 10
        max_length = 10
        beam_width = 3
        seed_len = len(self.seed_text)

        generated_text = beam_search_generate(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            seed_text=self.seed_text,
            max_length=max_length,
            beam_width=beam_width,
            device=self.device
        )
        # Basic checks: output is string, starts with seed
        self.assertIsInstance(generated_text, str)
        self.assertTrue(generated_text.startswith(self.seed_text))
        # Length check: should be between seed length and seed + max_length
        self.assertGreaterEqual(len(generated_text), seed_len)
        self.assertLessEqual(len(generated_text), seed_len + max_length)
        # Optionally log the output
        logging.info(f"Beam search (w={beam_width}) output: {generated_text}")

    def test_batch_generate(self):
        """Test batch text generation."""
        # This function needs more complex setup and checks, skip for now
        # prompts = [self.seed_text, "def"]
        prompts = [self.seed_text, "def", "ghi"]
        max_length = 10
        
        results = batch_generate(
            model=self.model,
            char_to_idx=self.char_to_idx,
            idx_to_char=self.idx_to_char,
            prompts=prompts,
            max_length=max_length,
            temperature=1.0, # Use temp > 0
            device=self.device
            # Add top_p/top_k if needed
        )
        # Check correct number of results
        self.assertEqual(len(results), len(prompts))
        
        # Check each result individually
        for i, text in enumerate(results):
            prompt = prompts[i]
            seed_len = len(prompt)
            # Basic checks: output is string, starts with prompt
            self.assertIsInstance(text, str)
            self.assertTrue(text.startswith(prompt))
            # Length check: between prompt length and prompt + max_length
            self.assertGreaterEqual(len(text), seed_len)
            self.assertLessEqual(len(text), seed_len + max_length)
            # Optionally log the output
            logging.info(f"Batch generation sample {i}: {text}")

if __name__ == '__main__':
    unittest.main() 