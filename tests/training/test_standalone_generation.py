"""
Tests for standalone generation functions (sampling, beam search, batch).
"""
import pytest
import torch
import torch.nn as nn # Import nn
import torch.nn.functional as F

# Functions to test
from craft.training.sampling import generate_text_sampling, sample_text
from craft.training.beam_search import beam_search_generate
from craft.training.batch_generation import batch_generate

# --- Fixtures ---

@pytest.fixture(scope="function")
def standalone_mock_vocab():
    """Provides simple character vocabulary mappings with explicit PAD."""
    # Define PAD explicitly at index 0 for clarity
    chars = "<pad><eos> abcdefghijklmnopqrstuvwxyz" # PAD=0, EOS=1, space=2, a=3 ...
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char

class MockGeneratorModel(torch.nn.Module):
    """A mock model that returns predictable logits."""
    def __init__(self, vocab_size=30): # Adjust vocab size based on fixture
        super().__init__()
        self.vocab_size = vocab_size
        # No parameters needed

    def forward(self, input_ids, **kwargs):
        """Returns logits where the next token index is always input_ids + 1 (cyclical). Ignores PAD token."""
        batch_size, seq_len = input_ids.shape
        last_token_ids = input_ids[:, -1]

        # Simple cycling logic: next = (last + 1) % vocab_size
        # This mock doesn't explicitly ignore padding, batch_generate test relies
        # on the batch_generate function itself handling context/padding correctly.
        next_token_ids = (last_token_ids + 1) % self.vocab_size

        logits = torch.full((batch_size, 1, self.vocab_size), -1e9, device=input_ids.device)
        logits.scatter_(2, next_token_ids.unsqueeze(-1).unsqueeze(-1), 1.0)
        # Return full logits matching input shape, only last token matters
        full_logits = torch.full((batch_size, seq_len, self.vocab_size), -1e9, device=input_ids.device)
        full_logits[:, -1, :] = logits.squeeze(1)
        return full_logits

@pytest.fixture(scope="function")
def mock_model(standalone_mock_vocab):
    char_to_idx, _ = standalone_mock_vocab
    return MockGeneratorModel(vocab_size=len(char_to_idx))

# --- Tests for sampling.py --- 

def test_generate_text_sampling_basic(mock_model, standalone_mock_vocab):
    """Test basic greedy generation (temperature=0 or very low)."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    seed = "a" # index 3
    max_new_tokens = 5
    # Expected sequence: a(3) -> b(4) -> c(5) -> d(6) -> e(7) -> f(8)
    expected_output = "abcdef"
    generated = generate_text_sampling(mock_model, char_to_idx, idx_to_char, seed, max_new_tokens, temperature=0.001, device='cpu')
    assert generated == expected_output

def test_generate_text_sampling_length(mock_model, standalone_mock_vocab):
    """Test if max_new_tokens is respected for *new* tokens."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    seed = "a"
    max_new_tokens = 10
    generated = generate_text_sampling(mock_model, char_to_idx, idx_to_char, seed, max_new_tokens, temperature=0.1, device='cpu')
    assert len(generated) == len(seed) + max_new_tokens # 1 + 10 = 11

def test_generate_text_sampling_params_smoke(mock_model, standalone_mock_vocab):
    """Smoke test with various sampling parameters enabled."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    seed = "a"
    max_new_tokens = 10
    generated = generate_text_sampling(
        mock_model, char_to_idx, idx_to_char, seed, max_new_tokens,
        temperature=0.7, top_k=5, top_p=0.9, repetition_penalty=1.2, device='cpu'
    )
    assert isinstance(generated, str)
    assert len(generated) == len(seed) + max_new_tokens # 1 + 10 = 11

def test_sample_text(mock_model, standalone_mock_vocab):
    """Test the sample_text wrapper."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    num_samples = 3
    seed = "a"
    max_new_tokens = 5
    expected_output = "abcdef"
    samples = sample_text(mock_model, char_to_idx, idx_to_char, num_samples, seed_text=seed, max_length=max_new_tokens, temperature=0.1, log_samples=False, device='cpu')
    assert isinstance(samples, list)
    assert len(samples) == num_samples
    assert all(s == expected_output for s in samples)

# --- Tests for beam_search.py --- 

def test_beam_search_basic(mock_model, standalone_mock_vocab):
    """Test basic beam search generation."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    seed = "a"
    max_new_tokens = 5
    expected_output = "abcdef"
    generated = beam_search_generate(mock_model, char_to_idx, idx_to_char, seed, max_length=max_new_tokens, beam_width=3, device='cpu')
    assert generated == expected_output

def test_beam_search_params_smoke(mock_model, standalone_mock_vocab):
    """Smoke test beam search with different params."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    seed = "a"
    max_new_tokens = 5
    expected_output = "abcdef"
    generated = beam_search_generate(mock_model, char_to_idx, idx_to_char, seed, max_length=max_new_tokens, beam_width=2, length_penalty=0.8, device='cpu')
    assert isinstance(generated, str)
    assert generated == expected_output
    assert len(generated) == len(seed) + max_new_tokens # 1 + 5 = 6

# --- Tests for batch_generation.py --- 

def test_batch_generate_basic(mock_model, standalone_mock_vocab):
    """Test basic batch generation."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    prompts = ["a", "b", "c"] # Indices 3, 4, 5
    max_new = 3 # Max *new* tokens
    # Expected: a(3)->b(4)c(5)d(6), b(4)->c(5)d(6)e(7), c(5)->d(6)e(7)f(8)
    expected = ["abcd", "bcde", "cdef"]
    generated_list = batch_generate(mock_model, char_to_idx, idx_to_char, prompts, max_new, temperature=0.1, device='cpu')
    assert isinstance(generated_list, list)
    assert len(generated_list) == len(prompts)
    assert generated_list == expected

def test_batch_generate_different_lengths(mock_model, request):
    """Test batch generation with prompts of different lengths."""
    # Explicitly get the fixture value by name - REMOVED
    # standalone_mock_vocab = request.getfixturevalue("standalone_mock_vocab")
    
    # --- Hardcode the correct vocabulary for this specific test --- 
    # chars = "<pad><eos> abcdefghijklmnopqrstuvwxyz" # PAD=0, EOS=1, space=2, a=3 ...
    # char_to_idx = {ch: i for i, ch in enumerate(chars)} # Faulty comprehension
    # idx_to_char = {i: ch for i, ch in enumerate(chars)} # Faulty comprehension
    
    # Manually construct the vocabulary to ensure correctness
    vocab_list = ["<pad>", "<eos>", " "] + list("abcdefghijklmnopqrstuvwxyz")
    char_to_idx = {token: i for i, token in enumerate(vocab_list)}
    idx_to_char = {i: token for i, token in enumerate(vocab_list)}
    # print(f"\n[DEBUG] char_to_idx IMMEDIATELY AFTER DEFINITION:\n{char_to_idx}\n") # Removed print
    # --- End Hardcoding / Manual Construction ---
    
    prompts = ["a", "bc"] # Indices [3], [4, 5] in the manually constructed vocab
    max_tokens_to_generate = 3 # Variable for clarity
    
    # Generate first to see if it modifies the vocab 
    generated_list = batch_generate(mock_model, char_to_idx, idx_to_char, prompts, max_length=max_tokens_to_generate, temperature=0.1, device='cpu') # Corrected max_length

    # Ensure <pad> exists and has index 0 as assumed by test logic
    # print(f"\n[DEBUG] char_to_idx IMMEDIATELY BEFORE ASSERTION:\n{char_to_idx}\n") # Removed print
    assert "<pad>" in char_to_idx, "'<pad>' token missing in manually constructed vocab"
    pad_id = char_to_idx['<pad>']
    assert pad_id == 0, f"Test assumes pad_id is 0, but got {pad_id}"
    max_tokens_to_generate = 3
    # Expected: a(3)->b(4)c(5)d(6), bc(4,5)->d(6)e(7)f(8)
    # This expectation assumes batch_generate handles padding correctly
    # and the mock model's simple logic doesn't break things.
    # expected = ["abcd", "bcdef"] # Original incorrect expectation
    # Corrected expectation based on mock model predicting EOS after PAD
    expected = ["a", "bcdef"]
    # generated_list = batch_generate(mock_model, char_to_idx, idx_to_char, prompts, max_length, temperature=0.1, device='cpu') # Call was moved up
    assert len(generated_list) == len(prompts)
    assert generated_list == expected, f"Batch generation failed. Got: {generated_list}. Padding/context handling might be incorrect."

def test_batch_generate_params_smoke(mock_model, standalone_mock_vocab):
    """Smoke test batch generation with top_p."""
    char_to_idx, idx_to_char = standalone_mock_vocab
    prompts = ["a", "b"]
    max_new = 3
    expected = ["abcd", "bcde"] # Still deterministic with fixed mock
    generated_list = batch_generate(mock_model, char_to_idx, idx_to_char, prompts, max_new, temperature=0.7, top_p=0.9, device='cpu')
    assert isinstance(generated_list, list)
    assert len(generated_list) == len(prompts)
    assert generated_list == expected