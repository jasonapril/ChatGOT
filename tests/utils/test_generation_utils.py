#!/usr/bin/env python
"""
Tests for src.craft.utils.generation
"""
import torch
import pytest
from unittest.mock import MagicMock, ANY
from unittest.mock import patch

# Function to test
from craft.utils.generation import top_k_top_p_filtering
from craft.utils.generation import generate_sample_text, sample_text

# --- Mocks --- 

class MockGenerationModel(torch.nn.Module):
    """Mock model with a controllable generate method."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # Mock the generate method
        self.generate = MagicMock()
        # Need a dummy parameter for .to(device) to work
        self._dummy_param = torch.nn.Parameter(torch.empty(0))
        
    def eval(self):
        # Mock eval method
        pass

class MockTokenizer:
    """Mock tokenizer with encode/decode."""
    def __init__(self):
        self.encode = MagicMock()
        self.decode = MagicMock()

class MockCharLevel:
    """Mock character-level tokenizer/dataset."""
    def __init__(self):
        self.char_to_idx = {'a': 0, 'b': 1, 'c': 2, '<pad>': 3, '<eos>': 4}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.decode = MagicMock(side_effect=self._mock_decode)
        
    def _mock_decode(self, ids):
        return "".join([self.idx_to_char.get(i, '?') for i in ids])


# --- Tests for top_k_top_p_filtering --- 

def test_top_k_filtering_basic():
    """Test basic top-k filtering."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]) # Batch size 1, vocab size 5
    top_k = 3
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k)
    # Expected: keep top 3 (indices 2, 3, 4 with values 3.0, 4.0, 5.0)
    expected_logits = torch.tensor([[-float('Inf'), -float('Inf'), 3.0, 4.0, 5.0]])
    assert torch.equal(filtered_logits, expected_logits)

def test_top_k_filtering_k_zero():
    """Test top-k filtering with k=0 (should be a no-op)."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    top_k = 0
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k)
    # Expected: original logits
    assert torch.equal(filtered_logits, logits)

def test_top_k_filtering_k_larger_than_vocab():
    """Test top-k filtering with k larger than vocab size (should be a no-op)."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    top_k = 10
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k)
    # Expected: original logits
    assert torch.equal(filtered_logits, logits)

def test_top_p_filtering_basic():
    """Test basic top-p (nucleus) filtering."""
    # Logits corresponding to probabilities: [~0.01, ~0.02, ~0.06, ~0.16, ~0.44, ~0.31]
    logits = torch.tensor([[1.0, 1.5, 2.5, 3.5, 4.5, 4.2]]) 
    top_p = 0.9 # Keep tokens until cumulative probability >= 0.9
    # Sorted probs: [~0.44(idx 4), ~0.31(idx 5), ~0.16(idx 3), ...]
    # Cumulative:   [~0.44        , ~0.75        , ~0.91        , ...]
    # Keep indices 4, 5, 3 (values 4.5, 4.2, 3.5)
    filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
    expected_logits = torch.tensor([[-float('Inf'), -float('Inf'), -float('Inf'), 3.5, 4.5, 4.2]])
    assert torch.equal(filtered_logits, expected_logits)

def test_top_p_filtering_p_one():
    """Test top-p filtering with p=1.0 (should be a no-op)."""
    logits = torch.tensor([[1.0, 1.5, 2.5, 3.5, 4.5, 4.2]]) 
    top_p = 1.0
    filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
    assert torch.equal(filtered_logits, logits)

def test_top_p_filtering_p_zero():
    """Test top-p filtering with p=0.0 (should keep only the top token)."""
    logits = torch.tensor([[1.0, 1.5, 2.5, 3.5, 4.5, 4.2]]) 
    top_p = 0.0
    filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
    # Expect only index 4 (value 4.5)
    expected_logits = torch.tensor([[-float('Inf'), -float('Inf'), -float('Inf'), -float('Inf'), 4.5, -float('Inf')]])
    assert torch.equal(filtered_logits, expected_logits)

def test_combined_top_k_top_p():
    """Test combined top-k and top-p filtering."""
    logits = torch.tensor([[1.0, 1.5, 2.5, 3.5, 4.5, 4.2]]) # Probs: [~0.01, ~0.02, ~0.06, ~0.16, ~0.44, ~0.31]
    top_k = 4 # Keep indices 5, 4, 3, 2 (values 4.2, 4.5, 3.5, 2.5)
    top_p = 0.8 # From the top_k tokens, keep until cumulative prob >= 0.8
    # Top-k logits: [[-inf, -inf, 2.5, 3.5, 4.5, 4.2]]
    # Top-k probs (renormalized): [~0.06 / S, ~0.16 / S, ~0.44 / S, ~0.31 / S] where S = ~0.97
    # Approx renormalized:         [~0.06 , ~0.16 , ~0.45 , ~0.32]
    # Sorted renormalized:         [~0.45(idx 4), ~0.32(idx 5), ~0.16(idx 3), ~0.06(idx 2)]
    # Cumulative:                  [~0.45        , ~0.77        , ~0.93        , ...]
    # Keep indices 4, 5, 3 (values 4.5, 4.2, 3.5)
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    expected_logits = torch.tensor([[-float('Inf'), -float('Inf'), -float('Inf'), 3.5, 4.5, 4.2]])
    assert torch.equal(filtered_logits, expected_logits) 

def test_top_k_top_p_no_filtering():
    """Test that no filtering occurs when top_k=0 and top_p=1.0"""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=1.0)
    assert torch.equal(logits, filtered_logits)

def test_top_k_filtering_only():
    """Test basic top-k filtering"""
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    expected_logits = torch.tensor([[-float('Inf'), 3.0, 2.0, -float('Inf')]])
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=2, top_p=1.0)
    assert torch.equal(filtered_logits, expected_logits)

def test_top_k_less_than_vocab():
    """Test top-k when k is less than vocab size"""
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    expected_logits = torch.tensor([[-float('Inf'), 3.0, 2.0]])
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=2, top_p=1.0)
    assert torch.equal(filtered_logits, expected_logits)

def test_top_k_greater_than_vocab():
    """Test top-k when k is greater than or equal to vocab size (should be no-op)"""
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    expected_logits = torch.tensor([[1.0, 3.0, 2.0]])
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=3, top_p=1.0)
    assert torch.equal(filtered_logits, expected_logits)
    filtered_logits_k4 = top_k_top_p_filtering(logits.clone(), top_k=4, top_p=1.0)
    assert torch.equal(filtered_logits_k4, expected_logits)

def test_top_p_filtering_only():
    """Test basic top-p (nucleus) filtering"""
    # Probabilities: [0.024, 0.643, 0.088, 0.244] -> Sorted: [0.643, 0.244, 0.088, 0.024]
    # Cumulative: [0.643, 0.887, 0.975, 1.0]
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    # With p=0.9, keep tokens with cumulative prob <= 0.9.
    # Indices kept: 1 (0.643), 3 (0.887), 2 (0.975 - this one is kept because it pushes over 0.9)
    # Indices removed: 0
    expected_logits = torch.tensor([[-float('Inf'), 4.0, 2.0, 3.0]]) # Updated expectation
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=0.9)
    assert torch.equal(filtered_logits, expected_logits)

def test_top_p_filtering_edge_cases():
    """Test top-p with p=0.0 (keep only highest) and p=1.0 (keep all)"""
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    # p=0.0 should effectively keep only the top token
    expected_logits_p0 = torch.tensor([[-float('Inf'), 4.0, -float('Inf'), -float('Inf')]])
    filtered_logits_p0 = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=0.0)
    assert torch.equal(filtered_logits_p0, expected_logits_p0)

    # p=1.0 should keep all tokens
    expected_logits_p1 = logits.clone()
    filtered_logits_p1 = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=1.0)
    assert torch.equal(filtered_logits_p1, expected_logits_p1)

def test_combined_filtering():
    """Test interaction of top-k and top-p filtering"""
    # Probabilities: [0.01, 0.2, 0.5, 0.09, 0.2] -> Sorted: [0.5, 0.2, 0.2, 0.09, 0.01]
    # Cumulative: [0.5, 0.7, 0.9, 0.99, 1.0]
    logits = torch.tensor([[1.0, 3.0, 4.0, 2.0, 3.0]])

    # top_k=3: keeps indices 1, 2, 4 (logits 3.0, 4.0, 3.0)
    # top_p=0.8: keeps indices 2, 1, 4 (logits 4.0, 3.0, 3.0, cumulative 0.5, 0.7, 0.9)
    # Combined: Intersection should keep indices 1, 2, 4
    # After top_k=3: [[-inf, 3.0, 4.0, -inf, 3.0]] -> Softmax -> Probs [~0, 0.26, 0.7, ~0, 0.26]
    # After top_p=0.8 on top_k results: Cumulative [0.7, 0.96, 0.99], keeps index 2
    # Correction: top_p applies first on full logits conceptually, then top_k
    # top_p=0.8 keeps indices 2, 1, 4 (logits 4.0, 3.0, 3.0)
    # top_k=3 applied to these results: Keeps all three (4.0, 3.0, 3.0)
    # Expected: keep indices 1, 2, 4

    expected_logits = torch.tensor([[-float('Inf'), 3.0, 4.0, -float('Inf'), 3.0]])
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=3, top_p=0.8)
    # print(f"\nCombined Filter Logits: {filtered_logits}") # Debug print
    assert torch.equal(filtered_logits, expected_logits)

def test_batch_filtering():
    """Test filtering with batch_size > 1"""
    logits = torch.tensor([
        [1.0, 3.0, 2.0], # top_k=1 -> [-inf, 3.0, -inf]
        [4.0, 0.0, 2.0]  # top_k=1 -> [4.0, -inf, -inf]
    ])
    expected_logits = torch.tensor([
        [-float('Inf'), 3.0, -float('Inf')],
        [4.0, -float('Inf'), -float('Inf')]
    ])
    filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=1, top_p=1.0)
    assert torch.equal(filtered_logits, expected_logits)

    # Test top_p with batch
    # Batch 1 probs: [0.09, 0.66, 0.24] -> Cum: [0.66, 0.90, 1.0] -> p=0.7 -> keep idx 1 (0.66), idx 2 (0.90)
    # Batch 2 probs: [0.84, 0.02, 0.14] -> Cum: [0.84, 0.98, 1.0] -> p=0.7 -> keep idx 0 (0.84)
    logits_p = torch.tensor([
        [1.0, 3.0, 2.0],
        [4.0, 0.0, 2.0]
    ])
    expected_logits_p = torch.tensor([
        [-float('Inf'), 3.0, 2.0],             # Updated expectation for batch 1
        [4.0, -float('Inf'), -float('Inf')]
    ])
    filtered_logits_p = top_k_top_p_filtering(logits_p.clone(), top_k=0, top_p=0.7)
    assert torch.equal(filtered_logits_p, expected_logits_p)

# --- Tests for generate_sample_text --- 

def test_generate_sample_text():
    """Test the generate_sample_text utility function."""
    mock_model = MockGenerationModel()
    mock_tokenizer = MockTokenizer()
    
    # Setup mock return values
    context = torch.tensor([[0, 1]]) # Mock input tensor
    generated_ids = torch.tensor([[0, 1, 2]]) # Mock output from model.generate
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "abc" # Mock decoded output
    
    # Call the function
    result = generate_sample_text(
        model=mock_model, 
        context=context, 
        max_new_tokens=1, 
        tokenizer=mock_tokenizer
    )
    
    # Assertions
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs
    assert torch.equal(call_kwargs['input_ids'], context)
    assert call_kwargs['max_new_tokens'] == 1
    
    mock_tokenizer.decode.assert_called_once_with(generated_ids[0].tolist())
    assert result == "abc"

# --- Tests for sample_text --- 

def test_sample_text_regular_tokenizer():
    """Test sample_text with a regular tokenizer."""
    device = 'cpu' # Explicitly set device for test predictability
    mock_model = MockGenerationModel(device=device)
    mock_tokenizer = MockTokenizer()
    
    prompt = "ab"
    encoded_prompt = torch.tensor([[0, 1]], device=device)
    generated_ids = torch.tensor([[0, 1, 2]])
    
    mock_tokenizer.encode.return_value = encoded_prompt
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "abc"
    
    result = sample_text(model=mock_model, prompt=prompt, max_length=1, tokenizer=mock_tokenizer, device=device)
    
    mock_tokenizer.encode.assert_called_once_with(prompt, return_tensors='pt')
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs
    # We need ANY here because the device might be cpu or cuda depending on availability - REMOVED
    # assert torch.equal(call_kwargs['input_ids'], encoded_prompt.to(ANY))
    # Check input_ids are on the correct device
    assert call_kwargs['input_ids'].device == torch.device(device)
    assert torch.equal(call_kwargs['input_ids'], encoded_prompt) 
    assert call_kwargs['max_new_tokens'] == 1
    
    mock_tokenizer.decode.assert_called_once_with(generated_ids[0].tolist())
    assert result == "abc"

def test_sample_text_char_level():
    """Test sample_text with a character-level tokenizer/dataset."""
    device = 'cpu' # Explicitly set device
    mock_model = MockGenerationModel(device=device)
    mock_char_level = MockCharLevel()
    
    prompt = "ab"
    # Encoded according to MockCharLevel: a=0, b=1
    encoded_prompt = torch.tensor([[0, 1]], device=device)
    generated_ids = torch.tensor([[0, 1, 2]]) # a, b, c
    
    mock_model.generate.return_value = generated_ids
    # Decode mock is handled by side_effect in MockCharLevel
    
    result = sample_text(model=mock_model, prompt=prompt, max_length=1, tokenizer=mock_char_level, device=device)
    
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs
    # assert torch.equal(call_kwargs['input_ids'], encoded_prompt.to(ANY))
    assert call_kwargs['input_ids'].device == torch.device(device)
    assert torch.equal(call_kwargs['input_ids'], encoded_prompt)
    assert call_kwargs['max_new_tokens'] == 1
    
    # Check that the internal decode mock was called correctly
    mock_char_level.decode.assert_called_once_with(generated_ids[0].tolist())
    assert result == "abc"

def test_sample_text_empty_prompt():
    """Test sample_text with an empty prompt."""
    device = 'cpu' # Explicitly set device
    mock_model = MockGenerationModel(device=device)
    mock_tokenizer = MockTokenizer()
    
    prompt = ""
    # Expect a default starting tensor, e.g., zeros
    default_start_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = torch.tensor([[0, 1, 2]])
    
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "abc"
    
    result = sample_text(model=mock_model, prompt=prompt, max_length=3, tokenizer=mock_tokenizer, device=device)
    
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs
    # assert torch.equal(call_kwargs['input_ids'], default_start_ids.to(ANY))
    assert call_kwargs['input_ids'].device == torch.device(device)
    assert torch.equal(call_kwargs['input_ids'], default_start_ids)
    assert call_kwargs['max_new_tokens'] == 3
    
    mock_tokenizer.decode.assert_called_once_with(generated_ids[0].tolist())
    assert result == "abc"

def test_sample_text_device_auto_detection():
    """Test that sample_text auto-detects the correct device (cuda if available, else cpu)"""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    # Ensure it doesn't accidentally have char_to_idx to avoid the wrong code path
    if hasattr(mock_tokenizer, 'char_to_idx'):
        delattr(mock_tokenizer, 'char_to_idx')
    # Simulate tokenizer returning tensor on CPU initially
    encoded_tensor_cpu = torch.tensor([[1, 2, 3]], device='cpu')
    mock_tokenizer.encode.return_value = encoded_tensor_cpu
    mock_tokenizer.decode.return_value = "Generated text"
    # Simulate model generating tensor on CPU initially
    generated_tensor_cpu = torch.tensor([[1, 2, 3, 4]], device='cpu')
    mock_model.generate.return_value = generated_tensor_cpu

    # Test with CUDA available
    # Use 'cuda' for model.to check, but check type for tensor device
    expected_cuda_device = torch.device('cuda')
    with patch('src.craft.utils.generation.torch.cuda.is_available', return_value=True):
        sample_text(mock_model, "test prompt", tokenizer=mock_tokenizer)

        # Assert model was moved to CUDA (without index)
        mock_model.to.assert_called_with(expected_cuda_device)

        # Assert input tensor passed to generate is on a CUDA device (ignore index)
        generate_call_args, generate_call_kwargs = mock_model.generate.call_args
        input_ids_tensor = generate_call_kwargs.get('input_ids')
        assert input_ids_tensor is not None, "input_ids not found in generate kwargs"
        assert input_ids_tensor.device.type == expected_cuda_device.type, \
               f"Input tensor device type should be {expected_cuda_device.type}, but was {input_ids_tensor.device.type} ({input_ids_tensor.device})"

    mock_model.reset_mock()
    mock_tokenizer.reset_mock()

    # Test with CUDA not available
    expected_cpu_device = torch.device('cpu')
    with patch('src.craft.utils.generation.torch.cuda.is_available', return_value=False):
        # Need to re-assign return value as the tensor might have been modified in place if .to() returned self
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]], device='cpu')
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]], device='cpu')

        sample_text(mock_model, "test prompt", tokenizer=mock_tokenizer)

        # Assert model was moved to CPU
        mock_model.to.assert_called_with(expected_cpu_device)

        # Assert input tensor passed to generate is on CPU
        generate_call_args, generate_call_kwargs = mock_model.generate.call_args
        input_ids_tensor = generate_call_kwargs.get('input_ids')
        assert input_ids_tensor is not None, "input_ids not found in generate kwargs"
        assert input_ids_tensor.device.type == expected_cpu_device.type, \
               f"Input tensor device type should be {expected_cpu_device.type}, but was {input_ids_tensor.device.type} ({input_ids_tensor.device})"