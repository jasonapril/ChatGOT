#!/usr/bin/env python
"""
Tests for src.craft.utils.generation
"""
import torch
import pytest
from unittest.mock import MagicMock, ANY
from unittest.mock import patch

# Import the module/functions under test
from craft.training.generation import TextGenerator # Import the class
from craft.utils.generation import top_k_top_p_filtering # Add this import

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

class MockChar:
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

# --- Tests for generate_sample_text (using mock Model) --- #

def test_generate_sample_text():
    """Test the generate_sample_text utility function."""
    mock_model = MockGenerationModel()
    mock_tokenizer = MockTokenizer()
    device = torch.device("cpu")

    # Instantiate the generator
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)

    # Setup mock return values
    context = torch.tensor([[0, 1]]) # Mock input tensor
    generated_ids = torch.tensor([[0, 1, 2]]) # Mock output from model.generate
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "abc" # Mock decoded output
    mock_tokenizer.encode.return_value = context.tolist() # Mock encode to return list

    # Call the function
    result = generator.generate_text(
        start_prompt="ab", # Prompt corresponding to context [0, 1]
        max_new_tokens=1, # Generate 1 new token (ID 2)
        do_sample=False # Use greedy for predictability
    )

    # Assertions
    # 1. Check model.generate was called (inside generator.generate_text)
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert torch.equal(call_kwargs['input_ids'], context.to(device))
    assert call_kwargs['max_new_tokens'] == 1
    assert not call_kwargs['do_sample'] # Greedy

    # 2. Check tokenizer.decode was called (inside generator.generate_text)
    # It should be called with the generated sequence excluding the prompt
    mock_tokenizer.decode.assert_called_once_with(generated_ids[:, context.shape[1]:].tolist()[0]) # Decode [2]
    
    # 3. Check the final result
    assert result == ["abc"] # generate_text returns a list of strings

def test_generate_sample_text_params_forwarding():
    """Test that generation parameters are forwarded correctly."""
    mock_model = MockGenerationModel()
    mock_tokenizer = MockTokenizer()
    device = torch.device("cpu")
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)

    context = torch.tensor([[0, 1]])
    generated_ids = torch.tensor([[0, 1, 2]])
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "decoded"
    mock_tokenizer.encode.return_value = context.tolist()

    generator.generate_text(
        start_prompt="ab",
        max_new_tokens=10,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        verbose=True
    )

    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert torch.equal(call_kwargs['input_ids'], context)
    assert call_kwargs['max_new_tokens'] == 10
    assert call_kwargs['temperature'] == 0.7
    assert call_kwargs['top_k'] == 50
    assert call_kwargs['top_p'] == 0.95
    assert call_kwargs['repetition_penalty'] == 1.2
    assert call_kwargs['verbose'] is True
    # Assert decode is called with *only* the generated part
    generated_part = generated_ids[:, context.shape[1]:].tolist()[0]
    mock_tokenizer.decode.assert_called_once_with(generated_part)

def test_generate_sample_text_max_tokens_zero():
    """Test generate_sample_text with max_new_tokens=0."""
    mock_model = MockGenerationModel()
    mock_tokenizer = MockTokenizer()
    device = torch.device("cpu")
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)

    context = torch.tensor([[0, 1]])
    # Expect generate to return only the context if max_new_tokens is 0
    # (or whatever the underlying model.generate does - we mock it)
    generated_ids_zero = context.clone()
    mock_model.generate.return_value = generated_ids_zero
    mock_tokenizer.decode.return_value = "ab" # Decode of context [0, 1]
    mock_tokenizer.encode.return_value = context.tolist()

    result = generator.generate_text(
        start_prompt="ab",
        max_new_tokens=0
    )

    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert call_kwargs['max_new_tokens'] == 0
    # If max_new_tokens is 0, the generated part is empty
    mock_tokenizer.decode.assert_called_once_with([])
    assert result == ["ab"] # Assuming decode of context [0, 1] is "ab"

# --- Tests for sample_text --- 

def test_sample_text_regular_tokenizer():
    """Test sample_text with a regular tokenizer."""
    device = 'cpu' # Explicitly set device for test predictability # TODO: Is this helpful?
    mock_model = MockGenerationModel(device=device)
    mock_tokenizer = MockTokenizer()
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)
    
    prompt = "ab"
    encoded_prompt = torch.tensor([[0, 1]], device=device)
    generated_ids = torch.tensor([[0, 1, 2]])
    
    mock_tokenizer.encode.return_value = encoded_prompt.tolist() # Needs list
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "abc"
    
    result = generator.generate_text(start_prompt=prompt, max_new_tokens=1)
    
    mock_tokenizer.encode.assert_called_once_with(prompt)
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert torch.equal(call_kwargs['input_ids'], encoded_prompt)
    assert call_kwargs['max_new_tokens'] == 1
    
    # Assert decode is called with *only* the generated part
    generated_part = generated_ids[:, encoded_prompt.shape[1]:].tolist()[0]
    mock_tokenizer.decode.assert_called_once_with(generated_part)
    assert result == ["abc"] # Returns list

def test_sample_text_params_forwarding():
    """Test that generation parameters are forwarded correctly in sample_text."""
    device = 'cpu'
    mock_model = MockGenerationModel(device=device)
    mock_tokenizer = MockTokenizer()
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)
    prompt = "test"
    encoded_prompt = torch.tensor([[1, 2, 3, 4]], device=device)
    generated_ids = torch.tensor([[1, 2, 3, 4, 5]])

    mock_tokenizer.encode.return_value = encoded_prompt.tolist()
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "decoded"

    generator.generate_text(
        start_prompt=prompt,
        max_new_tokens=10, # max_length -> max_new_tokens
        temperature=0.6,
        top_k=30,
        top_p=0.85,
        verbose=True
    )

    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert torch.equal(call_kwargs['input_ids'], encoded_prompt)
    assert call_kwargs['max_new_tokens'] == 10
    assert call_kwargs['temperature'] == 0.6
    assert call_kwargs['top_k'] == 30
    assert call_kwargs['top_p'] == 0.85
    # Assert decode is called with *only* the generated part
    generated_part = generated_ids[:, encoded_prompt.shape[1]:].tolist()[0]
    mock_tokenizer.decode.assert_called_once_with(generated_part)

def test_sample_text_max_length_zero():
    """Test sample_text with max_length=0."""
    device = 'cpu'
    mock_model = MockGenerationModel(device=device)
    mock_tokenizer = MockTokenizer()
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)
    prompt = "test"
    encoded_prompt = torch.tensor([[1, 2, 3, 4]], device=device)
    # Expect generate to return only the prompt if max_length is 0
    generated_ids_zero = encoded_prompt.clone()
    mock_model.generate.return_value = generated_ids_zero
    mock_tokenizer.encode.return_value = encoded_prompt.tolist()
    mock_tokenizer.decode.return_value = "test" # Decode of prompt

    result = generator.generate_text(
        start_prompt=prompt,
        max_new_tokens=0
    )

    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert call_kwargs['max_new_tokens'] == 0
    # Check decode was called with the empty generated part
    mock_tokenizer.decode.assert_called_once_with([]) # Decode empty generated part
    assert result == ["test"] # Returns list

def test_sample_text_char():
    """Test sample_text with a character-level tokenizer/dataset."""
    device = 'cpu' # Explicitly set device # TODO: Why are we forcing CPU?
    mock_model = MockGenerationModel(device=device)
    mock_char = MockChar()
    generator = TextGenerator(model=mock_model, dataset=mock_char, device=device)
    
    prompt = "ab"
    # Encoded according to MockChar: a=0, b=1
    encoded_prompt = torch.tensor([[0, 1]], device=device)
    generated_ids = torch.tensor([[0, 1, 2]]) # a, b, c
    
    mock_model.generate.return_value = generated_ids
    # Decode mock is handled by side_effect in MockChar
    
    result = generator.generate_text(start_prompt=prompt, max_new_tokens=1)
    
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert torch.equal(call_kwargs['input_ids'], encoded_prompt)
    assert call_kwargs['max_new_tokens'] == 1
    
    # Check that the internal decode mock was called correctly with the generated part
    generated_part = generated_ids[:, encoded_prompt.shape[1]:].tolist()[0]
    mock_char.decode.assert_called_once_with(generated_part)
    # Assert the result is only the decoded *generated* part
    assert result == ["c"] # Only the generated part 'c' should be returned

def test_sample_text_empty_prompt():
    """Test sample_text with an empty prompt."""
    device = 'cpu' # Explicitly set device # TODO: Why are we forcing CPU?
    mock_model = MockGenerationModel(device=device)
    mock_tokenizer = MockTokenizer()
    generator = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=device)
    
    prompt = ""
    # Expect a default starting tensor, e.g., zeros, handled by model.generate
    # TextGenerator will encode empty string
    encoded_empty_prompt = torch.tensor([[]], dtype=torch.long, device=device)
    mock_tokenizer.encode.return_value = [] # Empty list for empty string
    generated_ids = torch.tensor([[0, 1, 2]])
    
    mock_model.generate.return_value = generated_ids
    mock_tokenizer.decode.return_value = "abc"
    
    result = generator.generate_text(start_prompt=prompt, max_new_tokens=3)
    
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    # Check input_ids was based on encoded empty prompt
    assert call_kwargs['input_ids'].shape[1] == 0 # Check if input shape reflects empty prompt
    assert call_kwargs['max_new_tokens'] == 3
    # Assert decode is called with the generated part (which is the full output here)
    mock_tokenizer.decode.assert_called_once_with(generated_ids[0].tolist()) # Full output is generated part

    assert result == ["abc"] # Returns list

def test_sample_text_device_auto_detection():
    """Test that sample_text auto-detects the correct device (cuda if available, else cpu)"""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    # Ensure it doesn't accidentally have char_to_idx to avoid the wrong code path
    if hasattr(mock_tokenizer, 'char_to_idx'):
        delattr(mock_tokenizer, 'char_to_idx')
    # Simulate tokenizer returning tensor on CPU initially
    encoded_tensor_cpu = torch.tensor([[1, 2, 3]], device='cpu')
    mock_tokenizer.encode.return_value = encoded_tensor_cpu.tolist()
    mock_tokenizer.decode.return_value = "Generated text"
    # Simulate model generating tensor on CPU initially
    generated_tensor_cpu = torch.tensor([[1, 2, 3, 4]], device='cpu')
    mock_model.generate.return_value = generated_tensor_cpu
    # Mock model.to
    mock_model.to = MagicMock(return_value=mock_model)

    # Test with CUDA available
    # Use 'cuda' for model.to check, but check type for tensor device
    expected_cuda_device = torch.device('cuda')
    with patch('src.craft.utils.generation.torch.cuda.is_available', return_value=True):
        # Instantiate TextGenerator, device='auto' is implicit if None
        generator_cuda = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=expected_cuda_device)
        # Call generate
        generator_cuda.generate_text(start_prompt="test prompt", max_new_tokens=1)

    # Assert model was moved to CUDA
    mock_model.to.assert_called_with(expected_cuda_device) # Check model moved during TextGenerator init
    # Assert tensors passed to generate were on CUDA
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert call_kwargs['input_ids'].device.type == expected_cuda_device.type

    # Reset mocks and test CPU path
    mock_model.reset_mock()
    mock_tokenizer.reset_mock()
    mock_model.to = MagicMock(return_value=mock_model) # Re-mock .to
    mock_model.generate.return_value = generated_tensor_cpu # Reset generate return
    mock_tokenizer.encode.return_value = encoded_tensor_cpu.tolist() # Reset encode return

    expected_cpu_device = torch.device('cpu')
    with patch('src.craft.utils.generation.torch.cuda.is_available', return_value=False):
        # Instantiate TextGenerator, device='auto' is implicit if None
        generator_cpu = TextGenerator(model=mock_model, tokenizer=mock_tokenizer, device=expected_cpu_device)
        # Call generate
        generator_cpu.generate_text(start_prompt="test prompt", max_new_tokens=1)

    # Assert model was moved to CPU
    mock_model.to.assert_called_with(expected_cpu_device) # Check model moved during TextGenerator init
    # Assert tensors passed to generate were on CPU
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args
    assert call_kwargs['input_ids'].device.type == expected_cpu_device.type