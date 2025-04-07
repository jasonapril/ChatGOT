"""
Tests for standalone text generation sampling functions.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock, call
import torch.nn.functional as F

# Import functions to test
from craft.training.sampling import generate_text_manual_sampling, generate_samples_manual_sampling
from craft.data.tokenizers.base import Tokenizer # Correct base class name

# Basic fixture for character mappings
@pytest.fixture
def char_maps():
    char_to_idx = {'<unk>': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, ' ': 5}
    idx_to_char = {0: '<unk>', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: ' '}
    return char_to_idx, idx_to_char

# Mock Tokenizer fixture based on char_maps
@pytest.fixture
def mock_tokenizer(char_maps):
    char_to_idx, idx_to_char = char_maps
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.encode.side_effect = lambda text: [char_to_idx.get(c, 0) for c in text] # Simple char encoding
    tokenizer.decode.side_effect = lambda ids: "".join([idx_to_char.get(i, '<unk>') for i in ids])
    tokenizer.pad_token_id = None # Assume no padding needed for these tests
    tokenizer.eos_token_id = None # Assume no EOS needed for these tests
    tokenizer.vocab_size = len(char_to_idx)
    return tokenizer

@pytest.fixture
def mock_model():
    model = MagicMock(spec=torch.nn.Module)
    # Simulate returning logits for vocab size 6 (unk, a, b, c, d, space)
    # Make 'b' (index 2) the most likely next token after 'a' (index 1)
    def model_side_effect(input_ids):
        # Simple deterministic behavior for testing
        # Last token in context determines next output distribution
        last_token_idx = input_ids[0, -1].item()
        if last_token_idx == 1: # if last token is 'a'
            # Make 'b' most likely
            logits = torch.tensor([[-10.0, -10.0, 1.0, -10.0, -10.0, -10.0]], dtype=torch.float)
        elif last_token_idx == 2: # if last token is 'b'
             # Make 'c' most likely
             logits = torch.tensor([[-10.0, -10.0, -10.0, 1.0, -10.0, -10.0]], dtype=torch.float)
        elif last_token_idx == 3: # if last token is 'c'
            # Make 'd' most likely
            logits = torch.tensor([[-10.0, -10.0, -10.0, -10.0, 1.0, -10.0]], dtype=torch.float)
        else: # Default, make space most likely
            logits = torch.tensor([[-10.0, -10.0, -10.0, -10.0, -10.0, 1.0]], dtype=torch.float)
        # Return shape (batch_size, sequence_length, vocab_size)
        # For simplicity, assume model only looks at last token and returns next token logits
        # We need to expand dims to match typical model output
        return logits.unsqueeze(1) # -> (1, 1, 6)

    model.side_effect = model_side_effect # Correct: Use side_effect
    # Mock device attribute if needed
    model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))] # Make it iterable
    return model

# --- Tests for generate_text_sampling ---

def test_generate_basic_manual_sampling(mock_model, mock_tokenizer):
    seed_text = "a"
    max_length = 3 # Generate 3 more chars: b, c, d
    device = torch.device('cpu')

    # Use temperature 1 for predictable sampling based on mocked logits
    # Patch multinomial to make it deterministic for the test
    with patch('torch.multinomial') as mock_multinomial:
        # Make multinomial return the argmax (most likely token)
        def multinomial_side_effect(probs, num_samples):
            # probs shape is (batch_size, vocab_size)
            return torch.argmax(probs, dim=1, keepdim=True)
        mock_multinomial.side_effect = multinomial_side_effect

        generated_text = generate_text_manual_sampling(
            model=mock_model,
            tokenizer=mock_tokenizer,
            seed_text=seed_text,
            max_length=max_length,
            temperature=1.0, # No scaling
            device=device,
            top_k=0, # Disabled
            top_p=0.0, # Disabled
            repetition_penalty=1.0 # Disabled
        )

    # Expected sequence: a (seed) -> b -> c -> d
    assert generated_text == "abcd"

    # Verify model calls
    assert mock_model.call_count == max_length
    # Instead, check call arguments manually using torch.equal and .args/.kwargs
    assert len(mock_model.call_args_list) == max_length # Should be 3 calls

    # Check call 1
    call1_args, call1_kwargs = mock_model.call_args_list[0]
    expected_call1_arg = torch.tensor([[1]], device=device) # Initial context 'a'
    assert len(call1_args) == 1
    assert torch.equal(call1_args[0], expected_call1_arg), f"Call 1 arg mismatch: Got {call1_args[0]}, Expected {expected_call1_arg}"
    assert call1_kwargs == {}, f"Call 1 kwargs mismatch: Got {call1_kwargs}"

    # Check call 2
    call2_args, call2_kwargs = mock_model.call_args_list[1]
    expected_call2_arg = torch.tensor([[1, 2]], device=device) # Context 'ab'
    assert len(call2_args) == 1
    assert torch.equal(call2_args[0], expected_call2_arg), f"Call 2 arg mismatch: Got {call2_args[0]}, Expected {expected_call2_arg}"
    assert call2_kwargs == {}, f"Call 2 kwargs mismatch: Got {call2_kwargs}"

    # Check call 3
    call3_args, call3_kwargs = mock_model.call_args_list[2]
    expected_call3_arg = torch.tensor([[1, 2, 3]], device=device) # Context 'abc'
    assert len(call3_args) == 1
    assert torch.equal(call3_args[0], expected_call3_arg), f"Call 3 arg mismatch: Got {call3_args[0]}, Expected {expected_call3_arg}"
    assert call3_kwargs == {}, f"Call 3 kwargs mismatch: Got {call3_kwargs}"

    # Verify multinomial calls (sampling)
    assert mock_multinomial.call_count == max_length

def test_generate_manual_sampling_temperature(mock_model, mock_tokenizer):
    seed_text = "a"
    max_length = 1 # Only need one step
    device = torch.device('cpu')

    # --- Low Temperature (more deterministic) ---
    # Patch multinomial to be deterministic (argmax)
    with patch('torch.multinomial') as mock_multinomial_low_temp:
        mock_multinomial_low_temp.side_effect = lambda probs, num_samples: torch.argmax(probs, dim=1, keepdim=True)
        generated_text_low_temp = generate_text_manual_sampling(
            model=mock_model, # Model still prefers 'b' after 'a'
            tokenizer=mock_tokenizer,
            seed_text=seed_text,
            max_length=max_length,
            temperature=0.1, # Very low temperature
            device=device
        )
    assert generated_text_low_temp == "ab" # Expect the most likely token 'b'

    # --- High Temperature (more random) ---
    # Instead of checking output (hard with randomness), check probabilities passed to multinomial
    # Patch F.softmax to capture the probabilities
    captured_probs_high_temp = None
    original_softmax = F.softmax

    def softmax_wrapper(*args, **kwargs):
        nonlocal captured_probs_high_temp
        # Call the original softmax
        result = original_softmax(*args, **kwargs)
        # Capture the result (probabilities) just before multinomial sampling
        captured_probs_high_temp = result.clone()
        return result

    with patch('torch.nn.functional.softmax', new=softmax_wrapper):
         # We still need multinomial to return *something*
        with patch('torch.multinomial') as mock_multinomial_high_temp:
             # Return a fixed value, e.g., index 3 ('c'), doesn't matter for this check
            mock_multinomial_high_temp.return_value = torch.tensor([[3]])
            generate_text_manual_sampling(
                model=mock_model,
                tokenizer=mock_tokenizer,
                seed_text=seed_text,
                max_length=max_length,
                temperature=10.0, # Very high temperature
                device=device
            )

    assert captured_probs_high_temp is not None, "Softmax was not called or probabilities not captured"
    # Check that probabilities are flatter (less variance) with high temperature
    # The difference between max and min prob should be smaller
    prob_variance_high_temp = torch.var(captured_probs_high_temp).item()

    # --- Compare with Low Temperature Probs ---
    captured_probs_low_temp = None
    def softmax_wrapper_low_temp(*args, **kwargs):
        nonlocal captured_probs_low_temp
        result = original_softmax(*args, **kwargs)
        captured_probs_low_temp = result.clone()
        return result

    with patch('torch.nn.functional.softmax', new=softmax_wrapper_low_temp):
        with patch('torch.multinomial') as mock_multinomial_low_temp_check:
            mock_multinomial_low_temp_check.return_value = torch.tensor([[2]]) # Return 'b'
            generate_text_manual_sampling(
                model=mock_model,
                tokenizer=mock_tokenizer,
                seed_text=seed_text,
                max_length=max_length,
                temperature=0.1, # Very low temperature
                device=device
            )

    assert captured_probs_low_temp is not None
    prob_variance_low_temp = torch.var(captured_probs_low_temp).item()

    # High temp variance should be significantly lower than low temp variance
    assert prob_variance_high_temp < prob_variance_low_temp

def test_generate_manual_sampling_top_k(mock_model, mock_tokenizer):
    seed_text = "a" # Model output after 'a' is [-10, -10, 1.0, -10, -10, -10]
    max_length = 1
    device = torch.device('cpu')
    top_k_value = 2

    # Modify model output for this test to have more distinct high probabilities
    # Make 'b' (idx 2) and 'c' (idx 3) the top 2
    def model_side_effect_topk(input_ids):
        last_token_idx = input_ids[0, -1].item()
        if last_token_idx == 1: # if last token is 'a'
             # Logits: unk, a,  b,  c,   d, space
             logits = torch.tensor([[-10.0, -10.0, 1.0, 0.5, -1.0, -10.0]], dtype=torch.float)
        else:
             logits = torch.tensor([[-10.0]*6], dtype=torch.float) # Default unlikely
        return logits.unsqueeze(1)
    mock_model.side_effect = model_side_effect_topk # Correct

    captured_probs = None
    original_softmax = F.softmax
    def softmax_wrapper(*args, **kwargs):
        nonlocal captured_probs
        # Capture the *input* to softmax (logits after filtering)
        logits_input = args[0].clone()
        # Verify only top_k logits are not -inf
        assert torch.isneginf(logits_input).sum().item() == logits_input.numel() - top_k_value
        result = original_softmax(*args, **kwargs)
        # Capture the final probabilities
        captured_probs = result.clone()
        return result

    with patch('torch.nn.functional.softmax', new=softmax_wrapper):
        with patch('torch.multinomial') as mock_multinomial:
            # Return index 2 ('b'), the most likely of the top-k
            mock_multinomial.return_value = torch.tensor([[2]])
            generate_text_manual_sampling(
                model=mock_model,
                tokenizer=mock_tokenizer,
                seed_text=seed_text,
                max_length=max_length,
                temperature=1.0,
                device=device,
                top_k=top_k_value
            )

    assert captured_probs is not None, "Probabilities not captured"
    # Verify that only the top k probabilities are non-zero
    # The softmax of -inf is 0
    non_zero_probs = (captured_probs > 0).sum().item()
    assert non_zero_probs == top_k_value
    # Check that the non-zero probabilities correspond to indices 2 ('b') and 3 ('c')
    assert captured_probs[0, 2].item() > 0
    assert captured_probs[0, 3].item() > 0

def test_generate_manual_sampling_top_p(mock_model, mock_tokenizer):
    seed_text = "a"
    max_length = 1
    device = torch.device('cpu')
    top_p_value = 0.9 # Example value

    # Logits for seed 'a': [-10, -10, 1.0, 0.5, -1.0, -10]
    # Corresponds roughly to probs after softmax (T=1): [~0, ~0, 0.60, 0.36, 0.04, ~0]
    # Cumulative probs: [~0, ~0, 0.60, 0.96, 1.00, 1.00]
    # For top_p=0.9, we keep indices 2 ('b') and 3 ('c') because cumsum(0.60 + 0.36) = 0.96 > 0.9
    expected_kept_indices = {2, 3}

    # Use the same modified model output as top_k test
    def model_side_effect_topp(input_ids):
        last_token_idx = input_ids[0, -1].item()
        if last_token_idx == 1: # if last token is 'a'
             logits = torch.tensor([[-10.0, -10.0, 1.0, 0.5, -1.0, -10.0]], dtype=torch.float)
        else:
             logits = torch.tensor([[-10.0]*6], dtype=torch.float)
        return logits.unsqueeze(1)
    mock_model.side_effect = model_side_effect_topp # Correct

    captured_final_logits = None
    original_softmax = F.softmax
    def softmax_wrapper(*args, **kwargs):
        nonlocal captured_final_logits
        # Capture the input logits to the *final* softmax call
        captured_final_logits = args[0].clone()
        return original_softmax(*args, **kwargs)

    with patch('torch.nn.functional.softmax', new=softmax_wrapper):
        with patch('torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([[2]]) # Doesn't matter
            generate_text_manual_sampling(
                model=mock_model,
                tokenizer=mock_tokenizer,
                seed_text=seed_text,
                max_length=max_length,
                temperature=1.0,
                device=device,
                top_p=top_p_value
            )

    assert captured_final_logits is not None, "Final logits not captured"

    # Check which logits were *not* set to -inf
    kept_logits_mask = ~torch.isneginf(captured_final_logits[0])
    kept_indices = set(torch.where(kept_logits_mask)[0].tolist())

    assert kept_indices == expected_kept_indices

def test_generate_manual_sampling_repetition_penalty(mock_model, mock_tokenizer):
    seed_text = "ab"
    max_length = 1 # We only care about the step predicting after 'ab'
    device = torch.device('cpu')
    repetition_penalty_value = 1.5

    # Model normally predicts 'c' (idx 3) after 'b' (idx 2) with logit 1.0
    # Let's modify it to also give 'a' (idx 1) a high logit
    original_logit_for_a = 0.8
    def model_side_effect_rep_pen(input_ids):
        last_token_idx = input_ids[0, -1].item()
        if last_token_idx == 2: # if last token is 'b'
             # Make 'c' (3) most likely, but 'a' (1) also likely
             # Logits: unk,      a,    b,   c,    d, space
             logits = torch.tensor([[-10.0, original_logit_for_a, -10.0, 1.0, -10.0, -10.0]], dtype=torch.float)
        else:
             logits = torch.tensor([[-10.0]*6], dtype=torch.float)
        return logits.unsqueeze(1)
    mock_model.side_effect = model_side_effect_rep_pen # Correct

    captured_logits_before_softmax = None
    original_softmax = F.softmax
    def softmax_wrapper(*args, **kwargs):
        nonlocal captured_logits_before_softmax
        captured_logits_before_softmax = args[0].clone()
        return original_softmax(*args, **kwargs)

    with patch('torch.nn.functional.softmax', new=softmax_wrapper):
        with patch('torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([[3]]) # Predict 'c'
            generate_text_manual_sampling(
                model=mock_model,
                tokenizer=mock_tokenizer,
                seed_text=seed_text,
                max_length=max_length,
                temperature=1.0,
                device=device,
                repetition_penalty=repetition_penalty_value
            )

    assert captured_logits_before_softmax is not None

    # Check that the logit for 'a' (index 1), which is in context "ab", was penalized
    penalized_logit_for_a = captured_logits_before_softmax[0, 1].item()
    expected_penalized_logit = original_logit_for_a / repetition_penalty_value
    # Use approx due to floating point
    assert penalized_logit_for_a == pytest.approx(expected_penalized_logit)

    # Check that the logit for 'c' (index 3), which is not in context, was NOT penalized
    logit_for_c = captured_logits_before_softmax[0, 3].item()
    assert logit_for_c == pytest.approx(1.0) # Original logit from mock model

def test_generate_manual_sampling_unknown_seed_chars(mock_model, mock_tokenizer):
    """Test handling of unknown characters in the seed text."""
    seed_text = "axb"
    max_length = 1
    device = torch.device('cpu')

    # 'x' is not in char_to_idx, should map to <unk> (index 0)
    expected_initial_context = torch.tensor([[1, 0, 2]], device=device) # a=<unk>b

    # Patch multinomial to return a known next token, e.g., 'c' (index 3)
    with patch('torch.multinomial') as mock_multinomial:
        mock_multinomial.return_value = torch.tensor([[3]])

        generate_text_manual_sampling(
            model=mock_model,
            tokenizer=mock_tokenizer,
            seed_text=seed_text,
            max_length=max_length,
            temperature=1.0,
            device=device
        )

    # Assert the model was first called with the correct context including <unk>
    # mock_model.assert_any_call(expected_initial_context) # Fails due to tensor comparison
    # Check the *first* call specifically
    first_call_args = mock_model.call_args_list[0]
    assert torch.equal(first_call_args[0][0], expected_initial_context)

def test_generate_manual_sampling_context_truncation(mock_model, mock_tokenizer):
    """Test that context passed to model is truncated correctly."""
    seed_text = "abc" * 5 # Length 15
    max_length = 2 # Generate 2 more chars
    device = torch.device('cpu')
    model_max_len = 10 # Set a small max length for testing

    # Mock model.config.max_seq_length
    mock_model.config = MagicMock()
    mock_model.config.max_seq_length = model_max_len

    # Context will be [a,b,c,a,b,c,a,b,c,a,b,c,a,b,c] -> indices [1,2,3]*5
    initial_context = torch.tensor([[1,2,3]*5], device=device)

    # Patch multinomial - return space (5) then 'a' (1)
    with patch('torch.multinomial') as mock_multinomial:
        mock_multinomial.side_effect = [torch.tensor([[5]]), torch.tensor([[1]])]

        generate_text_manual_sampling(
            model=mock_model,
            tokenizer=mock_tokenizer,
            seed_text=seed_text,
            max_length=max_length,
            temperature=1.0,
            device=device
        )

    assert mock_model.call_count == max_length

    # First call: context is initial_context (len 15) -> exceeds 10
    # Should be truncated to last 10: [c,a,b,c,a,b,c,a,b,c] -> [3,1,2,3,1,2,3,1,2,3]
    expected_first_call_context = torch.tensor([[3,1,2,3,1,2,3,1,2,3]], device=device)
    first_call_arg = mock_model.call_args_list[0].args[0] # Access via .args
    # Check shape and content separately
    assert first_call_arg.shape[1] == model_max_len, "First call context length is wrong"
    assert torch.equal(first_call_arg, expected_first_call_context), "First call context content is wrong"

    # Second call: context = first_call_context + space (5)
    # Context: [3,1,2,3,1,2,3,1,2,3,5] (len 11) -> exceeds 10
    # Should be truncated to last 10: [1,2,3,1,2,3,1,2,3,5]
    expected_second_call_context = torch.tensor([[1,2,3,1,2,3,1,2,3,5]], device=device)
    second_call_arg = mock_model.call_args_list[1][0][0] # model(context)
    # Check shape and content separately
    assert second_call_arg.shape[1] == model_max_len, "Second call context length is wrong"
    assert torch.equal(second_call_arg, expected_second_call_context), "Second call context content is wrong"

def test_generate_manual_sampling_default_context_truncation(mock_model, mock_tokenizer):
    """Test context truncation uses default 1024 if model lacks config."""
    char_to_idx, idx_to_char = char_maps()
    default_max_len = 1024
    # Seed text exactly at the default limit
    seed_text = "a" * default_max_len # Length 1024
    max_length = 2 # Generate 2 more chars
    device = torch.device('cpu')

    # Ensure mock model *doesn't* have config or max_seq_length
    if hasattr(mock_model, 'config'):
        del mock_model.config
    # Or ensure config doesn't have the attribute
    # mock_model.config = MagicMock(spec=[]) # An empty spec config

    initial_context_len = len(seed_text)
    assert initial_context_len == default_max_len

    # Patch multinomial - return 'b' (2) then 'c' (3)
    with patch('torch.multinomial') as mock_multinomial:
        mock_multinomial.side_effect = [torch.tensor([[2]]), torch.tensor([[3]])]

        generate_text_manual_sampling(
            model=mock_model,
            tokenizer=mock_tokenizer,
            seed_text=seed_text,
            max_length=max_length,
            temperature=1.0,
            device=device
        )

    assert mock_model.call_count == max_length

    # First call inside loop: context is seed_text (len 1024). No truncation needed.
    first_call_arg = mock_model.call_args_list[0][0][0] # model(context)
    assert first_call_arg.shape[1] == initial_context_len

    # Second call inside loop: context = seed_text + 'b' (len 1025). Truncation expected.
    # The context passed to the model should be truncated to the default length (1024).
    second_call_arg = mock_model.call_args_list[1][0][0] # model(context)
    assert second_call_arg.shape[1] == default_max_len
    # Verify the content is the *last* 1024 tokens: ('a'*1023 + 'b')
    expected_second_call_context_list = [char_to_idx['a']] * (default_max_len - 1) + [char_to_idx['b']]
    expected_second_call_context = torch.tensor([expected_second_call_context_list], device=device)
    assert torch.equal(second_call_arg, expected_second_call_context)

# TODO: Add tests for other edge cases (e.g., empty seed)

# --- Tests for sample_text ---

@patch('craft.training.sampling.generate_text_manual_sampling')
def test_generate_samples_manual_sampling_calls_generate(mock_manual_sampling, mock_model, mock_tokenizer):
    """Test that generate_samples_manual_sampling calls generate_text_manual_sampling correctly."""
    prompts = ["a", "b"]
    max_new_tokens = 5
    device = torch.device('cpu')
    gen_kwargs = {'temperature': 0.9, 'top_k': 10}

    # Mock the return value of the underlying generate_text_manual_sampling
    # It receives seed_text, return seed + "_gen"
    mock_manual_sampling.side_effect = lambda seed_text, **kwargs: seed_text + "_gen"

    results = generate_samples_manual_sampling(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        device=device,
        **gen_kwargs
    )

    # Check the results
    assert results == ["a_gen", "b_gen"], f"Expected ['a_gen', 'b_gen'], got {results}"

    # Check that generate_text_manual_sampling was called for each prompt
    assert mock_manual_sampling.call_count == len(prompts)

    # Check arguments for the first call (to generate_text_manual_sampling)
    call1_args, call1_kwargs = mock_manual_sampling.call_args_list[0]
    assert call1_kwargs['model'] == mock_model
    assert call1_kwargs['tokenizer'] == mock_tokenizer
    assert call1_kwargs['seed_text'] == prompts[0]
    assert call1_kwargs['max_length'] == max_new_tokens
    assert call1_kwargs['device'] == device
    assert call1_kwargs['temperature'] == gen_kwargs['temperature']
    assert call1_kwargs['top_k'] == gen_kwargs['top_k']

@patch('craft.training.sampling.logging.info')
@patch('craft.training.sampling.generate_text_manual_sampling')
@patch('craft.training.sampling.time.time') # Mock time to control duration
def test_sample_text_logging(mock_time, mock_generate, mock_log_info, mock_model, mock_tokenizer):
    """Test logging messages in generate_samples_manual_sampling."""
    prompts = ["a"]
    max_new_tokens = 1

    # Mock the underlying generator
    mock_generate.return_value = "a_gen"

    # Mock time to simulate duration
    mock_time.side_effect = [100.0, 105.0] # Start time, end time

    generate_samples_manual_sampling(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        log_samples=True # Enable logging
    )

    # Check logging calls
    expected_calls = [
        call("Generating 1 samples with max_new_tokens=1..."), # Message might be the same
        call("Sample 1/1: Prompt='a'"),
        call("Generated: a_gen"),
        call("Finished generating 1 samples in 5.00 seconds."),
    ]
    mock_log_info.assert_has_calls(expected_calls)
    assert mock_log_info.call_count == len(expected_calls)

# TODO: Test edge case log_samples=False (already implicitly tested above) 