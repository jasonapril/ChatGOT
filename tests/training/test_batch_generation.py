"""
Tests for batch text generation function.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock, call
import torch.nn.functional as F

# Import function to test
from craft.training.batch_generation import batch_generate

# Fixture for character mappings, including <pad> and <eos>
@pytest.fixture
def char_maps_batch():
    char_to_idx = {'<unk>': 0, 'a': 1, 'b': 2, 'c': 3, ' ': 4, '<pad>': 5, '<eos>': 6}
    idx_to_char = {0: '<unk>', 1: 'a', 2: 'b', 3: 'c', 4: ' ', 5: '<pad>', 6: '<eos>'}
    return char_to_idx, idx_to_char

@pytest.fixture
def mock_model_batch():
    model = MagicMock(spec=torch.nn.Module)
    # Basic model that returns fixed logits based on last token
    vocab_size = 7 # unk, a, b, c, space, pad, eos

    def model_side_effect(input_ids):
        batch_size, seq_len = input_ids.shape
        # Get last non-pad token for each item in batch
        # This is a simplified mock, real model would use attention mask
        last_tokens = []
        for i in range(batch_size):
            # Find last non-padding token index
            non_pad_indices = (input_ids[i] != 5).nonzero(as_tuple=True)[0]
            last_non_pad_idx = non_pad_indices[-1].item() if len(non_pad_indices) > 0 else -1
            last_token = input_ids[i, last_non_pad_idx].item() if last_non_pad_idx != -1 else -1 # Use -1 if all pads
            last_tokens.append(last_token)

        all_logits = torch.full((batch_size, vocab_size), -10.0, dtype=torch.float)

        for i, last_token_idx in enumerate(last_tokens):
            if last_token_idx == 1: # 'a'
                all_logits[i, 2] = 1.0 # -> 'b'
            elif last_token_idx == 2: # 'b'
                all_logits[i, 3] = 1.0 # -> 'c'
            elif last_token_idx == 3: # 'c'
                 all_logits[i, 6] = 1.0 # -> <eos>
            else: # Default or <eos>
                all_logits[i, 6] = 1.0 # -> <eos>

        # Return shape (batch_size, sequence_length_out=1, vocab_size)
        return all_logits.unsqueeze(1)

    model.side_effect = model_side_effect
    model.parameters.return_value = [torch.nn.Parameter(torch.tensor([1.0]))] # For device check
    return model

# --- Tests for batch_generate ---

def test_batch_generate_basic(mock_model_batch, char_maps_batch):
    """Test basic batch generation with different prompt lengths and EOS stopping."""
    char_to_idx, idx_to_char = char_maps_batch
    prompts = ["a", "ab"]
    max_length = 4 # Max new tokens
    device = torch.device('cpu')

    # Patch multinomial for deterministic output (argmax)
    with patch('torch.multinomial') as mock_multinomial:
        mock_multinomial.side_effect = lambda probs, num_samples: torch.argmax(probs, dim=1, keepdim=True)

        generated_texts = batch_generate(
            model=mock_model_batch,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            prompts=prompts,
            max_length=max_length,
            temperature=1.0, # For deterministic argmax
            device=device,
            top_p=0.0 # Disable top_p for basic test
        )

    # Expected generations based on mock_model_batch:
    # Prompt 1: "a" -> b -> c -> <eos> (stops)
    # Prompt 2: "ab" -> c -> <eos> (stops)
    expected_texts = ["abc<eos>", "abc<eos>"]
    assert generated_texts == expected_texts

    # --- Verify model calls and contexts ---
    # Step 1: Both active. Contexts padded to length 2. pad_id=5
    #  [[1, 5], [1, 2]] # a<pad>, ab
    # Step 2: Both active. Model gets contexts based on previous step.
    #  Input to model: [[1, 5], [1, 2]] -> Output logits -> Sample b, c
    #  Internal contexts after sampling: [[1, 2], [2, 3]] (shifted)
    # Step 3: Both active. Model gets contexts:
    #  [[1, 2], [2, 3]] -> Output logits -> Sample c, <eos>
    #  Internal contexts after sampling: [[2, 3], [3, 6]] (shifted)
    # Step 4: Prompt 2 finished. Model gets context for prompt 1:
    #  [[2, 3]] -> Output logits -> Sample <eos>
    #  Internal context: [[3, 6]]
    # Step 5: Prompt 1 finished. Loop terminates.

    assert mock_model_batch.call_count == 3 # Steps 1, 2, 3 (model called with active contexts)

    # Check contexts passed to model
    call1_context = mock_model_batch.call_args_list[0][0][0]
    expected_call1_context = torch.tensor([[1, 5], [1, 2]], device=device)
    assert torch.equal(call1_context, expected_call1_context)

    call2_context = mock_model_batch.call_args_list[1][0][0]
    # Contexts are shifted left in the implementation after each token generation
    # After step 1 (sampling b, c), internal contexts before model call 2:
    # Prompt 1: a -> b context shifted -> [?] need to trace implementation detail
    # Let's trace the context update: contexts[i] = torch.cat((contexts[i:i+1, 1:], new_token_tensor), dim=1)
    # Start: ctx = [[1, 5], [1, 2]]
    # Step 1 samples: [2], [3]
    # Update ctx[0]: cat( [[5]], [[2]] ) -> [[5, 2]] ? No, uses full slice: cat( ctx[0:1, 1:], [[2]] ) -> cat( [[5]], [[2]] ) -> [[5, 2]]
    # Update ctx[1]: cat( [[2]], [[3]] ) -> [[2, 3]]
    # Ctx before step 2 call: [[5, 2], [2, 3]]
    expected_call2_context = torch.tensor([[5, 2], [2, 3]], device=device)
    assert torch.equal(call2_context, expected_call2_context)

    # Step 2 samples: [3], [6]
    # Update ctx[0]: cat( [[2]], [[3]] ) -> [[2, 3]]
    # Update ctx[1]: No update as active[1] becomes False
    # Ctx before step 3 call: [[2, 3]] (only active index 0 passed)
    call3_context = mock_model_batch.call_args_list[2][0][0]
    expected_call3_context = torch.tensor([[2, 3]], device=device)
    assert torch.equal(call3_context, expected_call3_context)

def test_batch_generate_top_p(mock_model_batch, char_maps_batch):
    """Test top_p sampling is applied correctly per sequence in the batch."""
    char_to_idx, idx_to_char = char_maps_batch
    # Prompts designed to yield different logits at step 1
    prompts = ["a", "b"] # -> expects logits for b, c ; expects logits for c, <eos>
    max_length = 1 # Only need first step
    device = torch.device('cpu')
    top_p_value = 0.7 # Example value

    # Based on mock_model_batch:
    # Logits for 'a' (idx 1): [..., 1.0 (b), ...] -> Probs ~ [..., 0.73 (b), ...]
    # Logits for 'b' (idx 2): [..., 1.0 (c), ...] -> Probs ~ [..., 0.73 (c), ...]
    # For top_p = 0.7, only the top token (b for prompt 1, c for prompt 2) should be kept.
    expected_kept_indices_batch = [{2}, {3}]

    captured_final_logits_batch = None
    original_softmax = F.softmax
    def softmax_wrapper(*args, **kwargs):
        nonlocal captured_final_logits_batch
        captured_final_logits_batch = args[0].clone()
        return original_softmax(*args, **kwargs)

    # Patch softmax and multinomial
    with patch('torch.nn.functional.softmax', new=softmax_wrapper):
        with patch('torch.multinomial') as mock_multinomial:
            # Return the most likely token according to top_p logic
            mock_multinomial.return_value = torch.tensor([[2], [3]]) # b, c

            batch_generate(
                model=mock_model_batch,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                prompts=prompts,
                max_length=max_length,
                temperature=1.0,
                device=device,
                top_p=top_p_value
            )

    assert captured_final_logits_batch is not None, "Final logits not captured"
    assert captured_final_logits_batch.shape[0] == len(prompts)

    # Check kept indices for each item in the batch
    for i in range(len(prompts)):
        kept_logits_mask = ~torch.isneginf(captured_final_logits_batch[i])
        kept_indices = set(torch.where(kept_logits_mask)[0].tolist())
        assert kept_indices == expected_kept_indices_batch[i], f"Incorrect kept indices for prompt {i} ('{prompts[i]}')"

def test_batch_generate_context_truncation(mock_model_batch, char_maps_batch):
    """Test context truncation in batch generation."""
    char_to_idx, idx_to_char = char_maps_batch
    default_max_len = 1024
    prompts = ["a" * default_max_len, "b" * (default_max_len - 1)] # One at limit, one below
    max_length = 2 # Generate 2 more steps
    device = torch.device('cpu')

    # Ensure model doesn't have config for default length check
    if hasattr(mock_model_batch, 'config'):
        del mock_model_batch.config

    # Patch multinomial to return fixed tokens
    # Step 1: Prompt 0 ('a'*) -> 'b' (2); Prompt 1 ('b'*) -> 'c' (3)
    # Step 2: Prompt 0 ('a'*+'b') -> 'c' (3); Prompt 1 ('b'*+'c') -> '<eos>' (6)
    with patch('torch.multinomial') as mock_multinomial:
        mock_multinomial.side_effect = [
            torch.tensor([[2], [3]]), # Step 1 results for [a, b]
            torch.tensor([[3], [6]])  # Step 2 results for [b, c]
        ]

        batch_generate(
            model=mock_model_batch,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            prompts=prompts,
            max_length=max_length,
            temperature=1.0,
            device=device,
            top_p=0.0
        )

    # Verify model calls and context lengths
    # Step 1: Contexts padded to 1024. Both active.
    # Step 2: Prompt 0 context length = 1024 (no truncate yet), Prompt 1 context length = 1024. Both active.
    #         Model gets [[...a, b], [...b, c]] - shape (2, 1024)
    # Step 3: Prompt 0 context length = 1024 (truncation happened), Prompt 1 finished.
    #         Model gets [[...b, c]] - shape (1, 1024) -- THIS STEP DOESN'T HAPPEN due to max_length=2
    # assert mock_model_batch.call_count == 3 # Incorrect expectation
    assert mock_model_batch.call_count == max_length # Should be called max_length times

    # Check context shape passed to model in each call
    call1_context = mock_model_batch.call_args_list[0][0][0]
    assert call1_context.shape == (2, default_max_len)

    call2_context = mock_model_batch.call_args_list[1][0][0]
    assert call2_context.shape == (2, default_max_len)

    # Remove check for call 3 as max_length was 2
    # call3_context = mock_model_batch.call_args_list[2][0][0]
    # assert call3_context.shape == (1, default_max_len) # Only 1 active, but still default length

def test_batch_generate_temperature(mock_model_batch, char_maps_batch):
    """Test that temperature scaling is applied correctly in batch mode."""
    char_to_idx, idx_to_char = char_maps_batch
    prompts = ["a"] # Single prompt for simplicity
    max_length = 1
    device = torch.device('cpu')
    test_temp = 0.5 # Example temperature

    # Mock model returns logits for 'a' -> [..., 1.0(b), ...]
    expected_logits_before_temp = torch.full((1, 7), -10.0, dtype=torch.float)
    expected_logits_before_temp[0, 2] = 1.0

    captured_logits_after_temp = None
    original_softmax = F.softmax
    def softmax_wrapper(*args, **kwargs):
        nonlocal captured_logits_after_temp
        # Capture logits after temp scaling, before softmax
        captured_logits_after_temp = args[0].clone()
        return original_softmax(*args, **kwargs)

    with patch('torch.nn.functional.softmax', new=softmax_wrapper):
        with patch('torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([[2]]) # Return 'b'

            batch_generate(
                model=mock_model_batch,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                prompts=prompts,
                max_length=max_length,
                temperature=test_temp,
                device=device,
                top_p=0.0
            )

    assert captured_logits_after_temp is not None, "Logits after temp not captured"

    # Verify captured logits are the original logits divided by temperature
    expected_logits_after_temp = expected_logits_before_temp / test_temp
    assert torch.allclose(captured_logits_after_temp, expected_logits_after_temp)

# TODO: Add tests for padding interaction, unknown chars

# TODO: Add tests for batch_generate 