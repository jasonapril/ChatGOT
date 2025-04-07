"""
Unit tests for the .generate() method of GenerativeModel.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from craft.config.schemas import GenerativeModelConfig
from craft.models.base import GenerativeModel
from .conftest import MockGenerativeModel

class TestGenerateMethod:
    """Tests the generate method implemented in GenerativeModel base class (pytest style)."""
    
    @pytest.fixture
    def gen_config(self):
        """Provides a standard GenerativeModelConfig for generate tests."""
        # Need to ensure vocab_size and d_model are set as MockGenerativeModel uses them
        config = GenerativeModelConfig(
            architecture="mock_generative_arch", # Add required discriminator
            max_seq_length=20
        )
        # Add required fields dynamically if not in base config schema
        config.vocab_size=10 
        config.d_model=8
        return config
    
    @pytest.fixture
    def generative_model(self, gen_config):
        """Provides a MockGenerativeModel instance with mocked forward."""
        mock_generative_config_dict = {
            "vocab_size": gen_config.vocab_size,
            "d_model": gen_config.d_model,
            "max_seq_length": gen_config.max_seq_length
        }
        model = MockGenerativeModel(
            vocab_size=gen_config.vocab_size, 
            d_model=gen_config.d_model,
            max_seq_length=gen_config.max_seq_length
        )
        # Mock the forward pass to return predictable logits
        # The mock needs to accept 'x' and return a tensor of shape (batch, seq, vocab)
        model.forward = MagicMock(name="mock_forward") 
        return model

    def _setup_forward_mock(self, model, return_logits):
        """Helper to configure the mocked forward pass output."""
        # Ensure the mock accepts 'x'
        def side_effect_func(x):
            batch_size = x.shape[0]
            # Use shape from provided logits if it has seq dim, else assume next token
            seq_len = return_logits.shape[1] if return_logits.ndim == 3 else 1 
            vocab_size = return_logits.shape[-1]
            # Return the predefined logits, expanded for batch if needed
            if batch_size > 1 and return_logits.shape[0] == 1:
                 # Ensure correct shape (batch, seq, vocab)
                 if return_logits.ndim == 3:
                     return return_logits.expand(batch_size, -1, -1).to(x.device)
                 else: # Handle case where return_logits is (1, vocab)
                     return return_logits.unsqueeze(1).expand(batch_size, seq_len, -1).to(x.device)
            # Ensure correct shape even for batch_size=1
            elif return_logits.ndim == 2: # (vocab,) or (1, vocab)
                 return return_logits.reshape(1, 1, vocab_size).expand(batch_size, seq_len, -1).to(x.device)
            return return_logits.to(x.device) # Assumes (batch, seq, vocab)
        model.forward.side_effect = side_effect_func

    def test_greedy_decoding(self, generative_model):
        """Test generate with greedy decoding (default)."""
        model = generative_model
        prompt = torch.tensor([[1, 2]]) # Batch size 1, seq len 2
        max_new = 3
        
        # Shape needs to match expected output of forward: (batch, seq, vocab)
        # For generate, only the *last* token logit matters for next token prediction.
        # Mock should return (batch_size, 1, vocab_size)
        next_logits = torch.zeros(1, 1, model.config.vocab_size)
        next_logits[0, 0, 5] = 1.0 
        
        self._setup_forward_mock(model, next_logits)
        
        generated_ids = model.generate(prompt, max_new_tokens=max_new, temperature=0.0) # Greedy
        
        expected_ids = torch.tensor([[1, 2, 5, 5, 5]])
        assert torch.equal(generated_ids, expected_ids)
        # Check forward was called correct number of times
        assert model.forward.call_count == max_new

    def test_temperature_sampling(self, generative_model):
        """Test generate with temperature sampling."""
        model = generative_model
        prompt = torch.tensor([[1]]) # Batch size 1
        max_new = 5
        
        # Fixed logits favoring token '3' but others possible
        # Make probabilities less skewed for temperature to have an effect
        logits = torch.zeros(1, 1, model.config.vocab_size)
        # logits[0, 0, 3] = 10.0 
        # logits[0, 0, 4] = 5.0
        # logits[0, 0, 5] = 1.0
        logits[0, 0, 3] = 5.0 # More reasonable logits
        logits[0, 0, 4] = 4.0
        logits[0, 0, 5] = 3.0
        self._setup_forward_mock(model, logits)
        
        torch.manual_seed(123) # For reproducibility
        generated_ids_temp = model.generate(prompt, max_new_tokens=max_new, temperature=0.8, top_k=0)
        
        torch.manual_seed(123) # Reset seed
        generated_ids_greedy = model.generate(prompt, max_new_tokens=max_new, temperature=0.0, top_k=0)
        
        assert generated_ids_temp.shape[1] == prompt.shape[1] + max_new
        # With temperature, output might differ from greedy
        # Check that *some* non-greedy tokens were likely picked
        assert not torch.equal(generated_ids_temp[:, prompt.shape[1]:],
                               generated_ids_greedy[:, prompt.shape[1]:])
        # Restore correct assertion: called once per token for each of the two generate calls
        assert model.forward.call_count == max_new * 2 

    def test_top_p_sampling(self, generative_model):
        """Test generate with top-p (nucleus) sampling."""
        model = generative_model
        prompt = torch.tensor([[1]])
        max_new = 5
        top_p = 0.5 # Choose p such that only token '3' is likely selected
        
        logits = torch.zeros(1, 1, model.config.vocab_size)
        logits[0, 0, 3] = 5.0 # High probability
        logits[0, 0, 4] = 2.0 # Lower
        logits[0, 0, 5] = 0.0 # Lowest
        self._setup_forward_mock(model, logits)

        generated_ids = model.generate(prompt, max_new_tokens=max_new, top_p=top_p, top_k=0)
        
        # With top_p=0.5, only token '3' should be sampled
        expected_suffix = torch.full((1, max_new), 3)
        assert torch.equal(generated_ids[:, -max_new:], expected_suffix)
        assert model.forward.call_count == max_new

    def test_max_new_tokens(self, generative_model):
        """Test generate stops after max_new_tokens."""
        model = generative_model
        prompt = torch.tensor([[1, 2, 3]])
        max_new = 7
        
        # Simple logits to avoid EOS or other stopping
        logits = torch.zeros(1, 1, model.config.vocab_size)
        logits[0, 0, 5] = 1.0
        self._setup_forward_mock(model, logits)

        generated_ids = model.generate(prompt, max_new_tokens=max_new)
        
        assert generated_ids.shape[1] == prompt.shape[1] + max_new
        assert model.forward.call_count == max_new

    def test_eos_token_stopping(self, generative_model):
        """Test generate stops when EOS token is generated."""
        model = generative_model
        eos_token_id = 0 # Assume EOS is token 0
        prompt = torch.tensor([[1, 2]])
        max_new = 10
        
        # Logits that will produce EOS (token 0) on the 3rd generation step
        logits_step1_2 = torch.zeros(1, 1, model.config.vocab_size); logits_step1_2[0, 0, 5] = 1.0
        # logits_step2 = torch.zeros(1, 1, model.config.vocab_size); logits_step2[0, 0, 6] = 1.0 # Simplify
        logits_step3 = torch.zeros(1, 1, model.config.vocab_size); logits_step3[0, 0, eos_token_id] = 1.0
        
        # Mock forward to return logits based on step: 5, 5, 0 (EOS)
        call_count = 0
        def side_effect_eos(x):
            nonlocal call_count
            call_count += 1
            batch_size = x.shape[0]
            if call_count <= 2: # Return token 5 for first two steps
                 return logits_step1_2.expand(batch_size, -1, -1).to(x.device)
            # if call_count == 2: return logits_step2.expand(batch_size, -1, -1).to(x.device)
            return logits_step3.expand(batch_size, -1, -1).to(x.device) # Return EOS logits
        model.forward.side_effect = side_effect_eos

        generated_ids = model.generate(
            prompt, 
            max_new_tokens=max_new, 
            eos_token_id=eos_token_id,
            temperature=0.0 # Explicitly enforce greedy
        )
        
        # expected_ids = torch.tensor([[1, 2, 5, 6, eos_token_id]]) # Original
        expected_ids = torch.tensor([[1, 2, 5, 5, eos_token_id]]) # Simplified expectation
        assert torch.equal(generated_ids, expected_ids)
        assert generated_ids.shape[1] < prompt.shape[1] + max_new
        assert model.forward.call_count == 3 # Stops after 3 calls

    def test_sequence_truncation(self, generative_model):
        """Test that input prompt is truncated if longer than max_seq_length - max_new."""
        model = generative_model
        max_len = model.config.max_seq_length # 20
        max_new = 5
        allowed_prompt_len = max_len - max_new # 15
        
        long_prompt = torch.arange(0, allowed_prompt_len + 3).unsqueeze(0) # shape (1, 18)
        expected_prompt = long_prompt[:, -allowed_prompt_len:] # shape (1, 15)
        
        # Set up mock to check the input to forward
        received_inputs = []
        def side_effect_capture(x):
            received_inputs.append(x.clone())
            # Return simple logits to allow generation
            logits = torch.zeros(x.shape[0], 1, model.config.vocab_size, device=x.device)
            logits[:, 0, 1] = 1.0
            return logits
        model.forward.side_effect = side_effect_capture

        generated_ids = model.generate(long_prompt, max_new_tokens=max_new)
        
        assert received_inputs # Ensure forward was called
        # Check that the *first* input passed to forward was the truncated prompt
        assert torch.equal(received_inputs[0], expected_prompt)
        assert generated_ids.shape[1] <= max_len # Can be shorter if EOS is hit 