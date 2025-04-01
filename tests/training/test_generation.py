"""
Tests for the TextGenerator wrapper class.
"""
import pytest
import torch
from torch import nn
from unittest.mock import MagicMock, ANY

# Use the wrapper class now
from craft.training.generation import TextGenerator 

# --- Fixtures ---

# Simple mock dataset/tokenizer stub
class MockDataset:
    def __init__(self):
        self.tokenizer = MagicMock()
        # Simulate tokenizer returning simple sequences
        self.tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        # Simulate decode returning a fixed string
        self.tokenizer.decode.return_value = " generated text"
        # Mimic pad/eos id attributes (can be None)
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.eos_token_id = 1
        self.char_to_idx = None # Ensure tokenizer is preferred
        
    def decode(self, ids, skip_special_tokens=False):
         # Simple mock decode - ignores ids and skip_special_tokens for simplicity
         return self.tokenizer.decode.return_value

@pytest.fixture
def mock_generator_wrapper_model():
    """ Mock model with a .generate() method """
    model = MagicMock(spec=nn.Module)
    # Simulate generate returning a sequence longer than the input
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]]) 
    # Add dummy config attributes used by TextGenerator
    model.config = MagicMock()
    model.config.pad_token_id = 0
    model.config.eos_token_id = 1
    return model

@pytest.fixture
def mock_dataset_for_generator():
    """ Provides an instance of the mock dataset """
    return MockDataset()

# --- Test TextGenerator --- #

def test_text_generator_init(mock_generator_wrapper_model, mock_dataset_for_generator):
    """ Test TextGenerator initialization """
    generator = TextGenerator(mock_generator_wrapper_model, torch.device("cpu"), {}, mock_dataset_for_generator)
    assert generator.model is mock_generator_wrapper_model
    assert generator.dataset is mock_dataset_for_generator

def test_text_generator_generate_text(mock_generator_wrapper_model, mock_dataset_for_generator):
    """ Test the main generate_text method of the wrapper """
    generator = TextGenerator(mock_generator_wrapper_model, torch.device("cpu"), {}, mock_dataset_for_generator)
    prompt = "start: "
    results = generator.generate_text(
        start_prompt=prompt,
        max_new_tokens=2, # Corresponds to the [4, 5] in mock generate output
        do_sample=False # Use greedy for predictability 
    )
    
    # Check generate was called
    mock_generator_wrapper_model.generate.assert_called_once()
    call_args = mock_generator_wrapper_model.generate.call_args[1]
    assert call_args["max_new_tokens"] == 2
    assert call_args["do_sample"] is False
    
    # Check decode was called 
    # mock_dataset_for_generator.tokenizer.decode.assert_called_once()
    # Note: decode is called within the test fixture's decode method wrapper

    # Check result (based on mocked decode)
    assert len(results) == 1
    assert results[0].strip() == "generated text" # Based on mock decode

# Mock Dataset object
class MockDataset:
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        # Use the actual token strings to get IDs
        self.pad_token_id = char_to_idx.get("<pad>", None)
        self.eos_token_id = char_to_idx.get("<eos>", None)
        # Mock the decode method
        self.decode = MagicMock(side_effect=self._mock_decode)

    def _mock_decode(self, ids, skip_special_tokens=True):
        chars = []
        pad_token = self.idx_to_char.get(self.pad_token_id) if self.pad_token_id is not None else None
        eos_token = self.idx_to_char.get(self.eos_token_id) if self.eos_token_id is not None else None

        for i in ids:
            token = self.idx_to_char.get(i, "<unk>") # Get the token string
            if skip_special_tokens:
                 # Check against the actual token strings
                 if pad_token and token == pad_token:
                     continue
                 if eos_token and token == eos_token:
                     continue
            chars.append(token) # Append the token string
        return "".join(chars)

# Mock Model with a generate method
class MockModelWithGenerate(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__() # Ensure superclass is initialized
        self.vocab_size = vocab_size
        # Mock the generate method
        self.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]])) # Return some dummy indices
        # Add dummy parameter for device placement
        self._dummy_param = torch.nn.Parameter(torch.empty(0))
        self.device = torch.device("cpu") # Assume cpu
        # Mock config if TextGenerator needs it
        self.config = type('obj', (object,), {
            'pad_token_id': None,
            'eos_token_id': None
        })()

    def to(self, device):
        self.device = device
        # print(f"MockModelWithGenerate moved to {device}") # Comment out print
        return self

    def eval(self):
        pass

    def train(self):
        pass

# Fixtures
@pytest.fixture
def generation_mock_vocab(): # Renamed from mock_vocab
    # Define special tokens as single units
    special_tokens = ["<pad>", "<eos>"]
    chars = "abcdefghij "
    vocab = special_tokens + list(chars)
    char_to_idx = {token: i for i, token in enumerate(vocab)}
    idx_to_char = {i: token for i, token in enumerate(vocab)}
    # Expected: {'<pad>': 0, '<eos>': 1, 'a': 2, 'b': 3, ...}
    return char_to_idx, idx_to_char

@pytest.fixture
def mock_dataset(generation_mock_vocab): # Use renamed fixture
    char_to_idx, idx_to_char = generation_mock_vocab
    return MockDataset(char_to_idx, idx_to_char)

@pytest.fixture
def mock_model(generation_mock_vocab): # Use renamed fixture
    char_to_idx, _ = generation_mock_vocab
    model = MockModelWithGenerate(vocab_size=len(char_to_idx))
    model.config.pad_token_id = char_to_idx.get("<pad>")
    model.config.eos_token_id = char_to_idx.get("<eos>")
    return model

@pytest.fixture
def mock_config(): # Basic config dict
    return {"some_config_param": 123}

# --- Tests ---
def test_text_generator_init(mock_model, mock_dataset, mock_config):
    """Test TextGenerator initialization."""
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset
    )
    assert generator.model == mock_model
    assert generator.dataset == mock_dataset
    assert generator.device == torch.device('cpu')
    assert generator.config == mock_config
    # Ensure mappings are accessed via the dataset object
    assert hasattr(generator.dataset, 'char_to_idx')
    assert hasattr(generator.dataset, 'idx_to_char')
    assert generator.dataset.char_to_idx is not None
    assert generator.dataset.idx_to_char is not None

def test_text_generator_generate_text(mock_model, mock_dataset, mock_config):
    """Test the generate_text method calls the model's generate method."""
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset
    )
    # Use the renamed fixture's vocab
    char_to_idx = mock_dataset.char_to_idx 
    idx_to_char = mock_dataset.idx_to_char

    seed = "a" # index 2 in the new vocab
    max_new = 5
    gen_kwargs = {"temperature": 0.8, "top_k": 50}

    # Mock the underlying model.generate return value
    # Seed 'a' (idx 2) -> generates indices [2, 3, 4, 5, 6, 7] (a, b, c, d, e)
    # Note: index 7 is 'f' in the new vocab
    mock_model.generate.return_value = torch.tensor([[2, 3, 4, 5, 6, 7]], device='cpu')

    result_texts = generator.generate_text(seed, max_new_tokens=max_new, **gen_kwargs)
    result_text = result_texts[0] # Get the first result

    # 1. Assert model.generate was called
    mock_model.generate.assert_called_once()
    call_args, call_kwargs = mock_model.generate.call_args

    # 2. Check arguments passed to model.generate (flexible check)
    expected_input_ids = torch.tensor([[char_to_idx[c] for c in seed]], device='cpu')
    assert torch.equal(call_kwargs['input_ids'], expected_input_ids)
    assert call_kwargs['max_new_tokens'] == max_new
    # Check other core kwargs passed through
    assert call_kwargs['temperature'] == 0.8
    assert call_kwargs['top_k'] == 50
    assert call_kwargs['do_sample'] is True
    # Check that the generator correctly identified and passed eos/pad tokens
    assert call_kwargs.get('eos_token_id') == generator.dataset.eos_token_id
    # Check pad_token_id was passed if it exists in the dataset
    if generator.dataset.pad_token_id is not None:
        assert call_kwargs.get('pad_token_id') == generator.dataset.pad_token_id
    else:
        assert 'pad_token_id' not in call_kwargs # Or assert it's None if that's the expected default

    # 3. Assert the result text is correctly decoded
    # Input was 'a', generated sequence is 'abcdef' -> decoded output (excluding prompt) is 'bcdef'
    # Note: indices [3, 4, 5, 6, 7] correspond to 'bcdef' in the new vocab
    expected_text = "bcdef"
    assert result_text == expected_text

    # 4. Assert dataset.decode was called correctly
    # Indices passed to decode should be the generated part: [3, 4, 5, 6, 7]
    expected_decode_ids = [3, 4, 5, 6, 7]
    mock_dataset.decode.assert_called_once_with(expected_decode_ids, skip_special_tokens=True)