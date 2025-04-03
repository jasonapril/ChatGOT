"""
Tests for the TextGenerator wrapper class.
"""
import pytest
import torch
from torch import nn
from unittest.mock import MagicMock, ANY, patch
import torch.nn.functional as F

# Use the wrapper class now
from craft.training.generation import TextGenerator

# --- Fixtures --- #

# Mock Dataset object
class MockTokenizer:
    def __init__(self, pad_id=None, eos_id=None):
        self.encode = MagicMock(return_value=torch.tensor([[0]])) # Default mock encode
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        # Add any other attributes TextGenerator might check
        self.name_or_path = "mock_tokenizer"

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

        # Add mock tokenizer instance
        self.tokenizer = MockTokenizer(pad_id=self.pad_token_id, eos_id=self.eos_token_id)

    def _mock_decode(self, ids, skip_special_tokens=True):
        # Ensure ids is a list for consistent comparison/iteration
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        chars = []
        pad_token = self.idx_to_char.get(self.pad_token_id) if self.pad_token_id is not None else None
        eos_token = self.idx_to_char.get(self.eos_token_id) if self.eos_token_id is not None else None

        for i in ids:
            token = self.idx_to_char.get(i, " ") # Get the token string
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

def test_text_generator_init_no_generate_warning(mock_dataset, mock_config):
    """Test warning log during __init__ if model lacks generate (line 37)."""
    # Create a mock model *without* generate
    mock_model_no_gen = MagicMock(spec=torch.nn.Module) # No generate method spec
    # Need to remove generate if MagicMock adds it by default somehow
    if hasattr(mock_model_no_gen, 'generate'):
        del mock_model_no_gen.generate

    # Patch the logger used by TextGenerator
    with patch('craft.training.generation.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        generator = TextGenerator(
            model=mock_model_no_gen,
            device=torch.device('cpu'),
            config=mock_config,
            dataset=mock_dataset
        )
        # Assert warning was logged
        mock_logger_instance.warning.assert_called_once()
        log_message = mock_logger_instance.warning.call_args[0][0]
        assert "does not have a standard `.generate()` method" in log_message

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
    expected_input_ids = torch.tensor([[0]], device='cpu') # Expect the mocked tokenizer output
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

# --- Tests for Encoding and Token ID Handling ---

@pytest.fixture
def mock_dataset_char_only(generation_mock_vocab):
    """Creates a MockDataset with only char_to_idx/idx_to_char (no tokenizer attr)."""
    char_to_idx, idx_to_char = generation_mock_vocab
    dataset = MockDataset(char_to_idx, idx_to_char)
    # Crucially, remove the tokenizer attribute if MockDataset adds one by default
    if hasattr(dataset, 'tokenizer'):
        delattr(dataset, 'tokenizer')
    return dataset

def test_generate_text_char_encoding_fallback(mock_model, mock_dataset_char_only, mock_config):
    """Test generate_text uses char_to_idx encoding when tokenizer is absent."""
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset_char_only # Use char-only dataset
    )
    char_to_idx = mock_dataset_char_only.char_to_idx
    seed = "ab" # indices 2, 3
    max_new = 3
    # Mock generate to return something based on char indices
    # Input [2, 3] -> generates [2, 3, 4, 5, 6] ('abcde')
    mock_model.generate.return_value = torch.tensor([[2, 3, 4, 5, 6]], device='cpu')

    results = generator.generate_text(seed, max_new_tokens=max_new)

    # Assert model.generate was called with char-based input_ids
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs
    expected_input_ids = torch.tensor([[char_to_idx['a'], char_to_idx['b']]], device='cpu')
    assert torch.equal(call_kwargs['input_ids'], expected_input_ids)

    # Assert result is decoded correctly (output 'cde')
    assert results[0] == "cde"
    # Assert dataset.decode was called with the right indices
    mock_dataset_char_only.decode.assert_called_once_with([4, 5, 6], skip_special_tokens=True)

def test_generate_text_no_encoding(mock_model, mock_config):
    """Test ValueError is raised if dataset has neither tokenizer nor char_to_idx."""
    # Create a dummy dataset object with no relevant attributes
    dummy_dataset = MagicMock(spec=[])
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=dummy_dataset
    )
    # Check that the method returns the error string list, not raises ValueError
    results = generator.generate_text("test")
    assert isinstance(results, list)
    assert len(results) == 1 # Default num_return_sequences
    assert "[Generation Error" in results[0]
    assert "Dataset must provide either a tokenizer" in results[0]

@pytest.mark.parametrize(
    "model_pad, model_eos, tok_pad, tok_eos, char_eos, expected_pad, expected_eos",
    [
        # Priority: Explicit > Model Config > Tokenizer > Char Fallback (EOS only)
        (50000, 50001, 0, 1, 2, 0, 1), # Tokenizer values used when model has different
        (None, None, 0, 1, 2, 0, 1),    # Tokenizer values used when model has None
        (50000, 50001, None, None, 2, 50000, 50001), # Model values used when tokenizer has None
        (None, None, None, None, 2, None, 2), # Char EOS fallback used when others are None
        (0, 1, None, None, None, 0, 1), # Model values used when tokenizer and char EOS are None
        (None, None, 0, None, 2, 0, 2), # Mix: Tokenizer PAD, Char EOS
        (50000, None, 0, 1, 2, 0, 1), # Mix: Tokenizer PAD/EOS override model PAD/None
        (None, 50001, 0, None, 2, 0, 50001), # Mix: Tokenizer PAD, Model EOS override char
    ]
)
@patch('craft.training.generation.logging.getLogger') # Mock logger to check debug messages
def test_generate_text_special_token_inference(
    mock_getLogger, model_pad, model_eos, tok_pad, tok_eos, char_eos, expected_pad, expected_eos,
    mock_model, mock_config, generation_mock_vocab
):
    """Test inference of pad_token_id and eos_token_id from different sources."""
    # Setup mock dataset and model based on parameters
    base_char_to_idx, base_idx_to_char = generation_mock_vocab
    
    # --- Create modified vocab based on char_eos for this specific test run ---
    current_char_to_idx = base_char_to_idx.copy()
    current_idx_to_char = base_idx_to_char.copy()
    if char_eos is not None:
        # Remove old EOS if it exists and conflicts
        if '<eos>' in current_char_to_idx and current_char_to_idx['<eos>'] != char_eos:
            old_eos_idx = current_char_to_idx['<eos>']
            if old_eos_idx in current_idx_to_char: # Ensure it exists before deleting
                 del current_idx_to_char[old_eos_idx]
        # Add the new/specified EOS
        current_char_to_idx['<eos>'] = char_eos
        current_idx_to_char[char_eos] = '<eos>'
    elif '<eos>' in current_char_to_idx: # char_eos is None, so remove existing EOS
         old_eos_idx = current_char_to_idx.pop('<eos>')
         if old_eos_idx in current_idx_to_char:
             del current_idx_to_char[old_eos_idx]
    # --- End vocab modification ---

    # Create dataset with potentially modified vocab
    dataset = MockDataset(current_char_to_idx, current_idx_to_char)
    # Add mock tokenizer attribute AFTER dataset init
    dataset.tokenizer = MockTokenizer(pad_id=tok_pad, eos_id=tok_eos)

    # Set model config values
    mock_model.config.pad_token_id = model_pad
    mock_model.config.eos_token_id = model_eos

    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=dataset
    )

    # Mock generate return value and call generate_text
    # Need to reset the generate mock for each parameterized run
    mock_model.generate.reset_mock()
    mock_model.generate.return_value = torch.tensor([[2, 3, 4]], device='cpu') # 'abc'
    generator.generate_text("a", max_new_tokens=2) 

    # Assert that model.generate was called with the correctly inferred token IDs
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs

    assert call_kwargs.get('pad_token_id') == expected_pad
    assert call_kwargs.get('eos_token_id') == expected_eos

    # Optional: Check logger debug messages for token IDs
    mock_logger_instance = mock_getLogger.return_value
    mock_logger_instance.debug.assert_any_call(f"Effective PAD token ID: {expected_pad}")
    mock_logger_instance.debug.assert_any_call(f"Effective EOS token ID: {expected_eos}")

def test_generate_text_greedy(mock_model, mock_dataset, mock_config):
    """Test generate_text with do_sample=False (greedy) (lines 154-158)."""
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset
    )
    # Mock generate return value
    mock_model.generate.return_value = torch.tensor([[2, 3, 4]], device='cpu') # 'abc'

    generator.generate_text("a", max_new_tokens=2, do_sample=False, temperature=0.1, top_k=1, top_p=0.5)

    # Assert generate called with do_sample=False and sampling params removed
    mock_model.generate.assert_called_once()
    call_kwargs = mock_model.generate.call_args.kwargs
    assert call_kwargs.get('do_sample') is False
    assert 'temperature' not in call_kwargs
    assert 'top_k' not in call_kwargs
    assert 'top_p' not in call_kwargs

def test_generate_text_raises_not_implemented(mock_dataset, mock_config):
    """Test generate_text raises NotImplementedError if model lacks generate (line 81)."""
    # Create a mock model *without* generate
    mock_model_no_gen = MagicMock(spec=torch.nn.Module) # No generate method spec
    if hasattr(mock_model_no_gen, 'generate'):
        del mock_model_no_gen.generate

    generator = TextGenerator(
        model=mock_model_no_gen,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset
    )

    # Assert calling generate_text raises the error
    # Adjust regex to match the actual error: "Model {type_name} does not have a .generate() method."
    with pytest.raises(NotImplementedError, match=r"Model \w+ does not have a \.generate\(\) method\."):
        generator.generate_text("start")

# Define a spec for the mock dataset without decode
class MockDatasetSpecNoDecode:
    def encode(self, *args, **kwargs): pass
    vocab_size = 0 # Or some default
    # Add char_to_idx to satisfy the initial encoding check in generate_text
    char_to_idx = {}

def test_generate_text_no_dataset_decode(mock_model, mock_config):
    """Test generate_text when dataset lacks a decode method (lines 171-172)."""
    # Create a mock dataset *without* decode using spec_set
    mock_dataset_no_decode = MagicMock(spec_set=MockDatasetSpecNoDecode)
    # Configure the required attributes/methods
    mock_dataset_no_decode.encode.return_value = torch.tensor([[1]], device='cpu')
    mock_dataset_no_decode.vocab_size = 10 # Set required attribute
    # char_to_idx is part of the spec now

    # Patch the logger *before* creating the TextGenerator instance
    with patch('craft.training.generation.logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance

        # Instantiate TextGenerator *inside* the patch context
        generator = TextGenerator(
            model=mock_model,
            device=torch.device('cpu'),
            config=mock_config,
            dataset=mock_dataset_no_decode
        )
        # Mock generate return value (needed for the call to succeed up to the decode check)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]], device='cpu')

        result = generator.generate_text("start")

        # Assert error was logged
        mock_logger_instance.error.assert_called_once()
        log_message = mock_logger_instance.error.call_args[0][0]
        assert "Dataset does not have a required 'decode' method" in log_message

        # Assert empty list is returned
        assert result == ["Error: Decoding not possible."]

@patch('craft.training.generation.logging.getLogger')
def test_generate_text_decode_type_error_fallback(mock_get_logger, mock_model, mock_dataset, mock_config):
    """Test decode fallback on TypeError (lines 181-182)."""
    # Configure the mock logger instance that getLogger will return
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    # Configure mock decode to raise TypeError only when skip_special_tokens is True
    def decode_side_effect(ids, skip_special_tokens=False):
        # Use tolist() for comparison as tensors might be on different devices or have grad
        if skip_special_tokens and ids == [1, 2, 3]:
            raise TypeError("Mock decode error with skip_special_tokens")
        elif ids == [1, 2, 3]:
            return "decoded text without skip"
        else:
             return "unexpected ids"
    mock_dataset.decode.side_effect = decode_side_effect
    # Mock dataset.tokenizer.encode to control input_ids
    input_ids_cpu = torch.tensor([[0]]) # Dummy prompt tensor
    mock_dataset.tokenizer.encode.return_value = input_ids_cpu

    # Mock generate return value - Includes prompt
    generated_ids_part = torch.tensor([[1, 2, 3]], device='cpu')
    # Note: model.generate output should be on the same device as input_ids
    # If TextGenerator moves input_ids to device, ensure generate returns tensor on that device
    mock_model.generate.return_value = torch.cat([input_ids_cpu, generated_ids_part], dim=-1).to(torch.device('cpu'))

    # Instantiate TextGenerator (uses default input_processor which calls mock_dataset.encode)
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset
    )

    result = generator.generate_text("start") # prompt_text content doesn't matter due to encode mock

    # Assert warning was logged about the fallback
    mock_logger_instance.warning.assert_called_once()
    log_message = mock_logger_instance.warning.call_args[0][0]
    assert "Decoding failed with skip_special_tokens=True" in log_message

    # Assert the fallback result is returned
    assert result == ["decoded text without skip"]

    # Check that decode was called twice (initial attempt + fallback)
    assert mock_dataset.decode.call_count == 2
    mock_dataset.decode.assert_any_call([1, 2, 3], skip_special_tokens=True)
    mock_dataset.decode.assert_called_with([1, 2, 3], skip_special_tokens=False)

@patch('craft.training.generation.logging.getLogger')
def test_generate_text_decode_exception_primary(mock_get_logger, mock_model, mock_dataset, mock_config):
    """Test generic Exception during primary decode attempt (lines 196-198)."""
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    # Configure decode to raise Exception on first call (skip_special_tokens=True)
    mock_dataset.decode.side_effect = Exception("Primary decode failed")
    # Mock dataset.tokenizer.encode
    input_ids_cpu = torch.tensor([[0]]) # Dummy prompt tensor
    mock_dataset.tokenizer.encode.return_value = input_ids_cpu

    generated_ids_part = torch.tensor([[1, 2, 3]], device='cpu')
    # Mock generate return value - Includes prompt
    mock_model.generate.return_value = torch.cat([input_ids_cpu, generated_ids_part], dim=-1).to(torch.device('cpu'))

    generator = TextGenerator(
        model=mock_model, device=torch.device('cpu'), config=mock_config, dataset=mock_dataset
    )
    result = generator.generate_text("start")

    # Assert error was logged
    mock_logger_instance.error.assert_called_once()
    log_message = mock_logger_instance.error.call_args[0][0]
    assert "Error decoding generated sequence" in log_message

    # Assert error string is returned
    assert result == ["[Decoding Error]"]
    # Check decode was called only once with the correct generated part (as a list)
    mock_dataset.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)

@patch('craft.training.generation.logging.getLogger')
def test_generate_text_decode_exception_fallback(mock_get_logger, mock_model, mock_dataset, mock_config):
    """Test generic Exception during fallback decode attempt (lines 193-195)."""
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    # Configure decode: TypeError on first call, Exception on second
    def decode_side_effect(ids, skip_special_tokens=False):
        if skip_special_tokens and ids == [1, 2, 3]: # Compare with list
            raise TypeError("Skip tokens error")
        elif not skip_special_tokens and ids == [1, 2, 3]:
            raise Exception("Fallback decode failed")
        else:
            return "unexpected ids"
    mock_dataset.decode.side_effect = decode_side_effect
    # Mock dataset.tokenizer.encode
    input_ids_cpu = torch.tensor([[0]]) # Dummy prompt tensor
    mock_dataset.tokenizer.encode.return_value = input_ids_cpu

    generated_ids_part = torch.tensor([[1, 2, 3]], device='cpu')
    # Mock generate return value - Includes prompt
    mock_model.generate.return_value = torch.cat([input_ids_cpu, generated_ids_part], dim=-1).to(torch.device('cpu'))

    generator = TextGenerator(
        model=mock_model, device=torch.device('cpu'), config=mock_config, dataset=mock_dataset
    )
    result = generator.generate_text("start")

    # Assert error was logged (from the second exception)
    mock_logger_instance.error.assert_called_once()
    log_message = mock_logger_instance.error.call_args[0][0]
    assert "Error during fallback decoding attempt" in log_message

    # Assert error string is returned
    assert result == ["[Decoding Error]"]

    # Check decode was called twice with the correct generated part (as lists)
    assert mock_dataset.decode.call_count == 2
    mock_dataset.decode.assert_any_call([1, 2, 3], skip_special_tokens=True)
    mock_dataset.decode.assert_called_with([1, 2, 3], skip_special_tokens=False)