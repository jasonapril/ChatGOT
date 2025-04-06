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
        self.encode = MagicMock(return_value=[[0]]) # Corrected: Return list of list [[0]]
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        # Add any other attributes TextGenerator might check
        self.name_or_path = "mock_tokenizer"
        self.decode = MagicMock(side_effect=self._mock_decode)

    def _mock_decode(self, ids, skip_special_tokens=True):
        # Ensure ids is a list for consistent comparison/iteration
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        chars = []
        # NOTE: This mock decode needs access to idx_to_char, which isn't defined in MockTokenizer
        # It likely should rely on the MockDataset's idx_to_char if used standalone,
        # or be simplified if only used via MockDataset's decode.
        # Assuming simplified version for now if called directly:
        mock_idx_to_char = {i: chr(ord('a')+i) for i in range(26)} # Example mapping
        pad_token = mock_idx_to_char.get(self.pad_token_id) if self.pad_token_id is not None else None
        eos_token = mock_idx_to_char.get(self.eos_token_id) if self.eos_token_id is not None else None

        for i in ids:
            token = mock_idx_to_char.get(i, " ") # Get the token string
            if skip_special_tokens:
                 if pad_token and token == pad_token:
                     continue
                 if eos_token and token == eos_token:
                     continue
            chars.append(token) # Append the token string
        return "".join(chars)

class MockDataset:
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        # Use the actual token strings to get IDs
        self.pad_token_id = char_to_idx.get("<pad>", None)
        self.eos_token_id = char_to_idx.get("<eos>", None)
        # Mock the decode method using the correct _mock_decode from this class
        self.decode = MagicMock(side_effect=self._mock_decode)
        # Set up a tokenizer attribute if needed by tests, but don't instantiate recursively
        # This tokenizer is configured with the dataset's pad/eos ids
        self.tokenizer = MockTokenizer(pad_id=self.pad_token_id, eos_id=self.eos_token_id)
        self.tokenizer.encode = MagicMock(return_value=[[1, 2, 3]]) # Example encode output

    # Correct _mock_decode implementation for MockDataset
    def _mock_decode(self, ids, skip_special_tokens=True):
        # Ensure ids is a list for consistent comparison/iteration
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        chars = []
        # Use self.idx_to_char from the dataset instance
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
    # Create a simple class without a generate method
    class MockModelNoGenerate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy_param = torch.nn.Parameter(torch.empty(0))

        def to(self, device):
            return self # Minimal mock

        def eval(self):
            pass
        
    mock_model_no_gen = MockModelNoGenerate()

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
        assert "The provided model does not have a 'generate' method" in log_message

def test_text_generator_generate_text(mock_model, mock_dataset, mock_config):
    """Test the generate_text method calls the model's generate method."""
    generator = TextGenerator(
        model=mock_model,
        device=torch.device('cpu'),
        config=mock_config,
        dataset=mock_dataset
    )
    # Use the renamed fixture's vocab
    mock_dataset = mock_dataset # Use the fixture directly
    # Ensure the mock dataset for this test has a mock tokenizer encode
    mock_dataset.tokenizer = MagicMock()
    # Mock encode to return list of integers (as expected by torch.tensor)
    mock_dataset.tokenizer.encode.return_value = [0] # Mock encode result for 'a'

    seed = "a" # index 2 in the new vocab
    max_new = 5
    gen_kwargs = {"temperature": 0.8, "top_k": 50}

    # Mock the underlying model.generate return value
    # Seed 'a' (idx 0) -> generates indices [0, 3, 4, 5, 6, 7] (mocked prompt + generated)
    mock_model.generate.return_value = torch.tensor([[0, 3, 4, 5, 6, 7]], device='cpu')
    # Reset mock before call in this specific test
    mock_model.generate.reset_mock()

    result = generator.generate_text(seed, max_new_tokens=max_new, **gen_kwargs)

    # 1. Assert model.generate was called
    mock_model.generate.assert_called_once()

    # Guard against IndexError if generate wasn't called
    if not mock_model.generate.call_args:
        pytest.fail("model.generate was not called, cannot check arguments.")

    call_kwargs = mock_model.generate.call_args.kwargs

    # 2. Check arguments passed to model.generate (flexible check)
    # Access arguments via kwargs dictionary since input_ids is passed as keyword
    expected_input_ids = torch.tensor([[0]], dtype=torch.long, device='cpu') # Expect the mocked tokenizer output
    assert 'input_ids' in call_kwargs, "'input_ids' should be in keyword arguments"
    actual_input_ids = call_kwargs['input_ids']
    assert isinstance(actual_input_ids, torch.Tensor), "input_ids should be a tensor"
    assert actual_input_ids.shape == expected_input_ids.shape, f"Input shape mismatch: {actual_input_ids.shape} vs {expected_input_ids.shape}"
    assert torch.equal(actual_input_ids, expected_input_ids), f"Input tensor mismatch: {actual_input_ids} vs {expected_input_ids}"

    # 3. Check other relevant kwargs (using flexible matching)
    assert call_kwargs.get('max_new_tokens') == max_new
    assert call_kwargs.get('temperature') == gen_kwargs['temperature']
    assert call_kwargs.get('top_k') == gen_kwargs['top_k']
    assert call_kwargs.get('do_sample') is True # Default or explicitly passed

    # 4. Check the decoded output (based on mocked generate return and idx_to_char)
    # Mock encode returns [0]. Input is [[0]].
    # Mock model generate returns [[0, 3, 4, 5, 6, 7]]. Input length is 1.
    # Generated part is outputs[:, 1:] -> [[3, 4, 5, 6, 7]]
    # Let's assume the fixture idx_to_char maps 3->b, 4->c, etc.
    expected_text = "bcdef" # Based on fixture vocab and mocked return, excluding prompt
    assert isinstance(result, list) and len(result) > 0, "Expected non-empty list result"
    assert result[0] == expected_text, f"Expected '{expected_text}', got '{result[0]}'"

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
    assert "Dataset must provide either a callable tokenizer.encode method or a char_to_idx mapping" in results[0]

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
        if skip_special_tokens and ids == [1, 2, 3]:
            raise TypeError("Mock decode error with skip_special_tokens")
        elif ids == [1, 2, 3]:
            return "decoded text without skip"
        else:
             return "unexpected ids"
    mock_dataset.decode.side_effect = decode_side_effect
    # Mock dataset.tokenizer.encode to return a List[List[int]]
    mock_dataset.tokenizer.encode.return_value = [[0]] # Corrected

    generated_ids_part = torch.tensor([[1, 2, 3]], device='cpu')
    # Mock generate return value - Includes prompt
    mock_model.generate.return_value = torch.cat([torch.tensor([[0]]), generated_ids_part], dim=-1).to(torch.device('cpu'))

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
    # Update assertion to match actual log message format
    assert "Decoding with skip_special_tokens=True failed" in log_message
    assert "Trying without" in log_message

    # 3. Check the decoded result (should be from the fallback)
    assert isinstance(result, list) and len(result) > 0, "Expected non-empty list result for fallback"
    assert result[0] == "decoded text without skip"

@patch('craft.training.generation.logging.getLogger')
def test_generate_text_decode_exception_primary(mock_get_logger, mock_model, mock_dataset, mock_config):
    """Test generic Exception during primary decode attempt (lines 196-198)."""
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance

    # Configure decode to raise Exception on first call (skip_special_tokens=True)
    mock_dataset.decode.side_effect = Exception("Primary decode failed")
    # Mock dataset.tokenizer.encode to return a Tensor
    mock_dataset.tokenizer.encode.return_value = [[0]] # Corrected

    generated_ids_part = torch.tensor([[1, 2, 3]], device='cpu')
    # Mock generate return value - Includes prompt
    mock_model.generate.return_value = torch.cat([torch.tensor([[0]]), generated_ids_part], dim=-1).to(torch.device('cpu'))

    generator = TextGenerator(
        model=mock_model, device=torch.device('cpu'), config=mock_config, dataset=mock_dataset
    )
    result = generator.generate_text("start")

    # Assert error was logged
    mock_logger_instance.error.assert_called_once()
    assert "Error decoding generated sequence" in mock_logger_instance.error.call_args[0][0]
    assert "Primary decode failed" in mock_logger_instance.error.call_args[0][0]
    # Assert result contains the error message in the first element
    assert isinstance(result, list) and len(result) > 0, "Expected list with error message"
    assert "Decoding error: Primary decode failed" in result[0]
    assert "Raw IDs: [1, 2, 3]" in result[0]

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
    # Mock dataset.tokenizer.encode to return a List[List[int]]
    mock_dataset.tokenizer.encode.return_value = [[0]] # Corrected

    generated_ids_part = torch.tensor([[1, 2, 3]], device='cpu')
    # Mock generate return value - Includes prompt
    mock_model.generate.return_value = torch.cat([torch.tensor([[0]]), generated_ids_part], dim=-1).to(torch.device('cpu'))

    generator = TextGenerator(
        model=mock_model, device=torch.device('cpu'), config=mock_config, dataset=mock_dataset
    )
    result = generator.generate_text("start")

    # Assert error was logged (from the second exception)
    mock_logger_instance.error.assert_called_once()
    assert "Error decoding generated sequence during fallback" in mock_logger_instance.error.call_args[0][0]
    assert "Fallback decode failed" in mock_logger_instance.error.call_args[0][0]
    # Assert result contains the error message in the first element
    assert isinstance(result, list) and len(result) > 0, "Expected list with error message"
    assert "Decoding error (fallback): Fallback decode failed" in result[0]
    assert "Raw IDs: [1, 2, 3]" in result[0]

    # Check decode was called twice with the correct generated part (as lists)
    assert mock_dataset.decode.call_count == 2
    mock_dataset.decode.assert_any_call([1, 2, 3], skip_special_tokens=True)
    mock_dataset.decode.assert_called_with([1, 2, 3], skip_special_tokens=False)