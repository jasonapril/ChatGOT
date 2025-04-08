import pytest
import pickle
import json
from pathlib import Path
import tempfile
import numpy as np

# Import the function to test
from craft.data.char_processor import process_char_data
# Import the tokenizer class to load the saved tokenizer
from craft.data.tokenizers.char import CharTokenizer

# --- Test Fixtures ---

@pytest.fixture(scope="function") # Use function scope for independent test runs
def temp_data_dir():
    """Creates a temporary directory for test input/output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(scope="function")
def sample_input_file(temp_data_dir):
    """Creates a sample input text file in the temp directory."""
    input_path = temp_data_dir / "input.txt"
    # Simple text with repeating characters for frequency tests
    content = "abcabcabc\n123123\nAAA aaa BBB bbb CCC ccc\nxyz"
    input_path.write_text(content, encoding='utf-8')
    yield input_path

# --- Test Functions --- #

def test_processor_success_defaults(sample_input_file, temp_data_dir):
    """Test successful processing with default settings (using splits tuple)."""
    input_path = sample_input_file
    output_dir = temp_data_dir
    train_path = output_dir / "train.pkl"
    val_path = output_dir / "val.pkl"
    test_path = output_dir / "test.pkl" # Check for test split too
    tokenizer_dir = output_dir / "tokenizer"

    # Run the processor using positional arguments and default splits
    output_files = process_char_data(str(input_path), str(output_dir))

    assert isinstance(output_files, dict)
    assert "train" in output_files
    assert "val" in output_files
    assert "test" in output_files

    # Check if output files exist
    assert Path(output_files["train"]).exists()
    assert Path(output_files["val"]).exists()
    assert Path(output_files["test"]).exists()
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()
    assert tokenizer_dir.exists() # Check tokenizer directory exists
    assert (tokenizer_dir / "vocab.json").exists()
    assert (tokenizer_dir / "tokenizer_config.json").exists()

    # Basic check of pickle file content (metadata is no longer here)
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    assert isinstance(train_data, np.ndarray)
    assert train_data.dtype == np.uint16
    assert len(train_data) > 0

    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    assert isinstance(val_data, np.ndarray)
    assert val_data.dtype == np.uint16
    assert len(val_data) > 0

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    assert isinstance(test_data, np.ndarray)
    assert test_data.dtype == np.uint16
    assert len(test_data) > 0

    # Check tokenizer loading and basic properties
    loaded_tokenizer = CharTokenizer.load_from_dir(str(tokenizer_dir))
    assert isinstance(loaded_tokenizer, CharTokenizer)
    assert loaded_tokenizer.vocab_size > 0

def test_processor_tokenizer_metadata_content(sample_input_file, temp_data_dir):
    """Test the content of the metadata stored in the saved tokenizer."""
    input_path = sample_input_file
    output_dir = temp_data_dir
    tokenizer_dir = output_dir / "tokenizer"

    process_char_data(str(input_path), str(output_dir))

    assert tokenizer_dir.exists()
    # Load the tokenizer using the class method
    tokenizer = CharTokenizer.load_from_dir(str(tokenizer_dir))

    # Check expected characters (case-sensitive)
    expected_chars = set("abc123\n ABCxyz")
    assert all(c in tokenizer.char_to_idx for c in expected_chars)
    assert len(tokenizer.char_to_idx) == tokenizer.vocab_size
    assert len(tokenizer.idx_to_char) == tokenizer.vocab_size

    # Check mapping consistency
    for char, idx in tokenizer.char_to_idx.items():
        assert tokenizer.idx_to_char[idx] == char

    # Check <unk> token is NOT present by default
    assert tokenizer.unk_token is None
    assert tokenizer.unk_token_id is None
    assert "<unk>" not in tokenizer.char_to_idx

    # Verify some tokens map correctly using the loaded tokenizer
    # Load using the class method
    tokenizer = CharTokenizer.load_from_dir(str(tokenizer_dir))
    assert train_tokens[0] == tokenizer.char_to_idx['a']
    assert val_tokens[0] == tokenizer.char_to_idx[input_content[expected_train_len]]

def test_processor_data_split_and_content(sample_input_file, temp_data_dir):
    """Test token counts and split ratio based on the splits tuple."""
    input_path = sample_input_file
    output_dir = temp_data_dir
    train_path = output_dir / "train.pkl"
    val_path = output_dir / "val.pkl"
    test_path = output_dir / "test.pkl"
    tokenizer_dir = output_dir / "tokenizer"

    input_content = input_path.read_text(encoding='utf-8')
    total_chars = len(input_content)
    test_splits = (0.7, 0.2, 0.1) # Use custom splits

    process_char_data(str(input_path), str(output_dir), splits=test_splits)

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    # Load directly as numpy arrays
    train_tokens = train_data
    val_tokens = val_data
    test_tokens = test_data

    total_processed_tokens = len(train_tokens) + len(val_tokens) + len(test_tokens)

    assert total_processed_tokens == total_chars

    # Check approximate split ratios (allow some tolerance due to integer division)
    expected_train_len = int(total_chars * test_splits[0])
    expected_val_len = int(total_chars * test_splits[1])
    expected_test_len = total_chars - expected_train_len - expected_val_len

    assert abs(len(train_tokens) - expected_train_len) <= 1
    assert abs(len(val_tokens) - expected_val_len) <= 1
    # Test split might absorb rounding errors
    assert abs(len(test_tokens) - expected_test_len) <= 2

    # Verify some tokens map correctly using the loaded tokenizer
    tokenizer = CharTokenizer.load_from_dir(str(tokenizer_dir))
    assert train_tokens[0] == tokenizer.char_to_idx['a']
    assert val_tokens[0] == tokenizer.char_to_idx[input_content[expected_train_len]]

def test_processor_invalid_input_path(temp_data_dir):
    """Test error handling for non-existent input file."""
    with pytest.raises(ValueError, match="Input file not found"):
        process_char_data("non_existent_file.txt", str(temp_data_dir))

def test_processor_invalid_splits_tuple(sample_input_file, temp_data_dir):
    """Test error handling for invalid splits tuple values."""
    input_path = sample_input_file
    output_dir = temp_data_dir
    
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        process_char_data(str(input_path), str(output_dir), splits=(0.8, 0.1, 0.2)) # Sum > 1
        
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        process_char_data(str(input_path), str(output_dir), splits=(0.7, 0.1, 0.1)) # Sum < 1

# (Tests will be added here) 