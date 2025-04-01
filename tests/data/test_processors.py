#!/usr/bin/env python
"""
Tests for src.craft.data.processors
"""
import pytest
import numpy as np
import os
import pickle
import json
from unittest.mock import patch, MagicMock

# Functions to test
from craft.data.processors import (
    split_data, prepare_text_data, prepare_data, 
    prepare_image_data, prepare_audio_data, prepare_json_data
)

# --- Fixtures ---

@pytest.fixture
def temp_text_file(tmp_path):
    """Create a temporary text file for testing."""
    d = tmp_path / "data_proc_test"
    d.mkdir()
    p = d / "input.txt"
    content = "abcdefghijklmnopqrstuvwxyz" * 2 # Simple repeatable content
    p.write_text(content, encoding='utf-8')
    return p

@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for testing."""
    d = tmp_path / "data_proc_json_test"
    d.mkdir()
    p = d / "input.json"
    content = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    p.write_text(json.dumps(content), encoding='utf-8')
    return p

@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Create a temporary JSONL file for testing."""
    d = tmp_path / "data_proc_jsonl_test"
    d.mkdir()
    p = d / "input.jsonl"
    content = [{"id": 1, "text": "line one"}, {"id": 2, "text": "line two"}]
    lines = [json.dumps(item) for item in content]
    p.write_text("\n".join(lines), encoding='utf-8')
    return p

# --- Tests for split_data --- 

def test_split_data_list():
    """Test split_data with a simple list."""
    data = list(range(100))
    train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    # Check for approximate content (shuffling happens)
    assert set(train + val + test) == set(data)
    # Check that splits are disjoint (after shuffling, elements should be unique)
    assert len(set(train).intersection(set(val))) == 0
    assert len(set(train).intersection(set(test))) == 0
    assert len(set(val).intersection(set(test))) == 0

def test_split_data_numpy():
    """Test split_data with a NumPy array."""
    data = np.arange(100)
    train, val, test = split_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    # Check for approximate content (shuffling happens)
    assert set(train).union(set(val)).union(set(test)) == set(data)
    # Check that splits are disjoint 
    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0

def test_split_data_ratios_dont_sum_to_one():
    """Test split_data raises error if ratios don't sum to 1."""
    data = list(range(10))
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_data(data, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

def test_split_data_reproducibility_with_seed():
    """Test that splitting is reproducible with the same seed."""
    data = list(range(50))
    train1, val1, test1 = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train2, val2, test2 = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2

def test_split_data_different_seed():
    """Test that splitting is different with a different seed."""
    data = list(range(50))
    train1, val1, test1 = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train2, val2, test2 = split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=123)
    
    # It's highly unlikely the splits will be identical with different seeds
    assert train1 != train2 
    assert val1 != val2
    assert test1 != test2

def test_split_data_edge_case_empty_input():
    """Test split_data with empty input list."""
    data = []
    train, val, test = split_data(data)
    assert train == []
    assert val == []
    assert test == []

def test_split_data_edge_case_small_input():
    """Test split_data with input smaller than number of splits."""
    data = [1, 2] # Only 2 items
    train, val, test = split_data(data, train_ratio=0.5, val_ratio=0.3, test_ratio=0.2)
    # Expect splits proportional to ratios, potentially empty
    assert len(train) == 1 # int(2 * 0.5) = 1
    assert len(val) == 0 # Corrected expectation
    assert len(test) == 1 # Corrected: test gets the remainder (2 - 1 - 0 = 1)
    assert set(train + val + test) == set(data) # Ensure all elements are accounted for
    
    # Let's re-run with ratios that give clearer results for size 2
    train, val, test = split_data(data, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 0
    assert set(train + val) == set(data)

# --- Tests for prepare_text_data --- 

def test_prepare_text_data_character(temp_text_file, tmp_path):
    """Test prepare_text_data with character-level tokenization."""
    input_file = str(temp_text_file)
    output_dir = str(tmp_path / "output_char")
    config = {
        'data': {
            'format': 'character',
            'split_ratios': [0.6, 0.2, 0.2] # 60/20/20 split
        }
    }
    
    output_paths = prepare_text_data(input_file, output_dir, config)
    
    # 1. Check if output files were created
    assert set(output_paths.keys()) == {'train', 'val', 'test'}
    train_path = output_paths['train']
    val_path = output_paths['val']
    test_path = output_paths['test']
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)
    
    # 2. Load and check content of one split (e.g., train)
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
        
    assert train_data['tokenizer_type'] == 'character'
    assert 'chars' in train_data
    assert 'char_to_idx' in train_data
    assert 'idx_to_char' in train_data
    assert 'token_ids' in train_data
    assert isinstance(train_data['token_ids'], np.ndarray)
    
    # 3. Check tokenization and split sizes
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    expected_total_len = len(original_text)
    expected_train_len = int(expected_total_len * 0.6)
    expected_val_len = int(expected_total_len * 0.2)
    # Test split calculation might differ slightly if total_len * ratio isn't exact int
    # Check the actual implementation if this fails - it uses int(), so truncation
    expected_test_len = expected_total_len - expected_train_len - expected_val_len

    assert len(train_data['token_ids']) == expected_train_len
    
    # Check if vocab seems correct (should include a-z)
    assert set('abcdefghijklmnopqrstuvwxyz').issubset(set(train_data['chars']))
    assert train_data['vocab_size'] == len(train_data['chars'])
    
    # Quick check of encoding/decoding one token
    first_char = original_text[0]
    first_token_id = train_data['char_to_idx'][first_char]
    assert train_data['token_ids'][0] == first_token_id
    assert train_data['idx_to_char'][first_token_id] == first_char
    
    # Check other splits sizes
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    assert len(val_data['token_ids']) == expected_val_len
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    assert len(test_data['token_ids']) == expected_test_len

@patch('craft.data.processors.AutoTokenizer') # Mock AutoTokenizer import
def test_prepare_text_data_pretrained(MockAutoTokenizer, temp_text_file, tmp_path):
    """Test prepare_text_data with a pretrained Hugging Face tokenizer."""
    # Setup mock tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.encode.return_value = list(range(52)) # Dummy token ids
    mock_tokenizer_instance.vocab_size = 1000 # Dummy vocab size
    MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer_instance
    
    input_file = str(temp_text_file)
    output_dir = str(tmp_path / "output_pretrained")
    tokenizer_name = "gpt2" # Example tokenizer name
    config = {
        'data': {
            'format': 'pretrained', 
            'tokenizer_name': tokenizer_name,
            'split_ratios': [0.7, 0.1, 0.2] 
        }
    }
    
    output_paths = prepare_text_data(input_file, output_dir, config)
    
    # 1. Check tokenizer loading was called
    MockAutoTokenizer.from_pretrained.assert_called_once_with(tokenizer_name)
    
    # 2. Check tokenizer encoding was called
    with open(input_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    mock_tokenizer_instance.encode.assert_called_once_with(original_text)
    
    # 3. Check output files and structure
    assert set(output_paths.keys()) == {'train', 'val', 'test'}
    train_path = output_paths['train']
    assert os.path.exists(train_path)
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    assert train_data['tokenizer_type'] == 'pretrained'
    assert train_data['tokenizer_name'] == tokenizer_name
    assert train_data['vocab_size'] == 1000
    assert 'token_ids' in train_data
    assert isinstance(train_data['token_ids'], np.ndarray)
    
    # 4. Check split sizes (based on mock token ids length 52)
    expected_total_len = 52
    expected_train_len = int(expected_total_len * 0.7) # 36
    expected_val_len = int(expected_total_len * 0.1)   # 5
    expected_test_len = expected_total_len - expected_train_len - expected_val_len # 11

    assert len(train_data['token_ids']) == expected_train_len
    
    # Check other splits
    with open(output_paths['val'], 'rb') as f:
        val_data = pickle.load(f)
    assert len(val_data['token_ids']) == expected_val_len
    with open(output_paths['test'], 'rb') as f:
        test_data = pickle.load(f)
    assert len(test_data['token_ids']) == expected_test_len

def test_prepare_text_data_invalid_format(temp_text_file, tmp_path):
    """Test prepare_text_data with an invalid format config."""
    input_file = str(temp_text_file)
    output_dir = str(tmp_path / "output_invalid")
    config = {'data': {'format': 'invalid_format'}}
    
    with pytest.raises(ValueError, match="Unsupported text format: invalid_format"):
        prepare_text_data(input_file, output_dir, config)

def test_prepare_text_data_missing_tokenizer_name(temp_text_file, tmp_path):
    """Test prepare_text_data raises error if tokenizer_name is missing for non-char format."""
    input_file = str(temp_text_file)
    output_dir = str(tmp_path / "output_missing_tok")
    config = {'data': {'format': 'pretrained'}} # Missing tokenizer_name
    
    with pytest.raises(ValueError, match="tokenizer_name must be provided"):
        prepare_text_data(input_file, output_dir, config)

def test_prepare_text_data_invalid_split_ratios(temp_text_file, tmp_path):
    """Test prepare_text_data raises error for invalid split_ratios."""
    input_file = str(temp_text_file)
    output_dir = str(tmp_path / "output_bad_split")
    
    bad_ratios = [
        [0.5, 0.5], # Too few elements
        [0.4, 0.4, 0.3], # Sum > 1
        [0.4, 0.4, 0.1]  # Sum < 1
    ]
    
    for ratios in bad_ratios:
        config = {'data': {'format': 'character', 'split_ratios': ratios}}
        with pytest.raises(ValueError, match="split_ratios must be a list of 3 numbers that sum to 1.0"):
            prepare_text_data(input_file, output_dir, config)

# --- Tests for placeholder functions ---

def test_prepare_image_data_raises():
    with pytest.raises(NotImplementedError):
        prepare_image_data("dummy.jpg", "dummy_dir")
        
def test_prepare_audio_data_raises():
    with pytest.raises(NotImplementedError):
        prepare_audio_data("dummy.wav", "dummy_dir")
        
# --- Tests for prepare_json_data ---
# NOTE: prepare_json_data currently only reads the file and returns None.
# We'll test that basic functionality for now.

# @patch('craft.data.processors.pickle') # Not needed yet
def test_prepare_json_data_basic(temp_json_file, tmp_path):
    """Test reading a basic JSON file runs without error."""
    input_file = str(temp_json_file)
    output_dir = str(tmp_path / "output_json")
    try:
        prepare_json_data(input_file, output_dir) 
    except Exception as e:
        pytest.fail(f"prepare_json_data raised unexpected exception: {e}")
    # TODO: Add more meaningful assertions once the function does more

# @patch('craft.data.processors.pickle') # Not needed yet
def test_prepare_jsonl_data_basic(temp_jsonl_file, tmp_path):
    """Test reading a basic JSONL file runs without error."""
    input_file = str(temp_jsonl_file)
    output_dir = str(tmp_path / "output_jsonl")
    try:
        prepare_json_data(input_file, output_dir)
    except Exception as e:
        pytest.fail(f"prepare_json_data raised unexpected exception: {e}")
    # TODO: Add more meaningful assertions once the function does more

# Since prepare_json_data doesn't do much yet, commenting out tests for now.
# We can uncomment and expand these when the function's logic is implemented.

# --- Tests for prepare_data (Dispatcher) ---

@patch('craft.data.processors.prepare_text_data')
def test_prepare_data_dispatch_text(mock_prepare_text, temp_text_file, tmp_path):
    """Test prepare_data correctly dispatches to prepare_text_data."""
    input_file = str(temp_text_file)
    output_dir = str(tmp_path / "output_dispatch_text")
    config = {'data': {'type': 'text'}} # Explicitly set type
    mock_prepare_text.return_value = {"train": "path/train.pkl"} # Mock return
    
    result = prepare_data(input_file, output_dir, config)
    
    mock_prepare_text.assert_called_once_with(input_file, output_dir, config)
    assert result == {"train": "path/train.pkl"}

@patch('craft.data.processors.prepare_text_data')
def test_prepare_data_infer_text(mock_prepare_text, temp_text_file, tmp_path):
    """Test prepare_data infers text type from extension."""
    input_file = str(temp_text_file) # Ends in .txt
    output_dir = str(tmp_path / "output_infer_text")
    config = {} # No type specified
    prepare_data(input_file, output_dir, config)
    mock_prepare_text.assert_called_once_with(input_file, output_dir, config)

@patch('craft.data.processors.prepare_json_data')
def test_prepare_data_dispatch_json(mock_prepare_json, temp_json_file, tmp_path):
    """Test prepare_data correctly dispatches to prepare_json_data."""
    input_file = str(temp_json_file)
    output_dir = str(tmp_path / "output_dispatch_json")
    config = {'data': {'type': 'json'}} 
    # prepare_json_data currently returns None, adjust if it changes
    mock_prepare_json.return_value = None 
    
    result = prepare_data(input_file, output_dir, config)
    
    mock_prepare_json.assert_called_once_with(input_file, output_dir, config)
    assert result is None 

@patch('craft.data.processors.prepare_json_data')
def test_prepare_data_infer_jsonl(mock_prepare_json, temp_jsonl_file, tmp_path):
    """Test prepare_data infers json type for .jsonl extension."""
    input_file = str(temp_jsonl_file) # Ends in .jsonl
    output_dir = str(tmp_path / "output_infer_jsonl")
    config = {} 
    prepare_data(input_file, output_dir, config)
    mock_prepare_json.assert_called_once_with(input_file, output_dir, config)

def test_prepare_data_unsupported_type(tmp_path):
    """Test prepare_data raises error for unsupported explicit type."""
    input_file = "dummy.xyz"
    output_dir = str(tmp_path / "output_unsupported")
    config = {'data': {'type': 'unsupported_format'}}
    with pytest.raises(ValueError, match="Unsupported data type: unsupported_format"):
        prepare_data(input_file, output_dir, config)

def test_prepare_data_cannot_infer_type(tmp_path):
    """Test prepare_data raises error when type cannot be inferred."""
    input_path_obj = tmp_path / "unknown.extension"
    input_path_obj.touch() # Create the dummy file using the Path object
    input_file = str(input_path_obj) # Now convert to string
    output_dir = str(tmp_path / "output_cannot_infer")
    config = {}
    with pytest.raises(ValueError, match="Cannot infer data type"):
        prepare_data(input_file, output_dir, config) 