"""
Tests for the CharTokenizer.
"""
import pytest
import os
import json
import pickle
from pathlib import Path

from craft.data.tokenizers.char import CharTokenizer

# --- Fixtures ---

@pytest.fixture
def sample_text_content():
    """Provides simple sample text content."""
    return "hello world\nthis is a test."

@pytest.fixture
def temp_text_file_for_tokenizer(tmp_path, sample_text_content):
    """Creates a temporary text file for tokenizer training."""
    p = tmp_path / "train_text.txt"
    p.write_text(sample_text_content, encoding='utf-8')
    return str(p)

@pytest.fixture
def char_tokenizer_empty():
    """Provides an empty CharTokenizer instance."""
    return CharTokenizer()

# --- Tests ---

def test_char_tokenizer_init_empty(char_tokenizer_empty):
    """Test initialization with default arguments."""
    assert char_tokenizer_empty.char_to_idx == {}
    assert char_tokenizer_empty.idx_to_char == {}
    assert char_tokenizer_empty.vocab_size == 0
    assert char_tokenizer_empty.config == {'model_type': 'char'}
    # Check base class defaults are inherited
    assert char_tokenizer_empty.unk_token == "<unk>"
    assert char_tokenizer_empty.eos_token == "</s>"
    # Check IDs - they might be None until vocab is built
    assert char_tokenizer_empty.unk_id is None
    assert char_tokenizer_empty.pad_id is None
    assert char_tokenizer_empty.bos_id is None
    assert char_tokenizer_empty.eos_id is None

def test_char_tokenizer_init_with_config():
    """Test initialization with a provided config."""
    # Use lowercase unk for consistency
    config = {"model_type": "char", "special_tokens": {"unk": "<unk>"}}
    # Instantiate directly, passing unk_token
    tokenizer = CharTokenizer(unk_token="<unk>")
    assert tokenizer.config == {'model_type': 'char'} # Config doesn't store base args
    assert tokenizer.char_to_idx == {}
    assert tokenizer.unk_token == "<unk>"

def test_char_tokenizer_train(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test training the tokenizer on a text file."""
    # Instantiate with lowercase unk token
    tokenizer = CharTokenizer(unk_token='<unk>')
    output_dir = str(tmp_path / "trained_char_tokenizer")
    
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    # 1. Check vocabulary creation
    char_set = set(sample_text_content)
    # Add expected special tokens (using lowercase unk)
    expected_specials = {tokenizer.pad_token, tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token}
    for special in expected_specials:
        if special: # Only add if defined
            char_set.add(special)
            
    expected_chars = sorted(list(char_set))
    expected_vocab = {char: i for i, char in enumerate(expected_chars)}
    expected_vocab_size = len(expected_chars)
    
    assert tokenizer.get_vocab_size() == expected_vocab_size
    assert tokenizer.char_to_idx == expected_vocab
    assert tokenizer.idx_to_char == {i: char for char, i in expected_vocab.items()}
    assert tokenizer.unk_token == '<unk>'
    assert tokenizer.unk_token_id == expected_vocab.get('<unk>') # Use .get() as UNK might be None
    
    # 2. Check if files were saved
    config_path = Path(output_dir) / "tokenizer_config.json"
    vocab_path = Path(output_dir) / "vocab.json" # CharTokenizer saves vocab too
    assert config_path.exists()
    assert vocab_path.exists()
    
    # 3. Check saved config content
    with open(config_path, 'r', encoding='utf-8') as f:
        saved_config = json.load(f)
    assert saved_config['model_type'] == 'char'
    assert saved_config['vocab_size'] == expected_vocab_size
    # Check saved special tokens section
    saved_specials = saved_config.get('special_tokens', {})
    assert saved_specials.get('unk_token') == '<unk>'
    assert saved_specials.get('unk_id') == tokenizer.unk_token_id
    # Verify other special tokens are saved if they were added
    assert saved_specials.get('pad_token') == tokenizer.pad_token
    assert saved_specials.get('bos_token') == tokenizer.bos_token
    assert saved_specials.get('eos_token') == tokenizer.eos_token
    
    # 4. Check saved vocab content
    with open(vocab_path, 'r', encoding='utf-8') as f:
        saved_vocab = json.load(f)
        assert saved_vocab == tokenizer.char_to_idx

def test_char_tokenizer_train_no_unk(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test training without an unknown token specified (unk_token=None)."""
    # Instantiate with unk_token=None
    tokenizer = CharTokenizer(unk_token=None)
    output_dir = str(tmp_path / "trained_char_tokenizer_no_unk")
    
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    char_set = set(sample_text_content)
    # Add other special tokens even if UNK is None
    expected_specials = {tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token}
    for special in expected_specials:
        if special: # Only add if defined
            char_set.add(special)
            
    expected_chars = sorted(list(char_set))
    expected_vocab_size = len(expected_chars)
    
    assert tokenizer.get_vocab_size() == expected_vocab_size
    assert tokenizer.unk_token is None
    assert tokenizer.unk_token_id is None
    assert '<unk>' not in tokenizer.char_to_idx
    
    # Check saved files
    config_path = Path(output_dir) / "tokenizer_config.json"
    vocab_path = Path(output_dir) / "vocab.json"
    assert config_path.exists()
    assert vocab_path.exists()
    with open(config_path, 'r', encoding='utf-8') as f:
        saved_config = json.load(f)
    assert saved_config['vocab_size'] == expected_vocab_size
    saved_specials = saved_config.get('special_tokens', {})
    assert 'unk_token' not in saved_specials
    assert 'unk_id' not in saved_specials
    # Verify other special tokens are saved
    assert saved_specials.get('pad_token') == tokenizer.pad_token
    assert saved_specials.get('bos_token') == tokenizer.bos_token
    assert saved_specials.get('eos_token') == tokenizer.eos_token

    # Check encoding skips unknown characters when no UNK is defined
    text_with_unknown = "test Z"
    encoded_unknown = tokenizer.encode(text_with_unknown)
    # Calculate expected IDs by only including known chars
    expected_ids_skip = []
    for char in text_with_unknown:
        token_id = tokenizer.char_to_idx.get(char)
        if token_id is not None:
            expected_ids_skip.append(token_id)
    assert encoded_unknown == expected_ids_skip # Should be IDs for 't', 'e', 's', 't', ' '

def test_char_tokenizer_encode_decode(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test encoding and decoding after training."""
    # Use lowercase unk
    tokenizer = CharTokenizer(unk_token='<unk>')
    output_dir = str(tmp_path / "trained_char_tokenizer_encdec")
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    text1 = "hello"
    expected_ids1 = [tokenizer.char_to_idx[c] for c in text1]
    encoded1 = tokenizer.encode(text1)
    assert encoded1 == expected_ids1
    decoded1 = tokenizer.decode(encoded1)
    assert decoded1 == text1
    
    text2 = "test with unknown char Z"
    encoded2 = tokenizer.encode(text2)
    # Expect 'Z' to be mapped to UNK id
    expected_ids2 = []
    unk_id = tokenizer.unk_token_id
    assert unk_id is not None # Make sure unk_id got set
    for char in text2:
        expected_ids2.append(tokenizer.char_to_idx.get(char, unk_id))
    assert encoded2 == expected_ids2
    
    # Decoding should produce the original string but with UNK token for Z
    decoded2 = tokenizer.decode(encoded2)
    # Encode maps 'Z' to unk_id. Decode maps unk_id back to unk_token ('<unk>').
    expected_decoded2 = text2.replace('Z', '<unk>')
    assert decoded2 == expected_decoded2

def test_char_tokenizer_encode_no_unk(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test encoding when no UNK token is defined."""
    tokenizer = CharTokenizer(unk_token=None) # No UNK
    output_dir = str(tmp_path / "trained_char_tokenizer_no_unk_enc")
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    text1 = "hello"
    encoded1 = tokenizer.encode(text1)
    assert all(isinstance(i, int) for i in encoded1)
    decoded1 = tokenizer.decode(encoded1)
    assert decoded1 == text1

    text_with_unknown = "helloZ"
    # Encoding unknown char without UNK should skip the unknown char
    encoded_unknown = tokenizer.encode(text_with_unknown)
    expected_ids_skip = [tokenizer.char_to_idx[c] for c in "hello"]
    assert encoded_unknown == expected_ids_skip
    decoded_skipped = tokenizer.decode(encoded_unknown)
    assert decoded_skipped == "hello"
    
    # Test decoding unknown ID without UNK token (should produce empty string for that ID)
    unknown_id = 9999 # Assume 9999 is not a valid ID
    # Get valid IDs for 'h' and 'e'
    h_id = tokenizer.char_to_idx.get('h')
    e_id = tokenizer.char_to_idx.get('e')
    assert h_id is not None
    assert e_id is not None
    decoded_unknown_id = tokenizer.decode([h_id, unknown_id, e_id])
    assert decoded_unknown_id == "he" # Unknown ID should be skipped/empty

def test_char_tokenizer_save_load(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test saving a trained tokenizer and loading it."""
    # 1. Train and save tokenizer with lowercase unk
    tokenizer_orig = CharTokenizer(unk_token='<unk>')
    output_dir = str(tmp_path / "save_load_test")
    tokenizer_orig.train(temp_text_file_for_tokenizer, output_dir)
    orig_vocab_map  = tokenizer_orig.char_to_idx.copy()
    orig_config     = tokenizer_orig.config.copy() # Get the internal config
    orig_vocab_size = tokenizer_orig.get_vocab_size()
    orig_unk_token = tokenizer_orig.unk_token
    orig_unk_id    = tokenizer_orig.unk_token_id

    # 2. Load tokenizer using the class method
    tokenizer_loaded = CharTokenizer.load(output_dir)

    # 3. Verify loaded state
    # Config might differ slightly due to added keys during save (like special_tokens)
    # Check key aspects instead of direct dict comparison
    assert tokenizer_loaded.config['model_type'] == orig_config['model_type']
    assert tokenizer_loaded.config['vocab_size'] == orig_vocab_size
    assert tokenizer_loaded.char_to_idx == orig_vocab_map
    assert tokenizer_loaded.get_vocab_size() == orig_vocab_size
    assert tokenizer_loaded.idx_to_char == {v: k for k, v in orig_vocab_map.items()}
    assert tokenizer_loaded.unk_token == orig_unk_token
    assert tokenizer_loaded.unk_token_id == orig_unk_id
    # Verify other special tokens are loaded correctly
    assert tokenizer_loaded.pad_token == tokenizer_orig.pad_token
    assert tokenizer_loaded.pad_id == tokenizer_orig.pad_id
    assert tokenizer_loaded.bos_token == tokenizer_orig.bos_token
    assert tokenizer_loaded.bos_id == tokenizer_orig.bos_id
    assert tokenizer_loaded.eos_token == tokenizer_orig.eos_token
    assert tokenizer_loaded.eos_id == tokenizer_orig.eos_id

    # 4. Test encoding/decoding with loaded tokenizer
    text = "hello Zorld"
    encoded = tokenizer_loaded.encode(text)
    decoded = tokenizer_loaded.decode(encoded)
    # Correct the expected string to include the space and the unk token
    expected_decoded = "hello <unk>orld"
    assert decoded == expected_decoded

def test_char_tokenizer_load_legacy_pickle(tmp_path, sample_text_content):
    """Test loading a tokenizer saved in the old pickle format (backward compatibility)."""
    # 1. Create a legacy pickle file manually (simulating old save format)
    legacy_dir = tmp_path / "legacy_tokenizer"
    legacy_dir.mkdir()
    legacy_pickle_path = legacy_dir / "char_tokenizer.pkl"
    
    # Build vocab based on sample text, using lowercase unk
    chars = sorted(list(set(sample_text_content))) + ['<unk>']
    vocab = {c: i for i, c in enumerate(chars)}
    legacy_config = {
        'model_type': 'char',
        'vocab_size': len(vocab),
        # Legacy config stored special tokens like this
        'special_tokens': {'unk': '<unk>'} 
    }
    legacy_data = {
        'config': legacy_config,
        'char_to_idx': vocab,
        'idx_to_char': {i: c for c, i in vocab.items()}
    }
    with open(legacy_pickle_path, 'wb') as f:
        pickle.dump(legacy_data, f)

    # 2. Load using the class method
    tokenizer = CharTokenizer.load(str(legacy_dir)) # Should detect and load pickle

    # 3. Verify loaded state
    # Check key attributes loaded from legacy data
    assert tokenizer.config['model_type'] == legacy_config['model_type']
    assert tokenizer.config['vocab_size'] == legacy_config['vocab_size']
    assert tokenizer.char_to_idx == legacy_data['char_to_idx']
    assert tokenizer.get_vocab_size() == legacy_config['vocab_size']
    assert tokenizer.idx_to_char == legacy_data['idx_to_char']
    assert tokenizer.unk_token == '<unk>' # Loaded from special_tokens
    assert tokenizer.unk_token_id == legacy_data['char_to_idx'].get('<unk>')
    # Verify other special tokens use defaults if not in legacy config
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.bos_token == "<bos>"
    assert tokenizer.eos_token == "<eos>"

def test_char_tokenizer_load_dir_not_found():
    """Test loading from a non-existent directory."""
    tokenizer = CharTokenizer()
    with pytest.raises(FileNotFoundError):
        tokenizer.load("non_existent_directory_12345")

# Add test for decoding with unknown ID when UNK is defined
def test_char_tokenizer_decode_unknown_id_with_unk(temp_text_file_for_tokenizer, tmp_path):
    """Test decoding an unknown ID returns the UNK token string."""
    # Use lowercase unk and a different symbol for clarity
    custom_unk = "<CUSTOM_UNK>"
    tokenizer = CharTokenizer(unk_token=custom_unk)
    output_dir = str(tmp_path / "decode_unk_test")
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    unknown_id = 9999 # Assumed to be an invalid ID
    # Get a valid ID
    h_id = tokenizer.char_to_idx.get('h')
    assert h_id is not None
    
    decoded = tokenizer.decode([h_id, unknown_id, h_id])
    # Expect the custom UNK token string
    assert decoded == f"h{custom_unk}h" 