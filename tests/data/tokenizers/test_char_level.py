"""
Tests for the CharLevelTokenizer.
"""
import pytest
import os
import json
import pickle
from pathlib import Path

from craft.data.tokenizers.char_level import CharLevelTokenizer

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

# --- Tests ---

def test_char_tokenizer_init_empty():
    """Test initializing an empty CharLevelTokenizer."""
    tokenizer = CharLevelTokenizer()
    assert tokenizer.char_to_idx == {}
    assert tokenizer.config == {'model_type': 'char_level'} # Should have default model_type
    assert tokenizer.get_vocab_size() == 0
    assert tokenizer.idx_to_char == {}
    assert tokenizer.unk_token is None
    assert tokenizer.unk_token_id is None

def test_char_tokenizer_init_with_config():
    """Test initializing with a specific config."""
    config = {
        'model_type': 'char_level',
        'special_tokens': {'unk': '<UNK>'}
    }
    tokenizer = CharLevelTokenizer(config=config)
    assert tokenizer.config == config
    assert tokenizer.char_to_idx == {}
    assert tokenizer.unk_token == '<UNK>'
    assert tokenizer.unk_token_id is None # ID depends on vocab

def test_char_tokenizer_train(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test training the tokenizer on a text file."""
    tokenizer = CharLevelTokenizer(config={'special_tokens': {'unk': '<UNK>'}})
    output_dir = str(tmp_path / "trained_char_tokenizer")
    
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    # 1. Check vocabulary creation
    char_set = set(sample_text_content)
    char_set.add('<UNK>') # Add UNK before sorting
    expected_chars = sorted(list(char_set))
    expected_vocab = {char: i for i, char in enumerate(expected_chars)}
    expected_vocab_size = len(expected_chars)
    
    assert tokenizer.get_vocab_size() == expected_vocab_size
    assert tokenizer.char_to_idx == expected_vocab
    assert tokenizer.idx_to_char == {i: char for char, i in expected_vocab.items()}
    assert tokenizer.unk_token == '<UNK>'
    assert tokenizer.unk_token_id == expected_vocab['<UNK>']
    
    # 2. Check if files were saved
    config_path = Path(output_dir) / "tokenizer_config.json"
    vocab_path = Path(output_dir) / "vocab.json" # CharLevelTokenizer saves vocab too
    assert config_path.exists()
    assert vocab_path.exists()
    
    # 3. Check saved config content
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    assert saved_config['model_type'] == 'char_level'
    assert saved_config['vocab_size'] == expected_vocab_size
    assert saved_config['special_tokens']['unk'] == '<UNK>'
    
    # 4. Check saved vocab content
    with open(vocab_path, 'r') as f:
        saved_vocab = json.load(f)
        assert saved_vocab == tokenizer.char_to_idx

def test_char_tokenizer_train_no_unk(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test training without an unknown token specified."""
    tokenizer = CharLevelTokenizer() # No UNK in config
    output_dir = str(tmp_path / "trained_char_tokenizer_no_unk")
    
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    expected_chars = sorted(list(set(sample_text_content)))
    expected_vocab_size = len(expected_chars)
    
    assert tokenizer.get_vocab_size() == expected_vocab_size
    assert tokenizer.unk_token is None
    assert tokenizer.unk_token_id is None
    assert '<UNK>' not in tokenizer.char_to_idx
    
    # Check saved files
    assert (Path(output_dir) / "tokenizer_config.json").exists()
    assert (Path(output_dir) / "vocab.json").exists()
    with open(Path(output_dir) / "tokenizer_config.json", 'r') as f:
        saved_config = json.load(f)
    assert saved_config['vocab_size'] == expected_vocab_size
    assert 'special_tokens' not in saved_config or saved_config.get('special_tokens') == {}

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
    tokenizer = CharLevelTokenizer(config={'special_tokens': {'unk': '<UNK>'}})
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
    for char in text2:
        expected_ids2.append(tokenizer.char_to_idx.get(char, unk_id))
    assert encoded2 == expected_ids2
    
    # Decoding should produce the original string but with UNK token for Z
    decoded2 = tokenizer.decode(encoded2)
    expected_decoded2 = "test with <UNK><UNK><UNK><UNK>ow<UNK> <UNK>har <UNK>"
    assert decoded2 == expected_decoded2

def test_char_tokenizer_encode_no_unk(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test encoding when no UNK token is defined."""
    tokenizer = CharLevelTokenizer() # No UNK
    output_dir = str(tmp_path / "trained_char_tokenizer_no_unk_enc")
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    text1 = "hello"
    encoded1 = tokenizer.encode(text1)
    assert all(isinstance(i, int) for i in encoded1)
    
    text_with_unknown = "helloZ"
    # Encoding unknown char without UNK should raise an error or skip?
    # Current implementation skips unknown characters if no unk_token_id
    encoded_unknown = tokenizer.encode(text_with_unknown)
    expected_ids_skip = [tokenizer.char_to_idx[c] for c in "hello"]
    assert encoded_unknown == expected_ids_skip
    
    # Decoding should work for known IDs
    decoded1 = tokenizer.decode(encoded1)
    assert decoded1 == text1
    
    # Test decoding unknown ID without UNK token (should produce empty string for that ID)
    unknown_id = 99 # Assume 99 is not a valid ID
    decoded_unknown_id = tokenizer.decode([tokenizer.char_to_idx['h'], unknown_id, tokenizer.char_to_idx['e']])
    assert decoded_unknown_id == "he" # Unknown ID should be skipped/empty

def test_char_tokenizer_save_load(temp_text_file_for_tokenizer, sample_text_content, tmp_path):
    """Test saving a trained tokenizer and loading it."""
    # 1. Train and save tokenizer
    tokenizer_orig = CharLevelTokenizer(config={'special_tokens': {'unk': '<UNK>'}})
    output_dir = str(tmp_path / "save_load_test")
    tokenizer_orig.train(temp_text_file_for_tokenizer, output_dir)
    orig_vocab_map  = tokenizer_orig.char_to_idx.copy()
    orig_config     = tokenizer_orig.config.copy()
    orig_vocab_size = tokenizer_orig.get_vocab_size()

    # 2. Load tokenizer using the class method
    tokenizer_loaded = CharLevelTokenizer.load(output_dir)

    # 3. Verify loaded state
    assert tokenizer_loaded.config == orig_config
    assert tokenizer_loaded.char_to_idx == orig_vocab_map
    assert tokenizer_loaded.get_vocab_size() == orig_vocab_size
    assert tokenizer_loaded.idx_to_char == {v: k for k, v in orig_vocab_map.items()}
    assert tokenizer_loaded.unk_token == '<UNK>'
    assert tokenizer_loaded.unk_token_id == orig_vocab_map['<UNK>']

    # 4. Test encoding/decoding with loaded tokenizer
    text = "hello Zorld"
    encoded = tokenizer_loaded.encode(text)
    decoded = tokenizer_loaded.decode(encoded)
    # Correct the expected string to include the space
    assert decoded == "hello <UNK>orld"

def test_char_tokenizer_load_legacy_pickle(tmp_path, sample_text_content):
    """Test loading a tokenizer saved in the old pickle format (backward compatibility)."""
    # 1. Create a legacy pickle file manually (simulating old save format)
    legacy_dir = tmp_path / "legacy_tokenizer"
    legacy_dir.mkdir()
    legacy_pickle_path = legacy_dir / "char_tokenizer.pkl"
    
    # Build vocab based on sample text
    chars = sorted(list(set(sample_text_content))) + ['<UNK>']
    vocab = {c: i for i, c in enumerate(chars)}
    legacy_data = {
        'config': {
            'model_type': 'char_level',
            'vocab_size': len(vocab),
            'special_tokens': {'unk': '<UNK>'}
        },
        'char_to_idx': vocab,
        'idx_to_char': {i: c for c, i in vocab.items()}
    }
    with open(legacy_pickle_path, 'wb') as f:
        pickle.dump(legacy_data, f)

    # 2. Load using the class method
    tokenizer = CharLevelTokenizer.load(str(legacy_dir)) # Should detect and load pickle

    # 3. Verify loaded state
    assert tokenizer.config == legacy_data['config']
    assert tokenizer.char_to_idx == legacy_data['char_to_idx']
    assert tokenizer.get_vocab_size() == legacy_data['config']['vocab_size']
    assert tokenizer.idx_to_char == legacy_data['idx_to_char']
    assert tokenizer.unk_token == '<UNK>'
    assert tokenizer.unk_token_id == legacy_data['char_to_idx']['<UNK>']

def test_char_tokenizer_load_dir_not_found():
    """Test loading from a non-existent directory."""
    tokenizer = CharLevelTokenizer()
    with pytest.raises(FileNotFoundError):
        tokenizer.load("non_existent_directory_12345")

# Add test for decoding with unknown ID when UNK is defined
def test_char_tokenizer_decode_unknown_id_with_unk(temp_text_file_for_tokenizer, tmp_path):
    """Test decoding an unknown ID returns the UNK token string."""
    tokenizer = CharLevelTokenizer(config={'special_tokens': {'unk': '<UNKNOWNSYMBOL>'}})
    output_dir = str(tmp_path / "decode_unk_test")
    tokenizer.train(temp_text_file_for_tokenizer, output_dir)
    
    unknown_id = 999 # Assumed to be an invalid ID
    known_id = tokenizer.char_to_idx['h']
    decoded = tokenizer.decode([known_id, unknown_id, known_id])
    assert decoded == "h<UNKNOWNSYMBOL>h" 