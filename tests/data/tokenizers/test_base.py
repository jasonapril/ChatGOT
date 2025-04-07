"""
Tests for the base Tokenizer class.
"""
import pytest
import os
import json
from pathlib import Path
from abc import ABC

from craft.data.tokenizers.base import Tokenizer

# --- Fixtures ---

@pytest.fixture
def dummy_tokenizer_config():
    """Provides a dummy config for a tokenizer."""
    return {"model_type": "dummy", "special_tokens": {"unk": "<UNK>"}}

@pytest.fixture
def concrete_tokenizer_class():
    """Creates a minimal concrete implementation of the abstract Tokenizer for testing."""
    class ConcreteTokenizer(Tokenizer):
        def __init__(self, vocab_size=10, **kwargs):
            super().__init__(**kwargs) # Pass kwargs to base init
            self._vocab_size = vocab_size
            self.kwargs = kwargs
            # Concrete specific attributes
            # Determine vocab based on kwargs or default
            vocab = kwargs.get('vocab') # Check if vocab is passed via kwargs
            if vocab is None:
                 # Use base class attributes for special tokens if available
                 # Use lowercase unk consistently
                 unk = self.unk_token if self.unk_token is not None else "<unk>" 
                 unk_id = self.unk_id if self.unk_id is not None else 0 
                 # Default vocab for testing, ensure UNK has its assigned ID
                 vocab = {"a": 0, "b": 1}
                 # Only add UNK if it doesn't clash with existing IDs
                 if unk not in vocab or vocab[unk] == unk_id:
                    vocab[unk] = unk_id
                 elif unk_id not in vocab.values(): # If ID is free, assign it
                    vocab[unk] = unk_id
                 else: # ID clash, maybe log a warning or handle differently? For test, let it be missing.
                    pass 

            self.vocab = vocab
            self.idx_to_char = {v: k for k, v in self.vocab.items()} # Add idx_to_char
            # Store other useful info for tests
            self.config = {
                # Use get_vocab_size() method to be safe
                "vocab_size": self.get_vocab_size(), 
                "model_type": 'concrete',
                "special_tokens": {"unk": self.unk_token} # Store based on base attr
            }
            # Update config with any other kwargs passed (excluding vocab)
            self.config.update({k: v for k, v in kwargs.items() if k != 'vocab'}) 

        def train(self, text_file: str, output_dir: str):
            pass # Not implemented for this test class
        
        # Add dummy load method required by BaseTokenizer
        def load(self, model_path: str) -> None:
            # Dummy load for testing base class functionality
            # In a real scenario, this would load vocab/merges etc.
            config_path = Path(model_path) / "tokenizer_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_cfg = json.load(f)
                    self.config.update(loaded_cfg) # Update internal config
            # Could potentially load a vocab file here too
            pass

        def encode(self, text: str) -> list[int]:
            # Use base class unk_id
            unk = self.unk_id
            # Ensure unk is not None before using it in get
            # Default to -1 or some other invalid ID if unk is None
            # Although base class defaults unk_id to 0, so it shouldn't be None here.
            return [self.vocab.get(c, unk) for c in text]

        def decode(self, token_ids: list[int]) -> str:
            # Simple dummy decode using idx_to_char
            # Use base class unk_token for unknown IDs
            unknown_char = self.unk_token if self.unk_token is not None else ""
            return "".join([self.idx_to_char.get(idx, unknown_char) for idx in token_ids])
        
        # Change property name to match base class abstract method
        def get_vocab_size(self) -> int:
            return len(self.vocab)
        
        # Base class doesn't define unk_token_id, 
        # it's up to specific implementations.
        # We can keep this for testing the concrete class.
        @property
        def unk_token_id(self) -> int | None:
             unk_token = self.config.get("special_tokens", {}).get("unk")
             return self.vocab.get(unk_token)
             
        # Implement the save method from the base class
        def save(self, output_dir: str) -> None:
            """Save the tokenizer configuration."""
            os.makedirs(output_dir, exist_ok=True)
            config_path = os.path.join(output_dir, "tokenizer_config.json")
            # Ensure config exists before saving
            config_to_save = self.config if self.config is not None else {}
            with open(config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)

    return ConcreteTokenizer

# --- Tests ---

def test_tokenizer_base_init(dummy_tokenizer_config):
    """Test basic initialization of the base class (via concrete subclass)."""
    class MinimalTokenizer(Tokenizer):
        def __init__(self, 
                     pad_token="<pad>", unk_token="<unk>", bos_token="<bos>", eos_token="<eos>",
                     pad_id=-1, unk_id=0, bos_id=1, eos_id=2):
            # Only call base init
            super().__init__(pad_token, unk_token, bos_token, eos_token, 
                             pad_id, unk_id, bos_id, eos_id)
        def train(self, text_iterator): pass
        def load(self, model_path): pass # Add dummy load
        def encode(self, text): return []
        def decode(self, token_ids): return ""
        def get_vocab_size(self): return 0 # Match base method name
        def save(self, output_dir): pass # Add dummy save

    # Test with defaults
    tokenizer = MinimalTokenizer()
    assert tokenizer.unk_token == "<unk>"
    # Test passing a value
    tokenizer_custom_unk = MinimalTokenizer(unk_token="<MY_UNK>")
    assert tokenizer_custom_unk.unk_token == "<MY_UNK>"

def test_tokenizer_abstract_methods():
    """Verify that abstract methods raise NotImplementedError if called on base."""
    # We can't instantiate BaseTokenizer directly, but we can check __abstractmethods__
    assert 'train' in Tokenizer.__abstractmethods__
    assert 'load' in Tokenizer.__abstractmethods__ # Check load is abstract
    assert 'encode' in Tokenizer.__abstractmethods__
    assert 'decode' in Tokenizer.__abstractmethods__
    assert 'get_vocab_size' in Tokenizer.__abstractmethods__ # Check correct name
    assert 'save' in Tokenizer.__abstractmethods__ # Check save is abstract

def test_tokenizer_save_load(concrete_tokenizer_class, dummy_tokenizer_config, tmp_path):
    """Test saving and loading a tokenizer configuration using concrete class methods."""
    # Pass the dummy config as kwargs directly
    tokenizer = concrete_tokenizer_class(**dummy_tokenizer_config.copy())
    save_dir = tmp_path / "tokenizer_save_test"
    
    # Test save (using the concrete implementation)
    tokenizer.save(str(save_dir))
    
    # Check if files were created
    config_path = save_dir / "tokenizer_config.json"
    assert config_path.exists()
    
    # Check config content
    with open(config_path, 'r') as f:
        loaded_config_data = json.load(f)
    # Construct expected config based on ConcreteTokenizer's logic
    expected_config = {
        "vocab_size": tokenizer.get_vocab_size(), # Size of the default/passed vocab
        "model_type": 'concrete', # Set by ConcreteTokenizer
        "special_tokens": {"unk": tokenizer.unk_token}, # From base class
         # Include the original kwargs passed if they are stored in self.config
    }
    # Add the original dummy config items back if they were stored
    expected_config.update(dummy_tokenizer_config)
    # Remove potential duplicates or update based on ConcreteTokenizer behavior
    expected_config["vocab_size"] = tokenizer.get_vocab_size() # Ensure correct size
    expected_config["special_tokens"] = {"unk": tokenizer.unk_token}

    # Compare relevant keys, ignore exact dict match if kwargs interfere
    assert loaded_config_data['model_type'] == expected_config['model_type']
    assert loaded_config_data['vocab_size'] == expected_config['vocab_size']
    assert loaded_config_data['special_tokens'] == expected_config['special_tokens']

    # Test load (using the concrete implementation)
    # Create a new instance and load into it
    new_tokenizer = concrete_tokenizer_class() # Start with default
    new_tokenizer.load(str(save_dir)) # Load should update the config
    # Check that loaded config matches saved config
    assert new_tokenizer.config['model_type'] == loaded_config_data['model_type']
    assert new_tokenizer.config['vocab_size'] == loaded_config_data['vocab_size']
    assert new_tokenizer.config['special_tokens'] == loaded_config_data['special_tokens']
    assert new_tokenizer.get_vocab_size() == loaded_config_data['vocab_size'] # Check if state updated

# Base class doesn't have a static load_config, removing test
# def test_tokenizer_load_config_not_found(tmp_path): ...

def test_concrete_tokenizer_properties(concrete_tokenizer_class, dummy_tokenizer_config):
    """Test properties implemented in the concrete test class."""
    # Pass dummy_config as kwargs
    tokenizer = concrete_tokenizer_class(**dummy_tokenizer_config.copy())
    # Base unk_id defaults to 0
    assert tokenizer.get_vocab_size() > 0 # Check vocab is populated
    assert tokenizer.unk_id == 0 # Default base unk_id
    # Check the unk_token_id property specifically
    assert tokenizer.unk_token_id == 0 

    # Test case where unk is not specified or different
    tokenizer_no_unk = concrete_tokenizer_class(unk_token=None)
    assert tokenizer_no_unk.unk_token is None
    assert tokenizer_no_unk.unk_id is None
    assert tokenizer_no_unk.unk_token_id is None

    tokenizer_custom_unk = concrete_tokenizer_class(unk_token="<MY_UNK>", unk_id=5)
    assert tokenizer_custom_unk.unk_token == "<MY_UNK>"
    assert tokenizer_custom_unk.unk_id == 5
    assert tokenizer_custom_unk.unk_token_id == 5

def test_concrete_tokenizer_encode_decode(concrete_tokenizer_class):
    """Test encode/decode methods implemented in the concrete test class."""
    tokenizer = concrete_tokenizer_class() # Uses default vocab {a:0, b:1, <unk>:0}
    text = "aba"
    encoded = tokenizer.encode(text)
    # 'a' -> 0, 'b' -> 1, 'a' -> 0
    assert encoded == [0, 1, 0] 
    
    decoded = tokenizer.decode(encoded)
    # 0 -> 'a', 1 -> 'b', 0 -> 'a'
    assert decoded == text 
    
    # Test unknown character
    text_unk = "abc"
    encoded_unk = tokenizer.encode(text_unk)
    # 'a' -> 0, 'b' -> 1, 'c' -> unk_id (0)
    assert encoded_unk == [0, 1, 0] 
    
    decoded_unk = tokenizer.decode(encoded_unk)
    # 0 -> 'a', 1 -> 'b', 0 -> 'a' (decodes back to 'a' since unk_id is 0)
    assert decoded_unk == "aba" 
    
    # Test decoding unknown id
    decoded_bad_id = tokenizer.decode([0, 99, 1])
    # 0 -> 'a', 99 -> unk_token ("<unk>"), 1 -> 'b'
    assert decoded_bad_id == "a<unk>b" 