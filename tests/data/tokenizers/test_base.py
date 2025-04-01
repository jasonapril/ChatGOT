"""
Tests for the base Tokenizer class.
"""
import pytest
import os
import json
from pathlib import Path

from craft.data.tokenizers.base import BaseTokenizer

# --- Fixtures ---

@pytest.fixture
def dummy_tokenizer_config():
    """Provides a dummy config for a tokenizer."""
    return {"model_type": "dummy", "special_tokens": {"unk": "<UNK>"}}

@pytest.fixture
def concrete_tokenizer_class():
    """Creates a minimal concrete implementation of the abstract Tokenizer for testing."""
    class ConcreteTokenizer(BaseTokenizer):
        def __init__(self, config=None, vocab=None):
            # BaseTokenizer __init__ expects config, but the test provided None sometimes
            # Let's ensure it always gets a dict
            super().__init__(config if config is not None else {})
            self.vocab = vocab or {"a": 0, "b": 1, "<UNK>": 2}
            # Config should be updated after super().__init__
            if self.config is None: self.config = {} # Ensure config exists
            self.config['vocab_size'] = len(self.vocab) # Example config update
            self.config['model_type'] = 'concrete' # Identify this type

        def train(self, text_iterator):
            # Dummy train does nothing for base tests
            pass 
        
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
            # Simple dummy encode
            return [self.vocab.get(c, self.vocab.get("<UNK>", -1)) for c in text]

        def decode(self, token_ids: list[int]) -> str:
            # Simple dummy decode
            idx_to_char = {v: k for k, v in self.vocab.items()}
            return "".join([idx_to_char.get(idx, "?") for idx in token_ids])
        
        # Change property name to match base class abstract method
        def get_vocab_size(self) -> int:
            return self.config.get('vocab_size', len(self.vocab))
        
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
    class MinimalTokenizer(BaseTokenizer):
        def __init__(self, config=None):
            super().__init__(config if config is not None else {})
        def train(self, text_iterator): pass
        def load(self, model_path): pass # Add dummy load
        def encode(self, text): return []
        def decode(self, token_ids): return ""
        def get_vocab_size(self): return 0 # Match base method name
        def save(self, output_dir): pass # Add dummy save

    tokenizer = MinimalTokenizer(config=dummy_tokenizer_config)
    assert tokenizer.config == dummy_tokenizer_config
    tokenizer_no_config = MinimalTokenizer()
    assert tokenizer_no_config.config == {} # Should default to empty dict

def test_tokenizer_abstract_methods():
    """Verify that abstract methods raise NotImplementedError if called on base."""
    # We can't instantiate BaseTokenizer directly, but we can check __abstractmethods__
    assert 'train' in BaseTokenizer.__abstractmethods__
    assert 'load' in BaseTokenizer.__abstractmethods__ # Check load is abstract
    assert 'encode' in BaseTokenizer.__abstractmethods__
    assert 'decode' in BaseTokenizer.__abstractmethods__
    assert 'get_vocab_size' in BaseTokenizer.__abstractmethods__ # Check correct name
    assert 'save' in BaseTokenizer.__abstractmethods__ # Check save is abstract

def test_tokenizer_save_load(concrete_tokenizer_class, dummy_tokenizer_config, tmp_path):
    """Test saving and loading a tokenizer configuration using concrete class methods."""
    tokenizer = concrete_tokenizer_class(config=dummy_tokenizer_config.copy())
    save_dir = tmp_path / "tokenizer_save_test"
    
    # Test save (using the concrete implementation)
    tokenizer.save(str(save_dir))
    
    # Check if files were created
    config_path = save_dir / "tokenizer_config.json"
    assert config_path.exists()
    
    # Check config content (should include vocab_size and model_type added by ConcreteTokenizer)
    with open(config_path, 'r') as f:
        loaded_config_data = json.load(f)
    expected_config = dummy_tokenizer_config.copy()
    expected_config['vocab_size'] = 3 
    expected_config['model_type'] = 'concrete' # Added by concrete class
    assert loaded_config_data == expected_config

    # Test load (using the concrete implementation)
    # Create a new instance and load into it
    new_tokenizer = concrete_tokenizer_class(config={}) # Start with empty config
    new_tokenizer.load(str(save_dir))
    assert new_tokenizer.config == expected_config
    assert new_tokenizer.get_vocab_size() == 3 # Check if state updated

    # Test saving to non-existent dir (should create it)
    save_dir_new = tmp_path / "new_dir" / "tokenizer"
    tokenizer.save(str(save_dir_new))
    assert (save_dir_new / "tokenizer_config.json").exists()

# Base class doesn't have a static load_config, removing test
# def test_tokenizer_load_config_not_found(tmp_path): ...

def test_concrete_tokenizer_properties(concrete_tokenizer_class, dummy_tokenizer_config):
    """Test properties implemented in the concrete test class."""
    tokenizer = concrete_tokenizer_class(config=dummy_tokenizer_config.copy())
    # Use the method name from the base class
    assert tokenizer.get_vocab_size() == 3 # From the default vocab {"a": 0, "b": 1, "<UNK>": 2}
    # Test the custom property we added to the concrete class for testing
    assert tokenizer.unk_token_id == 2 # ID of "<UNK>"

    tokenizer_no_unk = concrete_tokenizer_class(config={"model_type": "dummy"})
    assert tokenizer_no_unk.unk_token_id is None

def test_concrete_tokenizer_encode_decode(concrete_tokenizer_class):
    """Test encode/decode methods implemented in the concrete test class."""
    tokenizer = concrete_tokenizer_class()
    text = "aba"
    encoded = tokenizer.encode(text)
    assert encoded == [0, 1, 0]
    
    decoded = tokenizer.decode(encoded)
    assert decoded == text
    
    # Test unknown character
    text_unk = "abc"
    encoded_unk = tokenizer.encode(text_unk)
    assert encoded_unk == [0, 1, 2] # 'c' maps to <UNK> id 2
    
    decoded_unk = tokenizer.decode(encoded_unk)
    # Concrete class decode uses "?" for unknown ids, but finds "<UNK>" for known id 2
    assert decoded_unk == "ab<UNK>"
    
    # Test decoding unknown id
    decoded_bad_id = tokenizer.decode([0, 99, 1])
    assert decoded_bad_id == "a?b" # 99 is not in idx_to_char 