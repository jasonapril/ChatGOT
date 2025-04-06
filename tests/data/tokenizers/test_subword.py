"""
Tests for the SubwordTokenizer (using tokenizers library).
"""
import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Module under test
from craft.data.tokenizers.subword import SubwordTokenizer

# --- Fixtures ---

@pytest.fixture
def default_subword_config():
    """Provides a default configuration dictionary for SubwordTokenizer."""
    return {
        "model_type": "subword_bpe", # Added for potential future factory use
        "vocab_size": 5000,
        # Optional fields with defaults
        "pad_token": "<pad>",
        "unk_token": "]",
        "bos_token": "[",
        "eos_token": "]",
    }

@pytest.fixture
def mock_hf_tokenizer():
    """Provides a MagicMock for the tokenizers.Tokenizer instance."""
    mock = MagicMock(name="HFTokenizerInstance")
    mock.save.return_value = None
    mock_encoding = MagicMock()
    mock_encoding.ids = [5, 6, 7, 8] # Example IDs
    mock.encode.return_value = mock_encoding
    mock.decode.return_value = "decoded subword text" # Example text
    mock.get_vocab_size.return_value = 5000 # Example vocab size
    return mock

@pytest.fixture
def mock_bpe_model():
    """Provides a MagicMock for tokenizers.models.BPE."""
    return MagicMock(name="MockBPEModel")

@pytest.fixture
def mock_byte_level_pre_tokenizer():
    """Provides a MagicMock for tokenizers.pre_tokenizers.ByteLevel."""
    return MagicMock(name="MockByteLevel")

@pytest.fixture
def mock_bpe_trainer():
    """Provides a MagicMock for tokenizers.trainers.BpeTrainer."""
    return MagicMock(name="MockBpeTrainer")

# --- Test Class ---

class TestSubwordTokenizer:

    def test_init_success(self, default_subword_config):
        """Test successful initialization with a valid config."""
        tokenizer = SubwordTokenizer(config=default_subword_config)
        
        assert tokenizer.config == default_subword_config
        assert tokenizer.vocab_size == 5000
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "]"
        assert tokenizer.bos_token == "["
        assert tokenizer.eos_token == "]"
        assert tokenizer.tokenizer is None # HF Tokenizer not loaded/trained yet
        
    def test_init_missing_required_config(self):
        """Test initialization fails if required config keys are missing."""
        # Missing vocab_size
        with pytest.raises(KeyError, match='vocab_size'):
            SubwordTokenizer(config={"pad_token": "[PAD]"})
            
    def test_init_uses_defaults(self):
        """Test that defaults are used for optional config parameters."""
        minimal_config = {"vocab_size": 1000}
        tokenizer = SubwordTokenizer(config=minimal_config)
        
        assert tokenizer.config == minimal_config
        assert tokenizer.vocab_size == 1000
        # Check defaults
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.eos_token == "</s>"

    # --- Tests for methods requiring mocks --- # 

    @patch('craft.data.tokenizers.subword.Tokenizer')
    @patch('craft.data.tokenizers.subword.models.BPE')
    @patch('craft.data.tokenizers.subword.pre_tokenizers.ByteLevel')
    @patch('craft.data.tokenizers.subword.trainers.BpeTrainer')
    def test_train(
        self, 
        mock_bpe_trainer_cls, 
        mock_byte_level_cls, 
        mock_bpe_model_cls, 
        mock_hf_tokenizer_cls,
        default_subword_config, 
        mock_hf_tokenizer, # Mock instance
        mock_bpe_model, # Mock instance
        mock_byte_level_pre_tokenizer, # Mock instance
        mock_bpe_trainer # Mock instance
    ):
        """Test the train method initializes and trains the HF tokenizer."""
        # Configure mock classes to return mock instances
        mock_hf_tokenizer_cls.return_value = mock_hf_tokenizer
        mock_bpe_model_cls.return_value = mock_bpe_model
        mock_byte_level_cls.return_value = mock_byte_level_pre_tokenizer
        mock_bpe_trainer_cls.return_value = mock_bpe_trainer
        
        tokenizer = SubwordTokenizer(config=default_subword_config)
        input_files = ["dummy_train1.txt", "dummy_train2.txt"]
        output_dir = "dummy_output_dir" # TODO: Use tmp_path
        
        tokenizer.train(input_files, output_dir)
        
        # 1. Verify Tokenizer, Model, PreTokenizer, Trainer were instantiated
        mock_bpe_model_cls.assert_called_once()
        mock_hf_tokenizer_cls.assert_called_once_with(mock_bpe_model)
        mock_byte_level_cls.assert_called_once_with(add_prefix_space=False)
        mock_bpe_trainer_cls.assert_called_once_with(
            vocab_size=default_subword_config['vocab_size'],
            special_tokens=["<pad>", "]", "[", "]"] # From defaults
        )
        
        # 2. Verify pre_tokenizer was set
        assert mock_hf_tokenizer.pre_tokenizer == mock_byte_level_pre_tokenizer
        
        # 3. Verify tokenizer.train was called with the list of files
        mock_hf_tokenizer.train.assert_called_once_with(files=input_files, trainer=mock_bpe_trainer)
        
        # 4. Verify tokenizer.save was called
        expected_save_path = os.path.join(output_dir, "tokenizer.json")
        mock_hf_tokenizer.save.assert_called_once_with(expected_save_path)
        
        # 5. Verify internal state
        assert tokenizer.tokenizer == mock_hf_tokenizer

    @patch('craft.data.tokenizers.subword.Tokenizer.from_file')
    def test_load(self, mock_from_file, default_subword_config, mock_hf_tokenizer):
        """Test the load method calls Tokenizer.from_file."""
        mock_from_file.return_value = mock_hf_tokenizer
        tokenizer = SubwordTokenizer(config=default_subword_config)
        model_dir = "some/model/dir"
        expected_load_path = os.path.join(model_dir, "tokenizer.json")
        
        tokenizer.load(model_dir)
        
        mock_from_file.assert_called_once_with(expected_load_path)
        assert tokenizer.tokenizer == mock_hf_tokenizer
        
    def test_methods_raise_if_not_initialized(self, default_subword_config):
        """Test encode, decode, get_vocab_size, save raise error if not initialized."""
        tokenizer = SubwordTokenizer(config=default_subword_config)
        
        with pytest.raises(ValueError, match="Tokenizer not initialized"):
            tokenizer.encode("test")
        
        with pytest.raises(ValueError, match="Tokenizer not initialized"):
            tokenizer.decode([1, 2])

        # get_vocab_size should return config vocab_size if not initialized
        assert tokenizer.get_vocab_size() == default_subword_config['vocab_size'] 
        
        # Now initialize it minimally for the save test
        # tokenizer.tokenizer = MagicMock() 
        # The following check was incorrect; save only raises if self.tokenizer is None.
        # The correct check for the uninitialized case is done below.
        # with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
        #     pass 

        # Test save raises RuntimeError (from SubwordTokenizer) when tokenizer is None
        tokenizer_not_loaded = SubwordTokenizer(config=default_subword_config)
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
             tokenizer_not_loaded.save("some_dir")

    # Helper method to set up an initialized tokenizer
    def _setup_initialized_tokenizer(self, config, mock_hf_tokenizer_instance):
        tokenizer = SubwordTokenizer(config=config)
        tokenizer.tokenizer = mock_hf_tokenizer_instance # Manually assign the mock
        return tokenizer
            
    def test_encode(self, default_subword_config, mock_hf_tokenizer):
        """Test encode delegates to the HF tokenizer instance."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config, mock_hf_tokenizer)
        text = "encode this text"
        result = tokenizer.encode(text)
        
        mock_hf_tokenizer.encode.assert_called_once_with(text)
        assert result == [5, 6, 7, 8] # Matches mock encoding ids
        
    def test_decode(self, default_subword_config, mock_hf_tokenizer):
        """Test decode delegates to the HF tokenizer instance."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config, mock_hf_tokenizer)
        ids = [5, 6, 7, 8]
        result = tokenizer.decode(ids)
        
        mock_hf_tokenizer.decode.assert_called_once_with(ids)
        assert result == "decoded subword text" # Matches mock return value
        
    def test_get_vocab_size_initialized(self, default_subword_config, mock_hf_tokenizer):
        """Test get_vocab_size delegates to HF tokenizer when initialized."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config, mock_hf_tokenizer)
        result = tokenizer.get_vocab_size()
        
        mock_hf_tokenizer.get_vocab_size.assert_called_once()
        # Should return the value from the mock HF tokenizer, not the config
        assert result == 5000 # Matches mock return value 
        
    def test_save(self, default_subword_config, mock_hf_tokenizer, tmp_path):
        """Test save delegates to the HF tokenizer instance's save method."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config, mock_hf_tokenizer)
        save_dir = tmp_path / "save_output"
        expected_save_path = str(save_dir / "tokenizer.json")

        tokenizer.save(str(save_dir))
        
        mock_hf_tokenizer.save.assert_called_once_with(expected_save_path)

    def test_save_not_trained(self, tmp_path):
        """Test saving fails if tokenizer is not trained."""
        tokenizer_not_loaded = SubwordTokenizer(config={"vocab_size": 100})
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer_not_loaded.save(tmp_path / "some_dir")

