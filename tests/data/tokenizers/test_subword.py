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
        # Provide vocab_size directly, as config dict is no longer the primary init method
        tokenizer = SubwordTokenizer(
            vocab_size=default_subword_config['vocab_size'],
            unk_token=default_subword_config['unk_token'],
            bos_token=default_subword_config['bos_token'],
            eos_token=default_subword_config['eos_token']
        )
        
        # Config is now stored internally but not the primary interface
        # assert tokenizer.config == default_subword_config 
        assert tokenizer.vocab_size == 5000
        assert tokenizer.unk_token == "]" # From fixture
        assert tokenizer.bos_token == "[" # From fixture
        assert tokenizer.eos_token == "]" # From fixture
        assert tokenizer.tokenizer is None # HF Tokenizer not loaded/trained yet
        
    def test_init_missing_required_config(self):
        """Test initialization fails if required config keys are missing."""
        # Missing vocab_size
        with pytest.raises(TypeError, match="required positional argument: 'vocab_size'"):
            SubwordTokenizer() # Call without vocab_size
            
    def test_init_uses_defaults(self):
        """Test that defaults are used for optional config parameters."""
        # Provide only vocab_size
        tokenizer = SubwordTokenizer(vocab_size=1000) 
        
        assert tokenizer.vocab_size == 1000
        # Check defaults from base class
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "<unk>"
        # Check defaults set in SubwordTokenizer init signature
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.eos_token == "</s>"

    # --- Tests for methods requiring mocks --- # 

    @patch('craft.data.tokenizers.subword.HFTokenizer') # Patch the aliased HFTokenizer class
    @patch('craft.data.tokenizers.subword.models.BPE')
    @patch('craft.data.tokenizers.subword.pre_tokenizers.ByteLevel')
    @patch('craft.data.tokenizers.subword.trainers.BpeTrainer')
    def test_train(
        self, 
        mock_bpe_trainer_cls, 
        mock_byte_level_cls, 
        mock_bpe_model_cls, 
        mock_hf_tokenizer_cls, # Mock for the HFTokenizer *class* 
        default_subword_config, 
        mock_hf_tokenizer, # Mock instance provided by fixture
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
        
        # Instantiate with vocab_size and other params
        tokenizer = SubwordTokenizer(
            vocab_size=default_subword_config['vocab_size'],
            unk_token=default_subword_config['unk_token'],
            bos_token=default_subword_config['bos_token'],
            eos_token=default_subword_config['eos_token']
        )
        input_files = ["dummy_train1.txt", "dummy_train2.txt"]
        output_dir = "outputs/test_subword" # TODO: Use tmp_path
        
        tokenizer.train(input_files, output_dir)
        
        # 1. Verify Tokenizer, Model, PreTokenizer, Trainer were instantiated
        # BPE model gets unk_token passed now
        mock_bpe_model_cls.assert_called_once_with(unk_token=tokenizer.unk_token) 
        mock_hf_tokenizer_cls.assert_called_once_with(mock_bpe_model)
        mock_byte_level_cls.assert_called_once_with(add_prefix_space=False)
        mock_bpe_trainer_cls.assert_called_once_with(
            vocab_size=default_subword_config['vocab_size'],
            # Special tokens come from the base class attributes now
            special_tokens=[tokenizer.pad_token, tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token] 
        )
        
        # 2. Verify pre_tokenizer was set
        assert mock_hf_tokenizer.pre_tokenizer == mock_byte_level_pre_tokenizer
        
        # 3. Verify tokenizer.train was called with the list of files
        mock_hf_tokenizer.train.assert_called_once_with(files=input_files, trainer=mock_bpe_trainer)
        
        # 4. Verify tokenizer.save was called via the instance's save method
        # We now mock the instance's save method directly if needed, or check effects
        # Here, train calls self.save(), which calls self.tokenizer.save()
        expected_save_path = os.path.join(output_dir, "tokenizer.json")
        mock_hf_tokenizer.save.assert_called_once_with(expected_save_path)
        
        # 5. Verify internal state
        assert tokenizer.tokenizer == mock_hf_tokenizer

    # Patch the correct static method on the aliased HFTokenizer class
    @patch('craft.data.tokenizers.subword.HFTokenizer.from_file') 
    def test_load(self, mock_hf_from_file, default_subword_config, mock_hf_tokenizer):
        """Test the load method calls HFTokenizer.from_file and updates state."""
        # Configure the mock returned by from_file
        mock_hf_tokenizer.get_vocab_size.return_value = 5555 # Different size
        mock_hf_from_file.return_value = mock_hf_tokenizer
        
        # Instantiate with initial vocab size
        tokenizer = SubwordTokenizer(vocab_size=default_subword_config['vocab_size'])
        model_dir = "some/model/dir" # TODO: use tmp_path
        expected_load_path = os.path.join(model_dir, "tokenizer.json")
        
        # Mock os.path.exists needed by load
        with patch('os.path.exists', return_value=True) as mock_exists:
            tokenizer.load(model_dir)
            mock_exists.assert_called_once_with(expected_load_path)
        
        mock_hf_from_file.assert_called_once_with(expected_load_path)
        assert tokenizer.tokenizer == mock_hf_tokenizer
        # Assert that the instance's vocab_size was updated
        assert tokenizer.vocab_size == 5555 
        
    def test_methods_raise_if_not_initialized(self, default_subword_config):
        """Test encode, decode, save raise error if HF tokenizer instance not loaded/trained."""
        # Provide vocab_size on init
        tokenizer = SubwordTokenizer(vocab_size=default_subword_config['vocab_size'])
        
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.encode("test")
        
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.decode([1, 2])

        # get_vocab_size should return config vocab_size if not initialized
        assert tokenizer.get_vocab_size() == default_subword_config['vocab_size'] 
        
        # Test save raises RuntimeError when tokenizer is None
        with pytest.raises(RuntimeError, match="Tokenizer not initialized. Cannot save."):
             tokenizer.save("some_dir")

    # Helper method to set up an initialized tokenizer
    def _setup_initialized_tokenizer(self, vocab_size, mock_hf_tokenizer_instance):
        tokenizer = SubwordTokenizer(vocab_size=vocab_size)
        tokenizer.tokenizer = mock_hf_tokenizer_instance # Manually assign the mock
        return tokenizer
            
    def test_encode(self, default_subword_config, mock_hf_tokenizer):
        """Test encode delegates to the HF tokenizer instance."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config['vocab_size'], mock_hf_tokenizer)
        text = "encode this text"
        result = tokenizer.encode(text)
        
        # HF tokenizer encode returns an Encoding object, check the call
        mock_hf_tokenizer.encode.assert_called_once_with(text)
        # Assert the returned ids match the mock Encoding object's ids
        assert result == [5, 6, 7, 8] 
        
    def test_decode(self, default_subword_config, mock_hf_tokenizer):
        """Test decode delegates to the HF tokenizer instance."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config['vocab_size'], mock_hf_tokenizer)
        ids = [5, 6, 7, 8]
        # Test with default skip_special_tokens=True
        result_skipped = tokenizer.decode(ids)
        mock_hf_tokenizer.decode.assert_called_with(ids, skip_special_tokens=True)
        assert result_skipped == "decoded subword text" # Matches mock return value
        # Test with skip_special_tokens=False
        result_not_skipped = tokenizer.decode(ids, skip_special_tokens=False)
        mock_hf_tokenizer.decode.assert_called_with(ids, skip_special_tokens=False)
        assert result_not_skipped == "decoded subword text" # Mock doesn't change based on skip
        
    def test_get_vocab_size_initialized(self, default_subword_config, mock_hf_tokenizer):
        """Test get_vocab_size delegates to HF tokenizer when initialized."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config['vocab_size'], mock_hf_tokenizer)
        result = tokenizer.get_vocab_size()
        
        mock_hf_tokenizer.get_vocab_size.assert_called_once()
        # Should return the value from the mock HF tokenizer
        assert result == 5000 # Matches mock get_vocab_size return value 
        
    def test_save(self, default_subword_config, mock_hf_tokenizer, tmp_path):
        """Test save delegates to the HF tokenizer instance's save method."""
        tokenizer = self._setup_initialized_tokenizer(default_subword_config['vocab_size'], mock_hf_tokenizer)
        save_dir = tmp_path / "save_output"
        expected_save_path = str(save_dir / "tokenizer.json")

        # Mock os.makedirs which is called by save
        with patch('os.makedirs') as mock_mkdirs:
            tokenizer.save(str(save_dir))
            mock_mkdirs.assert_called_once_with(save_dir, exist_ok=True)
        
        # Assert that the internal HFTokenizer mock's save was called
        mock_hf_tokenizer.save.assert_called_once_with(expected_save_path)

    def test_save_not_trained(self, tmp_path):
        """Test saving fails if tokenizer is not trained."""
        # Provide vocab_size on init
        tokenizer_not_loaded = SubwordTokenizer(vocab_size=100)
        with pytest.raises(RuntimeError, match="Tokenizer not initialized. Cannot save."):
            tokenizer_not_loaded.save(tmp_path / "some_dir")

