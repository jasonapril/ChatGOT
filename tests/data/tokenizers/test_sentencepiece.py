"""
Tests for the SentencePieceTokenizer.
"""
import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Module under test
from craft.data.tokenizers.sentencepiece import SentencePieceTokenizer

# --- Fixtures ---

@pytest.fixture
def default_sp_config():
    """Provides a default configuration dictionary for SentencePieceTokenizer."""
    return {
        "model_type": "sentencepiece", # Added for potential future factory use
        "vocab_size": 1000,
        "model_prefix": "sp_model_test",
        # Optional fields with defaults
        "character_coverage": 0.9995,
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
    }

@pytest.fixture
def mock_sp_processor():
    """Provides a MagicMock for the sentencepiece.SentencePieceProcessor."""
    mock = MagicMock(name="SentencePieceProcessorInstance")
    mock.load.return_value = None
    mock.encode_as_ids.return_value = [10, 20, 30] # Example IDs
    mock.decode_ids.return_value = "decoded text" # Example text
    mock.get_piece_size.return_value = 1000 # Example vocab size
    return mock

@pytest.fixture
def mock_sp_trainer():
    """Provides a MagicMock for the sentencepiece.SentencePieceTrainer."""
    mock = MagicMock(name="SentencePieceTrainerClass")
    mock.train.return_value = None # train is a static/class method
    return mock

# --- Test Class --- 
# Using a class can help group tests and potentially share mocks if needed

class TestSentencePieceTokenizer:

    def test_init_success(self, default_sp_config):
        """Test successful initialization with a valid config."""
        tokenizer = SentencePieceTokenizer(config=default_sp_config)
        
        assert tokenizer.config == default_sp_config
        assert tokenizer.vocab_size == 1000
        assert tokenizer.model_prefix == "sp_model_test"
        assert tokenizer.character_coverage == 0.9995
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.pad_id == 0
        assert tokenizer.unk_id == 1
        assert tokenizer.bos_id == 2
        assert tokenizer.eos_id == 3
        assert tokenizer.sp is None # Processor not loaded yet
        assert tokenizer.model_path is None # Path not set yet

    def test_init_missing_required_config(self):
        """Test initialization fails if required config keys are missing."""
        # Missing vocab_size
        with pytest.raises(KeyError, match='vocab_size'):
            SentencePieceTokenizer(config={"model_prefix": "test"})
        
        # Missing model_prefix
        with pytest.raises(KeyError, match='model_prefix'):
            SentencePieceTokenizer(config={"vocab_size": 100})

    def test_init_uses_defaults(self):
        """Test that defaults are used for optional config parameters."""
        minimal_config = {
            "vocab_size": 500,
            "model_prefix": "minimal_model",
        }
        tokenizer = SentencePieceTokenizer(config=minimal_config)
        
        assert tokenizer.config == minimal_config # Config remains minimal
        assert tokenizer.vocab_size == 500
        assert tokenizer.model_prefix == "minimal_model"
        # Check defaults
        assert tokenizer.character_coverage == 1.0 # Default in code
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.eos_token == "</s>"
        assert tokenizer.pad_id == 0
        assert tokenizer.unk_id == 1
        assert tokenizer.bos_id == 2
        assert tokenizer.eos_id == 3
        
    # --- Tests for methods requiring mocks --- # 
    # These will use patching

    @patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', autospec=True)
    @patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceTrainer', autospec=True)
    def test_train(self, mock_sp_trainer_cls, mock_sp_processor_cls, default_sp_config, tmp_path, mock_sp_processor):
        """Test the train method calls SentencePieceTrainer and loads the model."""
        # Configure the mock processor class to return our instance mock
        mock_sp_processor_cls.return_value = mock_sp_processor
        
        tokenizer = SentencePieceTokenizer(config=default_sp_config)
        input_text_file = str(tmp_path / "input.txt")
        (tmp_path / "input.txt").touch() # Create dummy file
        output_dir = str(tmp_path / "sp_output")
        
        tokenizer.train(input_text_file, output_dir)
        
        # 1. Verify SentencePieceTrainer.train was called correctly
        expected_model_path_prefix = os.path.join(output_dir, default_sp_config['model_prefix'])
        mock_sp_trainer_cls.train.assert_called_once_with(
            input=input_text_file,
            model_prefix=expected_model_path_prefix,
            vocab_size=default_sp_config['vocab_size'],
            character_coverage=default_sp_config['character_coverage'],
            model_type='bpe', # Hardcoded in implementation
            pad_id=default_sp_config['pad_id'],
            unk_id=default_sp_config['unk_id'],
            bos_id=default_sp_config['bos_id'],
            eos_id=default_sp_config['eos_id'],
            pad_piece=default_sp_config['pad_token'],
            unk_piece=default_sp_config['unk_token'],
            bos_piece=default_sp_config['bos_token'],
            eos_piece=default_sp_config['eos_token'],
            train_extremely_large_corpus=True # Hardcoded in implementation
        )
        
        # 2. Verify the processor was instantiated and load was called
        mock_sp_processor_cls.assert_called_once()
        mock_sp_processor.load.assert_called_once_with(f"{expected_model_path_prefix}.model")
        
        # 3. Verify internal state
        assert tokenizer.sp == mock_sp_processor
        assert tokenizer.model_path == expected_model_path_prefix
        
    @patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', autospec=True)
    def test_load(self, mock_sp_processor_cls, default_sp_config, tmp_path, mock_sp_processor):
        """Test the load method instantiates and loads the SentencePieceProcessor."""
        mock_sp_processor_cls.return_value = mock_sp_processor
        tokenizer = SentencePieceTokenizer(config=default_sp_config)
        model_dir = tmp_path / "existing_model"
        model_dir.mkdir()
        # Create dummy model file that sp.load expects
        model_file_path = model_dir / f"{default_sp_config['model_prefix']}.model"
        model_file_path.touch()
        
        model_path_prefix = str(model_dir / default_sp_config['model_prefix'])
        tokenizer.load(model_path_prefix)
        
        # Verify the processor was instantiated and load was called
        mock_sp_processor_cls.assert_called_once()
        mock_sp_processor.load.assert_called_once_with(str(model_file_path))
        
        # Verify internal state
        assert tokenizer.sp == mock_sp_processor
        assert tokenizer.model_path == model_path_prefix
        
    def test_methods_raise_if_not_initialized(self, default_sp_config):
        """Test encode, decode, get_vocab_size raise RuntimeError if model not loaded."""
        tokenizer = SentencePieceTokenizer(config=default_sp_config)
        
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.encode("test")
        
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.decode([1, 2])

        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.get_vocab_size()
            
    # Helper method to set up an initialized tokenizer with mocks for testing methods
    def _setup_initialized_tokenizer(self, config, mock_processor):
        tokenizer = SentencePieceTokenizer(config=config)
        tokenizer.sp = mock_processor # Manually assign the mock processor
        tokenizer.model_path = "dummy/path/prefix" # Set dummy model path
        return tokenizer
            
    def test_encode(self, default_sp_config, mock_sp_processor):
        """Test encode delegates to the sentencepiece processor."""
        tokenizer = self._setup_initialized_tokenizer(default_sp_config, mock_sp_processor)
        text = "encode this text"
        result = tokenizer.encode(text)
        
        mock_sp_processor.encode_as_ids.assert_called_once_with(text)
        assert result == [10, 20, 30] # Matches mock return value
        
    def test_decode(self, default_sp_config, mock_sp_processor):
        """Test decode delegates to the sentencepiece processor."""
        tokenizer = self._setup_initialized_tokenizer(default_sp_config, mock_sp_processor)
        ids = [10, 20, 30]
        result = tokenizer.decode(ids)
        
        mock_sp_processor.decode_ids.assert_called_once_with(ids)
        assert result == "decoded text" # Matches mock return value
        
    def test_get_vocab_size(self, default_sp_config, mock_sp_processor):
        """Test get_vocab_size delegates to the sentencepiece processor."""
        tokenizer = self._setup_initialized_tokenizer(default_sp_config, mock_sp_processor)
        result = tokenizer.get_vocab_size()
        
        mock_sp_processor.get_piece_size.assert_called_once()
        assert result == 1000 # Matches mock return value
        
    def test_save(self, default_sp_config, tmp_path):
        """Test save writes the configuration to tokenizer_config.json."""
        tokenizer = SentencePieceTokenizer(config=default_sp_config)
        # Need to set model_path, otherwise save raises error
        tokenizer.model_path = str(tmp_path / "dummy_model_prefix") 
        
        save_dir = tmp_path / "save_output"
        tokenizer.save(str(save_dir))
        
        # Check if config file was created
        config_path = save_dir / "tokenizer_config.json"
        assert config_path.exists()
        
        # Check content of saved config
        with open(config_path, 'r') as f:
            saved_data = json.load(f)
            
        # The saved config should match the original config passed to __init__
        # plus any defaults *not* explicitly in the original minimal config if applicable.
        # In this case, default_sp_config had everything.
        expected_saved_config = default_sp_config.copy()
        # Remove the 'model_type' if it was added just for factory use
        expected_saved_config.pop('model_type', None) 
        assert saved_data == expected_saved_config

    def test_save_not_trained(self, default_sp_config):
        """Test save raises RuntimeError if train was never called (model_path is None)."""
        tokenizer = SentencePieceTokenizer(config=default_sp_config)
        with pytest.raises(RuntimeError, match="No model to save"):
            tokenizer.save("some_dir")

        