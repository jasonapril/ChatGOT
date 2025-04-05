"""
Tests for the SentencePieceTokenizer.
"""
import pytest
import os
import json
import logging # Import logging
import re # <--- Import re
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY, mock_open
from typing import Any, Dict

# Module under test
from craft.data.tokenizers.sentencepiece import SentencePieceTokenizer

# --- Fixtures ---

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

    def test_init_success(self):
        """Test successful initialization with keyword arguments."""
        tokenizer = SentencePieceTokenizer(
            vocab_size=1000,
            model_prefix="sp_model_test",
            character_coverage=0.9995,
            pad_token="<PAD>",
            unk_token="<UNK>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_id=99,
            unk_id=98,
            bos_id=97,
            eos_id=96,
        )

        assert tokenizer._initial_vocab_size == 1000
        assert tokenizer._initial_model_prefix == "sp_model_test"
        assert tokenizer.character_coverage == 0.9995
        assert tokenizer.pad_token == "<PAD>"
        assert tokenizer.unk_token == "<UNK>"
        assert tokenizer.bos_token == "<BOS>"
        assert tokenizer.eos_token == "<EOS>"
        assert tokenizer.pad_id == 99
        assert tokenizer.unk_id == 98
        assert tokenizer.bos_id == 97
        assert tokenizer.eos_id == 96
        assert tokenizer.sp is None # Processor not loaded yet
        assert tokenizer._loaded_model_path is None # Path not set yet

    def test_init_load_model(self, tmp_path, mock_sp_processor):
        """Test initialization successfully loads model if model_path is provided."""
        model_dir = tmp_path / "load_init_model"
        model_dir.mkdir()
        model_prefix = "my_sp_model"
        model_file_path = model_dir / f"{model_prefix}.model"
        model_file_path.touch() # Create dummy file
        model_path_arg = str(model_dir / model_prefix)

        with patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', return_value=mock_sp_processor) as mock_sp_proc_class, \
             patch('os.path.exists', return_value=True) as mock_exists: # Ensure exists returns True

            tokenizer = SentencePieceTokenizer(
                model_path=model_path_arg, # Provide path to load
                # Other args can be defaults or specific, load should override vocab_size
                vocab_size=500 # Initial config size, will be overridden
            )

            mock_exists.assert_called_once_with(str(model_file_path))
            mock_sp_proc_class.assert_called_once()
            mock_sp_processor.load.assert_called_once_with(str(model_file_path))
            assert tokenizer.sp == mock_sp_processor
            assert tokenizer._loaded_model_path == model_path_arg
            # Vocab size should come from the mocked processor's get_piece_size
            assert tokenizer.get_vocab_size() == 1000

    def test_init_load_model_file_not_found(self, tmp_path):
        """Test initialization logs error if model_path file doesn't exist."""
        model_path_arg = str(tmp_path / "nonexistent_model")

        with patch('os.path.exists', return_value=False) as mock_exists, \
             patch('craft.data.tokenizers.sentencepiece.logger') as mock_logger:
            # Expect init not to fail, but to log an error
            tokenizer = SentencePieceTokenizer(model_path=model_path_arg)
            mock_exists.assert_called_once_with(f"{model_path_arg}.model")
            mock_logger.error.assert_any_call(f"SentencePiece model file not found: {model_path_arg}.model")
            assert tokenizer.sp is None

    def test_init_warning_if_missing_args_for_training(self, caplog):
        """Test initialization warns if required training args missing and no model_path."""
        with caplog.at_level(logging.WARNING):
            # Missing vocab_size
            SentencePieceTokenizer(model_prefix="test")
            assert "missing vocab_size or model_prefix" in caplog.text
            caplog.clear()
            # Missing model_prefix
            SentencePieceTokenizer(vocab_size=100)
            assert "missing vocab_size or model_prefix" in caplog.text
            caplog.clear()
            # Both missing
            SentencePieceTokenizer()
            assert "missing vocab_size or model_prefix" in caplog.text

    def test_init_uses_defaults(self):
        """Test that defaults are used for optional config parameters."""
        tokenizer = SentencePieceTokenizer(
            vocab_size=500,
            model_prefix="minimal_model",
        )

        assert tokenizer._initial_vocab_size == 500
        assert tokenizer._initial_model_prefix == "minimal_model"
        # Check defaults from __init__ signature
        assert tokenizer.character_coverage == 1.0
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
    @patch('os.path.exists', return_value=True) # Mock exists for loading after train
    def test_train(self, mock_exists, mock_sp_trainer_cls, mock_sp_processor_cls, tmp_path, mock_sp_processor):
        """Test the train method calls SentencePieceTrainer and loads the model."""
        mock_sp_processor_cls.return_value = mock_sp_processor

        # Initialize with required args for training
        tokenizer = SentencePieceTokenizer(
            vocab_size=1000,
            model_prefix="sp_train_test",
            character_coverage=0.98,
            pad_token="<PAD>", unk_token="<UNK>", bos_token="<BOS>", eos_token="<EOS>",
            pad_id=99, unk_id=98, bos_id=97, eos_id=96,
        )
        input_files_list = [str(tmp_path / "input1.txt"), str(tmp_path / "input2.txt")]
        for f in input_files_list:
            Path(f).touch() # Create dummy files
        output_dir = str(tmp_path / "sp_output")

        input_files_str = ",".join(input_files_list)

        # Call train
        tokenizer.train(input_files_list, output_dir)

        # 1. Verify SentencePieceTrainer.train was called correctly
        expected_model_save_prefix = os.path.join(output_dir, "sp_train_test")
        mock_sp_trainer_cls.train.assert_called_once_with(
            input=input_files_str,
            model_prefix=expected_model_save_prefix,
            vocab_size=1000, # From init
            character_coverage=0.98, # From init
            model_type='bpe', # Hardcoded
            pad_id=99, # From init
            unk_id=98, # From init
            bos_id=97, # From init
            eos_id=96, # From init
            pad_piece="<PAD>", # From init
            unk_piece="<UNK>", # From init
            bos_piece="<BOS>", # From init
            eos_piece="<EOS>", # From init
            # Removed train_extremely_large_corpus
        )

        # 2. Verify the processor was instantiated and load was called (by _load_model)
        expected_model_file = f"{expected_model_save_prefix}.model"
        mock_exists.assert_any_call(expected_model_file) # Check if load checked existence
        mock_sp_processor_cls.assert_called_once()
        mock_sp_processor.load.assert_called_once_with(expected_model_file)

        # 3. Verify internal state
        assert tokenizer.sp == mock_sp_processor
        assert tokenizer._loaded_model_path == expected_model_save_prefix

    def test_train_raises_if_config_missing(self):
        """Test train raises ValueError if vocab_size or model_prefix missing."""
        tokenizer_no_vocab = SentencePieceTokenizer(model_prefix="test")
        with pytest.raises(ValueError, match="Cannot train .* without 'vocab_size' and 'model_prefix'"):
            tokenizer_no_vocab.train(["dummy.txt"], "dummy_out")

        tokenizer_no_prefix = SentencePieceTokenizer(vocab_size=100)
        with pytest.raises(ValueError, match="Cannot train .* without 'vocab_size' and 'model_prefix'"):
            tokenizer_no_prefix.train(["dummy.txt"], "dummy_out")

    @patch('os.path.exists', return_value=True)
    @patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', autospec=True)
    def test_load(self, mock_sp_processor_cls, mock_exists, tmp_path, mock_sp_processor):
        """Test the load method instantiates and loads the SentencePieceProcessor."""
        mock_sp_processor_cls.return_value = mock_sp_processor
        tokenizer = SentencePieceTokenizer(vocab_size=500, model_prefix="irrelevant") # Init doesn't matter for load
        model_dir = tmp_path / "existing_model"
        model_prefix_name = "my_sp_model"
        model_file_path = model_dir / f"{model_prefix_name}.model"

        model_path_prefix_arg = str(model_dir / model_prefix_name)
        tokenizer.load(model_path_prefix_arg)

        # Verify exists check, processor instantiation, and load call
        mock_exists.assert_called_once_with(str(model_file_path))
        mock_sp_processor_cls.assert_called_once()
        mock_sp_processor.load.assert_called_once_with(str(model_file_path))

        # Verify internal state
        assert tokenizer.sp == mock_sp_processor
        assert tokenizer._loaded_model_path == model_path_prefix_arg # Check internal path store

    @patch('os.path.exists', return_value=False)
    def test_load_file_not_found_raises(self, mock_exists, tmp_path):
        """Test load raises FileNotFoundError if the model file doesn't exist."""
        tokenizer = SentencePieceTokenizer()
        model_path_prefix = str(tmp_path / "nonexistent")
        expected_model_file = f"{model_path_prefix}.model"

        with pytest.raises(FileNotFoundError, match=re.escape(f"SentencePiece model file not found: {expected_model_file}")):
            tokenizer.load(model_path_prefix)
        mock_exists.assert_called_once_with(expected_model_file)

    def test_methods_raise_if_not_initialized(self):
        """Test encode, decode raise RuntimeError if model not loaded/trained."""
        # Initialize without model_path, don't call train/load
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_prefix="test")

        assert tokenizer.sp is None
        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.encode("test")

        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            tokenizer.decode([1, 2])

    def test_get_vocab_size_not_initialized(self):
        """Test get_vocab_size returns initial config value or 0 if model not loaded."""
        tokenizer_with_size = SentencePieceTokenizer(vocab_size=500, model_prefix="test")
        assert tokenizer_with_size.sp is None
        assert tokenizer_with_size.get_vocab_size() == 500 # Returns initial config value

        tokenizer_without_size = SentencePieceTokenizer(model_prefix="test")
        assert tokenizer_without_size.sp is None
        assert tokenizer_without_size._initial_vocab_size is None
        assert tokenizer_without_size.get_vocab_size() == 0 # Returns 0 if initial was None

    # Helper method to set up an initialized tokenizer with mocks for testing methods
    def _setup_initialized_tokenizer(self, mock_processor, **kwargs):
        # Initialize with some basic args, they don't matter if sp is mocked
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_prefix="dummy", **kwargs)
        tokenizer.sp = mock_processor # Manually assign the mock processor
        tokenizer._loaded_model_path = "dummy/path/prefix" # Set dummy internal path
        return tokenizer

    def test_encode(self, mock_sp_processor):
        """Test encode delegates to the sentencepiece processor."""
        tokenizer = self._setup_initialized_tokenizer(mock_sp_processor)
        text = "encode this text"
        result = tokenizer.encode(text)

        mock_sp_processor.encode_as_ids.assert_called_once_with(text)
        assert result == [10, 20, 30] # Matches mock return value

    def test_decode(self, mock_sp_processor):
        """Test decode delegates to the sentencepiece processor."""
        tokenizer = self._setup_initialized_tokenizer(mock_sp_processor)
        ids = [10, 20, 30]
        result = tokenizer.decode(ids)

        mock_sp_processor.decode_ids.assert_called_once_with(ids)
        assert result == "decoded text" # Matches mock return value

    def test_get_vocab_size_initialized(self, mock_sp_processor):
        """Test getting vocab size after initialization/loading."""
        tokenizer = self._setup_initialized_tokenizer(mock_sp_processor)
        mock_sp_processor.get_piece_size = MagicMock(return_value=555)
        assert tokenizer.get_vocab_size() == 555 # From mock processor

    def test_save(self, mock_sp_processor, tmp_path):
        """Test saving the tokenizer config."""
        # Initialize tokenizer with specific args
        tokenizer = SentencePieceTokenizer(
            vocab_size=1000, model_prefix="sp_save_test",
            character_coverage=0.99, pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
        tokenizer.sp = mock_sp_processor # Assign mock processor to allow get_vocab_size
        # Mock get_piece_size to return a potentially different size after loading/training
        mock_sp_processor.get_piece_size.return_value = 1001

        output_dir = tmp_path / "sp_save_test_output"

        # Mock os.makedirs, open, and json.dump
        with patch("os.makedirs") as mock_makedirs, \
             patch("builtins.open", mock_open()) as mock_file, \
             patch("json.dump") as mock_json_dump:

            tokenizer.save(str(output_dir))

            # Check directory creation was attempted
            mock_makedirs.assert_called_once_with(str(output_dir), exist_ok=True)

            # Check if config file was opened for writing
            config_save_path = str(output_dir / "tokenizer_config.json")
            mock_file.assert_called_once_with(config_save_path, "w", encoding="utf-8")

            # Construct the expected dictionary based on the save() method logic
            expected_config_to_save = {
                'vocab_size': 1001, # From mock_sp_processor.get_piece_size()
                'model_prefix': "sp_save_test", # From _initial_model_prefix
                'character_coverage': 0.99, # From init
                'pad_token': "<pad>", # Default
                'unk_token': "<unk>", # Default
                'bos_token': "<s>", # Default
                'eos_token': "</s>", # Default
                'pad_id': 0, # From init
                'unk_id': 1, # From init
                'bos_id': 2, # From init
                'eos_id': 3, # From init
                '_target_': 'craft.data.tokenizers.sentencepiece.SentencePieceTokenizer'
            }

            # Check json.dump was called with the correct dictionary
            mock_json_dump.assert_called_once_with(expected_config_to_save, mock_file(), indent=4)

    def test_save_logs_error(self, caplog):
        """Test that save logs an error if json.dump fails."""
        tokenizer = SentencePieceTokenizer(vocab_size=10, model_prefix="test")
        output_dir = "dummy_dir"

        with patch("os.makedirs"), \
             patch("builtins.open", mock_open()), \
             patch("json.dump", side_effect=TypeError("Cannot serialize")) as mock_json_dump:
            with caplog.at_level(logging.ERROR):
                 tokenizer.save(output_dir)

            mock_json_dump.assert_called_once()
            assert "Failed to save tokenizer config" in caplog.text
            assert "Cannot serialize" in caplog.text
