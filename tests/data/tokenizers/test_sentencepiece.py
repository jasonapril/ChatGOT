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
        """Test successful initialization with only required args (vocab_size, model_prefix)."""
        tokenizer = SentencePieceTokenizer(vocab_size=1000, model_prefix="test_prefix")
        # Check the vocab_size attribute directly
        assert tokenizer.vocab_size == 1000
        assert tokenizer.model_prefix == "test_prefix"
        # Check defaults
        assert tokenizer.character_coverage == 0.9995
        assert tokenizer.pad_token == "<PAD>"
        assert tokenizer.unk_token == "<UNK>"
        assert tokenizer.bos_token == "<BOS>"
        assert tokenizer.eos_token == "<EOS>"
        # Check IDs - they might be None until model is trained/loaded
        assert tokenizer.pad_id is None
        assert tokenizer.unk_id is None
        assert tokenizer.bos_id is None
        assert tokenizer.eos_id is None
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
        # Check defaults
        assert tokenizer.model_type == 'bpe'
        assert tokenizer.character_coverage == 0.9995
        # Check base token defaults
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.unk_token == "<unk>"
        # Defaults changed in SP Tokenizer definition
        assert tokenizer.bos_token == "<bos>"
        assert tokenizer.eos_token == "<eos>"

    # --- Tests for methods requiring mocks --- #
    # These will use patching

    @patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', autospec=True)
    @patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceTrainer', autospec=True)
    @patch('os.path.exists', return_value=True) # Mock exists for loading after train
    def test_train(self, mock_exists, mock_sp_trainer_cls, mock_sp_processor_cls, tmp_path, mock_sp_processor):
        """Test training the tokenizer."""
        # Setup
        input_files = [str(tmp_path / "input1.txt"), str(tmp_path / "input2.txt")]
        for f in input_files:
            Path(f).touch()
        input_files_str = ",".join(input_files)
        output_dir = str(tmp_path / "sp_output")

        tokenizer = SentencePieceTokenizer(
            vocab_size=1000,
            model_prefix="sp_train_test", # Prefix used for saving
            character_coverage=0.98,
            pad_piece="<PAD>", unk_piece="<UNK>", bos_piece="<BOS>", eos_piece="<EOS>"
        )

        # Mock SentencePieceProcessor that gets created after training
        mock_processor_instance = MagicMock()
        mock_processor_instance.load.side_effect = lambda path: None
        mock_processor_instance.get_piece_size.return_value = 1000 # Vocab size after load
        mock_sp_processor_cls.return_value = mock_processor_instance

        # Mock SentencePieceTrainer.Train method
        mock_train_method = MagicMock()

        # Patch the Train method directly
        with patch('sentencepiece.SentencePieceTrainer.Train', new=mock_train_method):
            tokenizer.train(input_files_str, output_dir)

            # 1. Check that SentencePieceTrainer.Train was called correctly
            expected_model_save_prefix = os.path.join(output_dir, "sp_train_test")
            cmd_args = [
                f"--input={input_files_str}",
                f"--model_prefix={expected_model_save_prefix}",
                f"--vocab_size=1000",
                f"--model_type=bpe", # Default model_type
                f"--character_coverage=0.98",
            ]
            # Add special tokens if defined
            special_token_map = {
                'pad_piece': "<PAD>", 'unk_piece': "<UNK>", 'bos_piece': "<BOS>", 'eos_piece': "<EOS>"
            }
            for key, value in special_token_map.items():
                 if value is not None:
                    cmd_args.append(f"--{key}={value}")
            expected_cmd_str = " ".join(cmd_args)

            # SentencePieceTrainer.Train expects a single string argument
            mock_train_method.assert_called_once_with(expected_cmd_str)

            # 2. Check that the processor was loaded after training
            mock_sp_processor_cls.assert_called_once() # Called once inside train
            mock_processor_instance.load.assert_called_once_with(f"{expected_model_save_prefix}.model")

            # 3. Check internal state updated
            assert tokenizer.sp == mock_processor_instance
            assert tokenizer._loaded_model_path == expected_model_save_prefix # Path should be prefix

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
        # Mock the processor's piece_to_id method to simulate loading
        mock_loaded_ids = {"<PAD>": 1, "<UNK>": 2, "<BOS>": 3, "<EOS>": 4}
        def mock_piece_to_id(piece):
            return mock_loaded_ids.get(piece, 999) # Return mock ID or default
        mock_sp_processor.piece_to_id = MagicMock(side_effect=mock_piece_to_id)
        # Mock vocab size
        mock_sp_processor.get_piece_size.return_value = 50 # Example loaded size

        mock_sp_processor_cls.return_value = mock_sp_processor
        tokenizer = SentencePieceTokenizer(vocab_size=500, model_prefix="irrelevant") # Init doesn't matter for load

        # --- Setup Save Directory (mimic saving process) --- #
        model_dir = tmp_path / "existing_model"
        model_dir.mkdir()
        model_prefix_name = "my_sp_model"
        model_file_path = model_dir / f"{model_prefix_name}.model"
        model_file_path.touch() # Need the .model file to exist

        # Create a dummy tokenizer_config.json, as load currently checks for it
        # (Ideally, load shouldn't *require* it if .model exists)
        config_data = {
            "model_type": "sentencepiece",
            "model_prefix": model_prefix_name,
            # Store some potentially outdated info
            "vocab_size": 40,
            "special_tokens": {
                "pad_token": "<PAD>", "pad_id": 99,
                "unk_token": "<UNK>", "unk_id": 98,
                "bos_token": "<BOS>", "bos_id": 97,
                "eos_token": "<EOS>", "eos_id": 96
            }
        }
        config_file_path = model_dir / f"{model_prefix_name}.json"
        with open(config_file_path, 'w') as f:
            json.dump(config_data, f)
        # --------------------------------------------------- #

        model_path_prefix_arg = str(model_dir / model_prefix_name)
        tokenizer.load(model_path_prefix_arg)

        # Verify exists check, processor instantiation, and load call
        mock_exists.assert_called_once_with(str(model_file_path))
        mock_sp_processor_cls.assert_called_once()
        mock_sp_processor.load.assert_called_once_with(str(model_file_path))

        # Verify internal state
        assert tokenizer.sp == mock_sp_processor
        assert tokenizer._loaded_model_path == model_path_prefix_arg # Check internal path store

        # Assert that internal IDs are updated from the loaded model (mocked piece_to_id)
        assert tokenizer.pad_id == mock_loaded_ids["<PAD>"]
        assert tokenizer.unk_id == mock_loaded_ids["<UNK>"]
        assert tokenizer.bos_id == mock_loaded_ids["<BOS>"]
        assert tokenizer.eos_id == mock_loaded_ids["<EOS>"]

        # Assert vocab size is updated from the loaded processor
        assert tokenizer.get_vocab_size() == 50

    @patch('os.path.exists', return_value=False)
    def test_load_file_not_found_raises(self, mock_exists, tmp_path):
        """Test load raises FileNotFoundError if metadata file is missing."""
        model_prefix = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError) as excinfo:
            SentencePieceTokenizer.load(model_prefix)
        # Expect error about metadata file (.json)
        assert f"SentencePiece metadata file not found: {model_prefix}.json" in str(excinfo.value)

    def test_methods_raise_if_not_initialized(self):
        """Test encode, decode raise RuntimeError if model not loaded/trained."""
        # Initialize without model_path, don't call train/load
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_prefix="test")

        assert tokenizer.sp is None
        with pytest.raises(RuntimeError) as excinfo:
            tokenizer.decode([1, 2])
        # Update regex to match the actual error message
        assert re.search(r'SentencePiece model not loaded', str(excinfo.value))

        # get_vocab_size should work even if not initialized
        assert tokenizer.get_vocab_size() == 100

    def test_get_vocab_size_not_initialized(self):
        """Test get_vocab_size returns configured size if not loaded/trained."""
        tokenizer = SentencePieceTokenizer(vocab_size=123, model_prefix="p")
        assert tokenizer.get_vocab_size() == 123
        # Test when vocab_size is None initially
        tokenizer_no_size = SentencePieceTokenizer(model_prefix="p2")
        assert tokenizer_no_size.get_vocab_size() == 0 # Should return 0 if None

    def test_get_vocab_size_initialized(self, mock_sp_processor):
        """Test get_vocab_size returns size from loaded model."""
        tokenizer = self._setup_initialized_tokenizer(mock_sp_processor)
        mock_sp_processor.get_piece_size = MagicMock(return_value=555)
        assert tokenizer.get_vocab_size() == 555 # From mock processor

    def test_save(self, mock_sp_processor, tmp_path):
        """Test saving the tokenizer config."""
        # Initialize tokenizer with specific args
        tokenizer = SentencePieceTokenizer(
            vocab_size=1000,
            model_prefix="sp_save_test",
            model_type='unigram', # Use a non-default type
            character_coverage=0.99,
            pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            extra_config_arg="hello" # Test extra kwargs
        )
        tokenizer.sp = mock_sp_processor # Assign mock processor
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
            # Should match the structure of tokenizer.config populated in __init__ and updated
            expected_config_to_save = {
                'vocab_size': 1001, # Updated from get_vocab_size()
                'model_prefix': "sp_save_test",
                'model_type': 'unigram', # From init
                'character_coverage': 0.99,
                'special_tokens': {}, # Base tokenizer doesn't store these separately in config
                'pad_token': "[PAD]",
                'unk_token': "[UNK]",
                'bos_token': "[BOS]",
                'eos_token': "[EOS]",
                'pad_id': 0,
                'unk_id': 1,
                'bos_id': 2,
                'eos_id': 3,
                'extra_config_arg': "hello", # Check extra kwarg saved
                '_target_': 'craft.data.tokenizers.sentencepiece.SentencePieceTokenizer'
            }

            # Check json.dump was called with the correct dictionary
            # Need to access the first argument of the call
            args, kwargs = mock_json_dump.call_args
            actual_saved_dict = args[0]
            assert actual_saved_dict == expected_config_to_save
            # Check other json.dump args
            assert args[1] == mock_file()
            assert kwargs == {'indent': 4}

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

    def test_init_with_model_path(self, mock_sp_processor_cls):
        """Test initialization with model_path attempts to load the model."""
        mock_processor = MagicMock()
        mock_processor.get_piece_size.return_value = 2000 # Mock loaded vocab size
        mock_sp_processor_cls.return_value = mock_processor

        model_path = "/fake/path/to/model" # Prefix, .model is appended internally
        with patch('os.path.exists', return_value=True):
            tokenizer = SentencePieceTokenizer(model_path=model_path)

        mock_sp_processor_cls.assert_called_once()
        mock_processor.load.assert_called_once_with(f"{model_path}.model")
        # Verify vocab size was updated from loaded model
        assert tokenizer.vocab_size == 2000
        assert tokenizer.sp == mock_processor
        assert tokenizer._loaded_model_path == model_path

    def test_init_load_model_file_not_found(self, mock_sp_processor_cls):
        """Test initialization logs error but doesn't fail if model_path file not found."""
        model_path = "/fake/path/nonexistent_model"
        # Ensure vocab_size is provided, otherwise init might warn/fail anyway
        with patch('os.path.exists', return_value=False), \
             patch.object(logging.getLogger('craft.data.tokenizers.sentencepiece'), 'error') as mock_log_error:
            tokenizer = SentencePieceTokenizer(model_path=model_path, vocab_size=500)

        # Processor should not have been called as exists returns False
        mock_sp_processor_cls.assert_not_called()
        mock_log_error.assert_called_once_with(f"Failed to load model during __init__ from path: {model_path}", exc_info=True)
        assert tokenizer.sp is None
        assert tokenizer.vocab_size == 500 # Should retain configured vocab size

    def test_init_uses_defaults(self):
        """Test initialization uses defaults for optional args."""
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_prefix="pref")
        assert tokenizer.model_type == 'bpe'
        assert tokenizer.character_coverage == 0.9995
        assert tokenizer.unk_token == "<unk>" # From base default
        assert tokenizer.bos_token == "<bos>"
        assert tokenizer.eos_token == "<eos>"
        assert tokenizer.pad_token == "<pad>"

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
            vocab_size=1000,
            model_prefix="sp_save_test",
            model_type='unigram', # Use a non-default type
            character_coverage=0.99,
            pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            extra_config_arg="hello" # Test extra kwargs
        )
        tokenizer.sp = mock_sp_processor # Assign mock processor
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
            # Should match the structure of tokenizer.config populated in __init__ and updated
            expected_config_to_save = {
                'vocab_size': 1001, # Updated from get_vocab_size()
                'model_prefix': "sp_save_test",
                'model_type': 'unigram', # From init
                'character_coverage': 0.99,
                'special_tokens': {}, # Base tokenizer doesn't store these separately in config
                'pad_token': "[PAD]",
                'unk_token': "[UNK]",
                'bos_token': "[BOS]",
                'eos_token': "[EOS]",
                'pad_id': 0,
                'unk_id': 1,
                'bos_id': 2,
                'eos_id': 3,
                'extra_config_arg': "hello", # Check extra kwarg saved
                '_target_': 'craft.data.tokenizers.sentencepiece.SentencePieceTokenizer'
            }

            # Check json.dump was called with the correct dictionary
            # Need to access the first argument of the call
            args, kwargs = mock_json_dump.call_args
            actual_saved_dict = args[0]
            assert actual_saved_dict == expected_config_to_save
            # Check other json.dump args
            assert args[1] == mock_file()
            assert kwargs == {'indent': 4}

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

    # Helper method to set up an initialized tokenizer with mocks for testing methods
    def _setup_initialized_tokenizer(self, mock_processor, **kwargs):
        # Initialize with some basic args, they don't matter if sp is mocked
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_prefix="dummy", **kwargs)
        tokenizer.sp = mock_processor # Manually assign the mock processor
        tokenizer._loaded_model_path = "dummy/path/prefix" # Set dummy internal path
        return tokenizer

    @patch('sentencepiece.SentencePieceProcessor')
    def test_load_success(self, mock_sp_processor_cls, tmp_path):
        # Ensure the correct files exist
        mock_metadata_path = tmp_path / "load_success_test" / "sp_load_test.json"
        mock_model_path = tmp_path / "load_success_test" / "sp_load_test.model"
        mock_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        # Save mock config data to the JSON file
        mock_config_data = {
            "model_type": "unigram",
            "vocab_size": 1500,
            "model_prefix": "sp_load_test",
            "special_tokens": {"unk_token": "[UNK]", "unk_id": 1}
        }
        with open(mock_metadata_path, 'w') as f:
            json.dump(mock_config_data, f)
        # Create a dummy model file
        mock_model_path.touch()

        # Mock the processor
        mock_processor = MagicMock()
        mock_processor.get_piece_size.return_value = 1600 # Simulate loaded size
        mock_sp_processor_cls.return_value = mock_processor

        # Load the tokenizer
        model_prefix_to_load = str(tmp_path / "load_success_test" / "sp_load_test")
        tokenizer = SentencePieceTokenizer.load(model_prefix_to_load)

        # Assertions
        mock_processor.load.assert_called_once_with(str(mock_model_path))
        assert tokenizer.sp == mock_processor
        assert tokenizer.get_vocab_size() == 1600 # Should use loaded model size
        assert tokenizer.model_type == "unigram" # From loaded config
        assert tokenizer.unk_token == "[UNK]" # From loaded config
        assert tokenizer.unk_id == 1 # From loaded config
        # Check that the path is stored internally
        assert tokenizer._loaded_model_path == str(mock_model_path)

    @patch('sentencepiece.SentencePieceProcessor')
    def test_load_failure(self, mock_sp_processor_cls, tmp_path):
        model_prefix = str(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError) as excinfo:
            SentencePieceTokenizer.load(model_prefix)
        # Expect error about metadata file (.json)
        assert f"SentencePiece metadata file not found: {model_prefix}.json" in str(excinfo.value)

    def test_methods_raise_if_not_initialized(self):
        # Initialize without model_path, don't call train/load
        tokenizer = SentencePieceTokenizer(vocab_size=100, model_prefix="test")

        assert tokenizer.sp is None
        with pytest.raises(RuntimeError) as excinfo:
            tokenizer.decode([1, 2])
        # Update regex to match the actual error message
        assert re.search(r'SentencePiece model not loaded', str(excinfo.value))

        # get_vocab_size should work even if not initialized
        assert tokenizer.get_vocab_size() == 100
