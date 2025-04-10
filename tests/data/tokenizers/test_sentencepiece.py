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
from craft.data.tokenizers.sentencepiece import SentencePieceTokenizer, METADATA_FILENAME # Import METADATA_FILENAME

# --- Fixtures ---

@pytest.fixture
def mock_sp_processor():
    """Provides a MagicMock for the sentencepiece.SentencePieceProcessor."""
    mock = MagicMock(name="SentencePieceProcessorInstance")
    mock.load.return_value = None
    mock.encode.return_value = [10, 20, 30] # Example IDs for encode (non-batch)
    mock.encode_as_ids = MagicMock(return_value=[10, 20, 30]) # Explicit mock for encode_as_ids
    mock.Encode = MagicMock(return_value=[[10, 20, 30]]) # For batch encode?
    mock.decode_ids.return_value = "decoded text" # Example text
    mock.decode = MagicMock(return_value="decoded text") # Add mock for decode method
    mock.get_piece_size.return_value = 1000 # Example vocab size
    mock.id_to_piece = MagicMock(side_effect=lambda id: f"<piece_{id}>") # Mock id_to_piece
    mock.piece_to_id = MagicMock(side_effect=lambda piece: {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}.get(piece, 99))
    mock.pad_id = MagicMock(return_value=0)
    mock.unk_id = MagicMock(return_value=1)
    mock.bos_id = MagicMock(return_value=2)
    mock.eos_id = MagicMock(return_value=3)
    return mock

@pytest.fixture
def sp_tokenizer_fixture(tmp_path, mock_sp_processor):
    """Fixture to create a valid SentencePieceTokenizer instance for testing methods."""
    model_dir = tmp_path / "sp_fixture_model"
    model_dir.mkdir()
    # Create dummy model file
    (model_dir / SentencePieceTokenizer.MODEL_FILENAME).touch()
    # Create dummy metadata
    metadata = {
        "vocab_size": mock_sp_processor.get_piece_size(),
        "special_tokens_map": {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<bos>",
            "eos": "<eos>"
        }
    }
    with open(model_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f)

    # Patch the SentencePieceProcessor class to return our mock instance
    with patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', return_value=mock_sp_processor):
        tokenizer = SentencePieceTokenizer(model_path=str(model_dir))
        yield tokenizer # Yield the tokenizer instance

@pytest.fixture
def setup_dummy_tokenizer_files(tmp_path):
    """Helper fixture to create dummy files for tests that need just the path."""
    model_dir = tmp_path / "dummy_sp_files"
    model_dir.mkdir()
    (model_dir / SentencePieceTokenizer.MODEL_FILENAME).touch()
    metadata = {"vocab_size": 50, "special_tokens_map": {}}
    with open(model_dir / METADATA_FILENAME, "w") as f:
        json.dump(metadata, f)
    return model_dir

# --- Test Class ---
# Using a class can help group tests and potentially share mocks if needed

class TestSentencePieceTokenizer:

    # Test init
    def test_init_success(self, tmp_path):
        """Test successful initialization requires model_path and loads."""
        model_dir = tmp_path / "init_success_model"
        model_dir.mkdir()
        # Create dummy model file
        (model_dir / SentencePieceTokenizer.MODEL_FILENAME).touch()
        # Create dummy metadata
        metadata = {
            "vocab_size": 1000,
            "special_tokens_map": {"pad": "<PAD>"}
        }
        with open(model_dir / METADATA_FILENAME, "w") as f:
            json.dump(metadata, f)

        # Mock the processor to avoid real loading issues
        mock_processor = MagicMock()
        mock_processor.load.return_value = None
        mock_processor.get_piece_size.return_value = 1000
        mock_processor.pad_id.return_value = 0 # Example ID

        # Patch the SentencePieceProcessor class to return the mock
        with patch('craft.data.tokenizers.sentencepiece.spm.SentencePieceProcessor', return_value=mock_processor) as mock_proc_cls:
            tokenizer = SentencePieceTokenizer(model_path=str(model_dir))

            # Assertions
            mock_proc_cls.assert_called_once() # Check processor was instantiated
            mock_processor.load.assert_called_once_with(str(model_dir / SentencePieceTokenizer.MODEL_FILENAME)) # Check load called
            assert tokenizer.model_dir == model_dir
            assert tokenizer.sp_model == mock_processor
            assert tokenizer.vocab_size == 1000
            assert tokenizer.pad_token_id == 0

    def test_init_load_model_dir_not_found(self, tmp_path):
        """Test initialization raises FileNotFoundError if model directory doesn't exist."""
        non_existent_dir = tmp_path / "non_existent_dir_init"
        with pytest.raises(FileNotFoundError, match=f"SentencePiece model directory not found: {non_existent_dir}"):
            SentencePieceTokenizer(model_path=str(non_existent_dir))

    def test_init_load_model_file_not_found(self, tmp_path):
        """Test initialization raises FileNotFoundError if .model file is missing."""
        model_dir = tmp_path / "missing_model_file"
        model_dir.mkdir()
        # Missing .model file
        metadata = {"vocab_size": 50}
        with open(model_dir / METADATA_FILENAME, "w") as f:
            json.dump(metadata, f)
        with pytest.raises(FileNotFoundError, match=f"SentencePiece model file not found: .*{SentencePieceTokenizer.MODEL_FILENAME}"):
            SentencePieceTokenizer(model_path=str(model_dir))

    def test_init_load_metadata_file_not_found(self, tmp_path):
        """Test initialization raises FileNotFoundError if metadata file is missing."""
        model_dir = tmp_path / "missing_metadata_file"
        model_dir.mkdir()
        (model_dir / SentencePieceTokenizer.MODEL_FILENAME).touch()
        # Missing metadata file
        with pytest.raises(FileNotFoundError, match=f"SentencePiece metadata file not found: .*{METADATA_FILENAME}"):
            SentencePieceTokenizer(model_path=str(model_dir))

    # Test methods using the fixture
    def test_encode(self, sp_tokenizer_fixture, mock_sp_processor):
        """Test encode method calls the underlying processor."""
        text_to_encode = "encode this"
        sp_tokenizer_fixture.encode(text_to_encode)
        mock_sp_processor.encode.assert_called_once_with(text_to_encode)

    def test_decode(self, sp_tokenizer_fixture, mock_sp_processor):
        """Test decode method calls the underlying processor."""
        ids_to_decode = [10, 20, 30]
        sp_tokenizer_fixture.decode(ids_to_decode)
        mock_sp_processor.decode.assert_called_once_with(ids_to_decode)

    def test_get_vocab_size_initialized(self, sp_tokenizer_fixture, mock_sp_processor):
        """Test get_vocab_size returns the value from the loaded processor."""
        assert sp_tokenizer_fixture.get_vocab_size() == mock_sp_processor.get_piece_size()

    def test_save_not_implemented(self, sp_tokenizer_fixture: SentencePieceTokenizer):
        """Test that save method is not implemented (use trainer)."""
        with pytest.raises(NotImplementedError):
            sp_tokenizer_fixture.save("some_dir") # TODO: Save to temp dir

    def test_train_not_implemented(self, sp_tokenizer_fixture: SentencePieceTokenizer):
        """Test that train method is not implemented (use trainer)."""
        with pytest.raises(NotImplementedError):
            sp_tokenizer_fixture.train("some_file", "some_dir") # TODO: Save to temp dir

    # Test load_from_prefix class method
    def test_load_from_prefix_success(self, setup_dummy_tokenizer_files):
        """Test the load_from_prefix class method successfully creates an instance."""
        model_dir = setup_dummy_tokenizer_files
        # Patch the processor load within the test
        with patch("sentencepiece.SentencePieceProcessor.load", return_value=None) as mock_load:
            tokenizer = SentencePieceTokenizer.load_from_prefix(str(model_dir))
            assert isinstance(tokenizer, SentencePieceTokenizer)
            assert tokenizer.model_dir == model_dir
            mock_load.assert_called_once()

    def test_load_from_prefix_dir_not_found(self, tmp_path):
        """Test load_from_prefix raises error if directory doesn't exist."""
        non_existent_dir = tmp_path / "non_existent_dir_for_load"
        with pytest.raises(FileNotFoundError):
            SentencePieceTokenizer.load_from_prefix(str(non_existent_dir))
