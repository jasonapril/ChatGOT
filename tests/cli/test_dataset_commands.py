import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import os
import pickle
import shutil
from unittest.mock import patch
import json
import numpy as np

# Import the Typer app that contains the command to test
from craft.cli.dataset_commands import dataset_app
from craft.cli.run import app

# Create a runner instance
runner = CliRunner()

# --- Test Fixtures ---

@pytest.fixture(scope="function")
def temp_data_dir():
    """Creates a temporary directory for test input/output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(scope="function")
def sample_raw_file(temp_data_dir):
    """Creates a sample raw input text file."""
    input_path = temp_data_dir / "raw_input.txt"
    content = "This is line one.\nThis is line two.\nAnd the third line.\n" * 10
    input_path.write_text(content, encoding='utf-8')
    yield input_path

@pytest.fixture(scope="function")
def dummy_tokenizer_file(temp_data_dir):
    """Creates a dummy tokenizer file (e.g., for SentencePiece)."""
    tokenizer_path_prefix = temp_data_dir / "dummy_spm"
    model_file = tokenizer_path_prefix.with_suffix(".model")
    model_file.touch()
    yield tokenizer_path_prefix

@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory structure for testing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    tokenizers_dir = tmp_path / "tokenizers"
    raw_dir.mkdir()
    processed_dir.mkdir()
    tokenizers_dir.mkdir()

    # Create a dummy input file
    (raw_dir / "input.txt").write_text("This is a test file.\nRepeated content helps ensure vocab is small.\n" * 5, encoding='utf-8')

    return tmp_path

@pytest.fixture
def dummy_spm_tokenizer(test_data_dir: Path) -> Path:
    """Train a dummy SentencePiece tokenizer for testing prepare command."""
    input_file = test_data_dir / "raw" / "input.txt"
    tokenizer_output_dir = test_data_dir / "tokenizers"
    model_prefix = "test_spm"
    vocab_size = 50 # Small vocab for test

    result = runner.invoke(app, [
        "data",
        "train-tokenizer",
        "--input-files", str(input_file),
        "--output-dir", str(tokenizer_output_dir),
        "--vocab-size", str(vocab_size),
        "--model-prefix", model_prefix,
        "--model-type", "bpe", # Use bpe for example
    ])

    assert result.exit_code == 0, f"Tokenizer training failed: {result.stdout}"
    model_path_prefix = tokenizer_output_dir / model_prefix
    assert (model_path_prefix.with_suffix(".model")).exists()
    assert (model_path_prefix.with_suffix(".vocab")).exists()
    return model_path_prefix # Return the prefix path

# --- Test Functions --- #

def test_prepare_char_success(test_data_dir: Path):
    """Test successful dataset preparation with type='char'."""
    input_file = test_data_dir / "raw" / "input.txt"
    output_dir = test_data_dir / "processed" / "char_output"

    result = runner.invoke(app, [
        "data",
        "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "char",
        # Using default splits (0.9, 0.05, 0.05)
    ])

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}: {result.stdout}"
    assert output_dir.exists()

    # Check for expected files
    train_pkl = output_dir / "train.pkl"
    val_pkl = output_dir / "val.pkl"
    test_pkl = output_dir / "test.pkl"
    metadata_json = output_dir / "metadata.json"
    tokenizer_dir = output_dir / "tokenizer"

    assert train_pkl.exists()
    assert val_pkl.exists()
    assert test_pkl.exists()
    assert metadata_json.exists()
    assert tokenizer_dir.is_dir()
    assert (tokenizer_dir / "tokenizer_config.json").exists() # CharTokenizer saves config

    # Check content of metadata.json (basic checks)
    with open(metadata_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    assert metadata["data_format"] == "character"
    assert metadata["tokenizer_type"] == "<class 'craft.data.tokenizers.char.CharTokenizer'>"
    assert isinstance(metadata["vocab_size"], int)
    assert len(metadata["split_ratios"]) == 3
    assert "train" in metadata["split_sizes"]

    # Check content of pkl files (basic checks)
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    assert isinstance(train_data, np.ndarray)
    assert train_data.dtype == np.uint16 # As saved by char_processor
    assert len(train_data) > 0

    with open(val_pkl, 'rb') as f:
        val_data = pickle.load(f)
    assert isinstance(val_data, np.ndarray)

    with open(test_pkl, 'rb') as f:
        test_data = pickle.load(f)
    assert isinstance(test_data, np.ndarray)

    # Rough check of split sizes based on default 0.9, 0.05, 0.05
    total_tokens = metadata["total_tokens"]
    assert abs(len(train_data) / total_tokens - 0.9) < 0.01
    assert abs(len(val_data) / total_tokens - 0.05) < 0.01
    assert abs(len(test_data) / total_tokens - 0.05) < 0.01

def test_prepare_command_subword_success(sample_raw_file, dummy_tokenizer_file, temp_data_dir):
    """Test the 'prepare' command with type='subword' successfully."""
    input_path = sample_raw_file
    output_dir = temp_data_dir / "subword_output"
    tokenizer_path = dummy_tokenizer_file
    split_ratios = "0.8,0.1,0.1"

    # Mock the SentencePieceTokenizer methods within the command scope
    # Use nested with statements for clarity or ensure correct multi-patch syntax
    with patch('craft.cli.dataset_commands.SentencePieceTokenizer.load_from_prefix') as mock_load:
        with patch('craft.cli.dataset_commands.SentencePieceTokenizer.encode') as mock_encode:
            # Configure mocks
            mock_tokenizer_instance = mock_load.return_value
            mock_tokenizer_instance.get_vocab_size.return_value = 100
            mock_encode.return_value = list(range(50))

            result = runner.invoke(
                dataset_app, 
                [
                    "prepare", 
                    "--input-path", str(input_path), 
                    "--output-dir", str(output_dir),
                    "--type", "subword",
                    "--tokenizer-path", str(tokenizer_path),
                    "--split-ratios", split_ratios
                ],
                catch_exceptions=False
            )
        
    print(f"CLI Output (subword success):\n{result.stdout}")
    assert result.exit_code == 0
    assert "Dataset preparation complete" in result.stdout
    mock_load.assert_called_once_with(str(tokenizer_path))
    mock_encode.assert_called_once()

    assert (output_dir / "train.pkl").exists()
    assert (output_dir / "val.pkl").exists()
    assert (output_dir / "test.pkl").exists()

    with open(output_dir / "train.pkl", "rb") as f: train_data = pickle.load(f)
    with open(output_dir / "val.pkl", "rb") as f: val_data = pickle.load(f)
    with open(output_dir / "test.pkl", "rb") as f: test_data = pickle.load(f)
    
    total_tokens = 50
    assert len(train_data) == pytest.approx(total_tokens * 0.8, abs=1)
    assert len(val_data) == pytest.approx(total_tokens * 0.1, abs=1)
    assert len(test_data) == pytest.approx(total_tokens * 0.1, abs=1)

def test_prepare_command_missing_tokenizer_for_subword(sample_raw_file, temp_data_dir):
    """Test failure when --tokenizer-path is missing for type='subword'."""
    input_path = sample_raw_file
    output_dir = temp_data_dir / "subword_fail"

    result = runner.invoke(
        dataset_app, 
        [
            "prepare", 
            "--input-path", str(input_path), 
            "--output-dir", str(output_dir),
            "--type", "subword",
            "--split-ratios", "0.8,0.1,0.1"
        ]
    )
    
    assert result.exit_code != 0
    assert "tokenizer-path is required" in result.stdout

def test_prepare_command_invalid_splits(sample_raw_file, temp_data_dir):
    """Test failure with invalid split ratios."""
    input_path = sample_raw_file
    output_dir = temp_data_dir / "char_fail"

    result = runner.invoke(
        dataset_app, 
        [
            "prepare", 
            "--input-path", str(input_path), 
            "--output-dir", str(output_dir),
            "--type", "char",
            "--split-ratios", "0.8,0.3,0.1" # Sum > 1
        ]
    )
    
    assert result.exit_code != 0
    assert "Invalid split ratios" in result.stdout

def test_prepare_command_force_option(sample_raw_file, temp_data_dir):
    """Test that the --force option cleans the directory (mocking shutil)."""
    input_path = sample_raw_file
    output_dir = temp_data_dir / "force_test"
    output_dir.mkdir()
    # Create dummy files/dirs to be deleted
    (output_dir / "train.pkl").touch()
    (output_dir / "val.pkl").touch()
    (output_dir / "metadata.json").touch()
    (output_dir / "tokenizer").mkdir()
    (output_dir / "tokenizer" / "vocab.json").touch()

    # Patch the functions that actually perform deletion/processing
    # Use nested with for clarity
    with patch('craft.cli.dataset_commands.process_char_data') as mock_process:
        with patch('pathlib.Path.unlink') as mock_unlink:
            with patch('shutil.rmtree') as mock_rmtree:
                
                mock_process.return_value = {'train': '', 'val': '', 'test': ''}

                result = runner.invoke(
                    dataset_app, 
                    [
                        "prepare", 
                        "--input-path", str(input_path), 
                        "--output-dir", str(output_dir),
                        "--type", "char",
                        "--force"
                    ],
                    catch_exceptions=False
                )
        
    assert result.exit_code == 0
    assert "Cleaning up existing files/dirs" in result.stdout
    assert mock_unlink.call_count >= 3
    assert mock_rmtree.call_count == 1
    mock_process.assert_called_once()

def test_prepare_subword_success(test_data_dir: Path, dummy_spm_tokenizer: Path):
    """Test successful dataset preparation with type='subword'."""
    input_file = test_data_dir / "raw" / "input.txt"
    output_dir = test_data_dir / "processed" / "subword_output"
    tokenizer_path = dummy_spm_tokenizer # Path prefix from fixture
    splits = "0.8,0.1,0.1" # Use custom splits for testing this arg

    result = runner.invoke(app, [
        "data",
        "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "subword",
        "--tokenizer-path", str(tokenizer_path),
        "--split-ratios", splits,
    ])

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}: {result.stdout}"
    assert output_dir.exists()

    # Check for expected files
    train_pkl = output_dir / "train.pkl"
    val_pkl = output_dir / "val.pkl"
    test_pkl = output_dir / "test.pkl"
    metadata_json = output_dir / "metadata.json"

    assert train_pkl.exists()
    assert val_pkl.exists()
    assert test_pkl.exists()
    assert metadata_json.exists()

    # Check content of metadata.json (basic checks)
    with open(metadata_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    assert metadata["data_format"] == "subword"
    assert metadata["tokenizer_type"] == "SentencePiece"
    assert Path(metadata["tokenizer_model_path"]).exists() # Check path exists
    assert Path(metadata["tokenizer_model_path"]) == tokenizer_path.resolve().with_suffix(".model") # Check it points to the .model file
    assert isinstance(metadata["vocab_size"], int)
    assert metadata["split_ratios"] == [0.8, 0.1, 0.1]
    assert "train" in metadata["split_sizes"]

    # Check content of pkl files (basic checks)
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    assert isinstance(train_data, np.ndarray)
    assert train_data.dtype == np.int32 # As saved by subword logic
    assert len(train_data) > 0

    # Rough check of split sizes based on 0.8, 0.1, 0.1
    total_tokens = metadata["total_tokens"]
    assert total_tokens > 0 # Ensure we tokenized something
    with open(val_pkl, 'rb') as f: val_data = pickle.load(f)
    with open(test_pkl, 'rb') as f: test_data = pickle.load(f)
    assert abs(len(train_data) / total_tokens - 0.8) < 0.02 # Allow slightly more tolerance
    assert abs(len(val_data) / total_tokens - 0.1) < 0.02
    assert abs(len(test_data) / total_tokens - 0.1) < 0.02

def test_prepare_invalid_splits(test_data_dir: Path):
    """Test dataset preparation fails with invalid split ratios."""
    input_file = test_data_dir / "raw" / "input.txt"
    output_dir = test_data_dir / "processed" / "invalid_split_output"

    invalid_splits = [
        "0.8,0.1", # Too few
        "0.7,0.1,0.1,0.1", # Too many
        "0.8,0.3,0.1", # Sum > 1
        "0.5,0.1,0.1", # Sum < 1
        "abc,def,ghi" # Not numbers
    ]

    for splits in invalid_splits:
        result = runner.invoke(app, [
            "data",
            "prepare",
            "--input-path", str(input_file),
            "--output-dir", str(output_dir),
            "--type", "char", # Use char type for simplicity
            "--split-ratios", splits,
        ], catch_exceptions=False) # Catch exceptions to check exit code

        assert result.exit_code != 0, f"Expected failure for splits '{splits}' but got exit code 0"
        # Check for specific error message if possible/consistent
        assert "Invalid split ratios" in result.stdout or "value is not a valid float" in result.stdout
        # Ensure output dir wasn't partially created or left messy (optional check)
        # assert not output_dir.exists() or len(list(output_dir.iterdir())) == 0

def test_prepare_force_flag(test_data_dir: Path):
    """Test the --force flag correctly cleans the output directory."""
    input_file = test_data_dir / "raw" / "input.txt"
    output_dir = test_data_dir / "processed" / "force_test_output"

    # 1. Run once successfully
    result1 = runner.invoke(app, [
        "data", "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "char",
    ])
    assert result1.exit_code == 0, "Initial run failed"
    assert (output_dir / "train.pkl").exists()

    # 2. Add an extra file
    extra_file = output_dir / "extra_file.txt"
    extra_file.write_text("This should be deleted by --force")
    assert extra_file.exists()

    # 3. Run again without --force (should fail implicitly or explicitly, depending on command logic)
    # Let's assume for now the command is designed to potentially overwrite or error
    # if output exists. We primarily test if --force *cleans*.

    # 4. Run again *with* --force
    result_force = runner.invoke(app, [
        "data", "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "char",
        "--force",
    ])
    assert result_force.exit_code == 0, f"Run with --force failed: {result_force.stdout}"

    # Check standard files were recreated
    assert (output_dir / "train.pkl").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "tokenizer" / "tokenizer_config.json").exists()

    # Crucially, check the extra file was deleted
    assert not extra_file.exists(), "Extra file was not deleted by --force"

def test_prepare_missing_input(test_data_dir: Path):
    """Test dataset preparation fails if input file is missing."""
    input_file = test_data_dir / "raw" / "nonexistent_input.txt"
    output_dir = test_data_dir / "processed" / "missing_input_output"

    result = runner.invoke(app, [
        "data", "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "char",
    ], catch_exceptions=False)

    assert result.exit_code != 0
    # Typer automatically handles 'exists=True' validation error
    assert "Invalid value for '--input-path'" in result.stdout
    assert "does not exist" in result.stdout
    assert not output_dir.exists()

def test_prepare_subword_missing_tokenizer(test_data_dir: Path):
    """Test dataset preparation fails for type='subword' if tokenizer path is missing or invalid."""
    input_file = test_data_dir / "raw" / "input.txt"
    output_dir = test_data_dir / "processed" / "missing_tokenizer_output"
    nonexistent_tokenizer = test_data_dir / "tokenizers" / "nonexistent_spm"

    # Test missing --tokenizer-path
    result_missing = runner.invoke(app, [
        "data", "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "subword",
        "--split-ratios", "0.9,0.05,0.05",
        # Missing --tokenizer-path
    ], catch_exceptions=False)
    assert result_missing.exit_code != 0
    assert "tokenizer-path is required when --type='subword'" in result_missing.stdout
    assert not output_dir.exists()

    # Test invalid/non-existent tokenizer path
    result_invalid = runner.invoke(app, [
        "data", "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "subword",
        "--tokenizer-path", str(nonexistent_tokenizer),
        "--split-ratios", "0.9,0.05,0.05",
    ], catch_exceptions=False)
    assert result_invalid.exit_code != 0
    # Check for loading error message (might vary based on SentencePiece)
    assert "Failed to load tokenizer" in result_invalid.stdout or "No such file" in result_invalid.stdout
    assert not output_dir.exists()

def test_prepare_invalid_type(test_data_dir: Path):
    """Test dataset preparation fails for invalid --type."""
    input_file = test_data_dir / "raw" / "input.txt"
    output_dir = test_data_dir / "processed" / "invalid_type_output"

    result = runner.invoke(app, [
        "data", "prepare",
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--type", "word", # Invalid type
    ], catch_exceptions=False)

    assert result.exit_code != 0
    # Check for error message related to invalid type
    assert "Invalid processing type specified" in result.stdout
    assert not output_dir.exists()

# --- TODO: Add more tests --- #
# (Potentially add tests for edge cases like empty input file if needed)