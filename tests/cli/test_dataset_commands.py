import pytest
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import os
import pickle
import shutil
from unittest.mock import patch

# Import the Typer app that contains the command to test
from craft.cli.dataset_commands import dataset_app

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

# --- Test Functions --- #

def test_prepare_command_char_success(sample_raw_file, temp_data_dir):
    """Test the 'prepare' command with type='char' successfully."""
    input_path = sample_raw_file
    output_dir = temp_data_dir / "char_output"
    
    # Patch the underlying function called by the command for char type
    with patch('craft.cli.dataset_commands.process_char_data') as mock_process:
        mock_process.return_value = { # Simulate successful return
            'train': str(output_dir / 'train.pkl'), 
            'val': str(output_dir / 'val.pkl'),
            'test': str(output_dir / 'test.pkl'),
        }
        
        result = runner.invoke(
            dataset_app, 
            [
                "prepare", 
                "--input-path", str(input_path), 
                "--output-dir", str(output_dir),
                "--type", "char",
            ],
            catch_exceptions=False
        )
        
    print(f"CLI Output (char success):\n{result.stdout}")
    assert result.exit_code == 0
    assert "Dataset preparation complete" in result.stdout
    mock_process.assert_called_once_with(
        input_path=str(input_path),
        output_dir=str(output_dir),
        splits=(0.9, 0.05, 0.05) # Verify default splits were passed
    )

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