# tests/cli/integration/test_cli_dataset.py
import pytest
import subprocess
import sys
import os
from pathlib import Path
import shutil
import yaml
import pickle
import numpy as np

# Add project root to sys.path to allow importing test assets relative to root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CRAFT_CLI_MODULE = "craft.cli.run" # Module path for python -m

# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_input_file(tmp_path_factory):
    """Creates a dummy input text file for data preparation."""
    # Use tmp_path_factory for session/module scope
    # tmp_path is function scoped
    input_dir = tmp_path_factory.mktemp("cli_dataset_input")
    input_file = input_dir / "sample.txt"
    # Simple text content
    input_file.write_text("Line one.\nLine two with chars.\nFinal line.")
    return input_file

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provides a temporary directory for dataset output for a single test."""
    # Use function-scoped tmp_path for isolation
    output_dir = tmp_path / "cli_dataset_output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture(scope="module")
def minimal_subword_config(tmp_path_factory):
    """Creates a minimal subword config YAML for testing."""
    config_dir = tmp_path_factory.mktemp("cli_subword_config")
    config_file = config_dir / "subword_test_config.yaml"
    config_content = {
        'data': {
            'type': 'text',
            'format': 'subword', # Specify subword format
            'tokenizer': {
                '_target_': 'craft.data.tokenizers.sentencepiece.SentencePieceTokenizer',
                'model_type': 'bpe', # Use BPE for simplicity
                'vocab_size': 100, # Small vocab for testing
                'special_tokens': {'unk': '<UNK>'}, # Define UNK
                # Other parameters can be default for testing
            },
            'val_split_ratio': 0.1, # Need some splits
            'test_split_ratio': 0.1,
            'random_seed': 42
        }
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    return config_file

# --- Test Functions ---

# Helper to run and decode subprocess output safely
def run_cli_command(command):
    print(f"\nRunning command: {' '.join(command)}")
    # Capture raw bytes
    process = subprocess.run(command, capture_output=True)

    # Decode safely
    stdout_str = process.stdout.decode('utf-8', errors='replace')
    stderr_str = process.stderr.decode('utf-8', errors='replace')

    print(f"--- Subprocess STDOUT ---\n{stdout_str}")
    print(f"--- Subprocess STDERR ---\n{stderr_str}")

    return process.returncode, stdout_str, stderr_str

@pytest.mark.integration
def test_cli_dataset_prepare_char(sample_input_file, tmp_output_dir):
    """Tests 'dataset prepare' for character tokenization (default behavior)."""
    # The CLI command now uses the output dir directly.
    # We test the default (char) processing by *not* providing --config.

    command = [
        sys.executable,
        "-m", CRAFT_CLI_MODULE,
        "dataset", "prepare",
        f"--input={sample_input_file}",
        f"--output-dir={tmp_output_dir}", # Use the base tmp dir
        "--force",
    ]

    returncode, _, stderr_str = run_cli_command(command)

    assert returncode == 0, f"CLI dataset prepare (char) failed with stderr: {stderr_str}"

    # Check output dir exists first
    assert tmp_output_dir.exists(), f"Output directory {tmp_output_dir} was not created."
    assert tmp_output_dir.is_dir(), f"Output path {tmp_output_dir} is not a directory."

    # Check for expected output files directly in tmp_output_dir
    assert (tmp_output_dir / "train.pkl").exists(), "train.pkl missing"
    assert (tmp_output_dir / "val.pkl").exists(), "val.pkl missing"
    assert (tmp_output_dir / "test.pkl").exists(), "test.pkl missing"
    assert (tmp_output_dir / "tokenizer").exists(), "tokenizer directory missing"
    assert (tmp_output_dir / "tokenizer" / "tokenizer_config.json").exists(), "tokenizer_config.json missing"
    assert (tmp_output_dir / "tokenizer" / "vocab.json").exists(), "vocab.json missing"
    assert (tmp_output_dir / "metadata.json").exists(), "metadata.json missing"

    # Basic content check (load train split)
    try:
        with open(tmp_output_dir / "train.pkl", "rb") as f:
            train_data = pickle.load(f)
        assert isinstance(train_data, np.ndarray), "Train data is not a numpy array"
        assert train_data.dtype == np.uint16 # Char processor uses uint16
    except Exception as e:
        pytest.fail(f"Failed to load or validate train.pkl: {e}")

@pytest.mark.integration
def test_cli_dataset_prepare_subword(sample_input_file, tmp_output_dir, minimal_subword_config):
    """Tests 'dataset prepare' for subword tokenization using a config file."""
    # The CLI command now uses the output dir directly.

    command = [
        sys.executable,
        "-m", CRAFT_CLI_MODULE,
        "dataset", "prepare",
        f"--input={sample_input_file}",
        f"--output-dir={tmp_output_dir}", # Use the base tmp dir
        f"--config={minimal_subword_config}",
        "--force",
    ]

    returncode, _, stderr_str = run_cli_command(command)

    assert returncode == 0, f"CLI dataset prepare (subword) failed with stderr: {stderr_str}"

    # Check output dir exists first
    assert tmp_output_dir.exists(), f"Output directory {tmp_output_dir} was not created."
    assert tmp_output_dir.is_dir(), f"Output path {tmp_output_dir} is not a directory."

    # Check for expected output files directly in tmp_output_dir
    assert (tmp_output_dir / "train.pkl").exists(), "train.pkl missing"
    assert (tmp_output_dir / "val.pkl").exists(), "val.pkl missing"
    # Test split might be empty and skipped, so don't assert its existence strictly
    # assert (tmp_output_dir / "test.pkl").exists()
    assert (tmp_output_dir / "tokenizer").exists(), "tokenizer directory missing"
    assert (tmp_output_dir / "tokenizer" / "tokenizer_config.json").exists(), "tokenizer_config.json missing"
    # Get model prefix from config to check correct model file name
    model_prefix = "subword_test" # From fixture subword_test_config.yaml
    assert (tmp_output_dir / "tokenizer" / f"{model_prefix}.model").exists(), f"{model_prefix}.model missing"
    assert (tmp_output_dir / "metadata.json").exists(), "metadata.json missing"

    # Basic content check (load train split)
    try:
        with open(tmp_output_dir / "train.pkl", "rb") as f:
            train_data = pickle.load(f)
        assert isinstance(train_data, np.ndarray), "Train data is not a numpy array"
        assert train_data.dtype == np.int64 # Subword processor uses int64
    except Exception as e:
        pytest.fail(f"Failed to load or validate train.pkl: {e}") 