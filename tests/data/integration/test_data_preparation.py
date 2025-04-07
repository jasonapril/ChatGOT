"""
Integration tests for the data preparation pipeline (scripts/train_tokenizer.py and scripts/prepare_data.py).
"""
import pytest
import subprocess
import sys
import os
import shutil
import pickle
import numpy as np
from pathlib import Path

# Get project root assuming tests are run from the root or within tests/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TEST_ASSETS_DIR = PROJECT_ROOT / "tests" / "assets" # Assuming a test assets directory exists

@pytest.fixture(scope="function")
def data_prep_env(tmp_path):
    """Fixture to create a temporary environment for data prep tests."""
    # Create directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    tokenizer_dir = tmp_path / "tokenizers"
    input_dir.mkdir()
    output_dir.mkdir()
    tokenizer_dir.mkdir()

    # Create a simple input text file
    input_file = input_dir / "sample_text.txt"
    input_file.write_text(
        """This is the first line.\n"""
        """This is the second line, with some repetition.\n"""
        """And a third line.\n"""
        """First line again.\n"""
        """Second line again.\n"""
    )

    yield {
        "tmp_path": tmp_path,
        "input_dir": input_dir,
        "input_file": input_file,
        "output_dir": output_dir,
        "tokenizer_dir": tokenizer_dir
    }
    # Teardown is handled automatically by tmp_path fixture

def run_script(script_name: str, args: list[str]) -> subprocess.CompletedProcess:
    """Helper function to run a script using subprocess."""
    script_path = SCRIPTS_DIR / script_name
    command = [sys.executable, str(script_path)] + args
    print(f"Running command: {' '.join(command)}") # For debugging
    # Ensure PROJECT_ROOT is in PYTHONPATH so imports like 'from craft... ' work
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{str(PROJECT_ROOT)}{os.pathsep}{pythonpath}"

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False, # Don't automatically raise exception on non-zero exit
        cwd=PROJECT_ROOT, # Run from project root
        env=env
    )
    if result.returncode != 0:
        print(f"Script Error Output:\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")
    return result

# TODO: Implement actual tests

def test_prepare_data_char_mode(data_prep_env):
    """Test the prepare_data.py script in character mode."""
    output_subdir = data_prep_env["output_dir"] / "char_output"
    output_subdir.mkdir()

    args = [
        "--type", "char",
        "--input-path", str(data_prep_env["input_file"]),
        "--output-dir", str(output_subdir),
        # Using default split ratios [0.9, 0.05, 0.05]
    ]

    result = run_script("prepare_data.py", args)

    assert result.returncode == 0, "Script execution failed"

    # --- Assertions ---
    # 1. Check for expected output files (names might depend on char_processor implementation)
    #    Example: Assuming it creates train.bin, val.bin, meta.pkl
    assert (output_subdir / "train.bin").is_file()
    assert (output_subdir / "val.bin").is_file()
    # assert (output_subdir / "test.bin").is_file() # Check if test split is created
    assert (output_subdir / "meta.pkl").is_file()

    # 2. (Optional) Load meta.pkl and check vocab size or content
    with open(output_subdir / "meta.pkl", 'rb') as f:
         meta = pickle.load(f)
    assert 'vocab_size' in meta
    assert 'itos' in meta
    assert 'stoi' in meta
    # Add more specific checks based on expected vocab from input_file

    # 3. (Optional) Check basic properties of bin files (e.g., not empty)
    assert (output_subdir / "train.bin").stat().st_size > 0
    assert (output_subdir / "val.bin").stat().st_size > 0


def test_prepare_data_subword_mode(data_prep_env):
    """Test the train_tokenizer.py and prepare_data.py scripts in subword mode."""
    # --- Step 1: Train Tokenizer ---
    tokenizer_output_subdir = data_prep_env["tokenizer_dir"] / "sp_test_model"
    # No need to mkdir, train_tokenizer should create it

    train_args = [
        "--tokenizer_type", "sentencepiece",
        "--input_files", str(data_prep_env["input_file"]),
        "--output_dir", str(tokenizer_output_subdir),
        "--vocab_size", "50", # Small vocab for testing
        "--model_prefix", "test_sp" # Prefix for .model and .vocab
    ]

    train_result = run_script("train_tokenizer.py", train_args)

    assert train_result.returncode == 0, "Tokenizer training script failed"

    # --- Assertions (Train Tokenizer) ---
    assert tokenizer_output_subdir.is_dir()
    assert (tokenizer_output_subdir / "test_sp.model").is_file()
    assert (tokenizer_output_subdir / "test_sp.vocab").is_file()
    # Check for config file as well (assuming train_tokenizer saves it)
    assert (tokenizer_output_subdir / "tokenizer_config.json").is_file()

    # --- Step 2: Prepare Data using trained tokenizer ---
    data_output_subdir = data_prep_env["output_dir"] / "subword_output"
    # No need to mkdir, prepare_data should create it based on args

    prepare_args = [
        "--type", "subword",
        "--input-path", str(data_prep_env["input_file"]),
        "--output-dir", str(data_output_subdir),
        "--tokenizer-path", str(tokenizer_output_subdir / "test_sp"), # Pass the model *prefix*
        # Using default split ratios [0.9, 0.05, 0.05]
    ]

    prepare_result = run_script("prepare_data.py", prepare_args)

    assert prepare_result.returncode == 0, "Data preparation script failed"

    # --- Assertions (Prepare Data) ---
    assert data_output_subdir.is_dir()
    assert (data_output_subdir / "train.pkl").is_file()
    assert (data_output_subdir / "val.pkl").is_file()
    assert (data_output_subdir / "test.pkl").is_file()

    # (Optional) Load pkl files and check basic properties
    for split in ["train", "val", "test"]:
        pkl_file = data_output_subdir / f"{split}.pkl"
        assert pkl_file.stat().st_size > 0 # Check not empty
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        assert isinstance(data, np.ndarray), f"{split}.pkl should contain a numpy array"
        assert np.issubdtype(data.dtype, np.integer), f"{split}.pkl array should have integer dtype"
        # Could add checks on token ID range based on vocab size if needed 