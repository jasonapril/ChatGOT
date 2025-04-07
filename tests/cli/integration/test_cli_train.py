# tests/cli/integration/test_cli_train.py
import pytest
import subprocess
import sys
import os
from pathlib import Path
import shutil
import json
import pickle
import numpy as np

# Add project root to sys.path to allow importing test assets relative to root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CRAFT_CLI_MODULE = "craft.cli.run" # Module path for python -m

# --- Helper Functions ---

def run_script(script_name: str, args: list[str]) -> subprocess.CompletedProcess:
    """Helper function to run a script using subprocess."""
    script_path = SCRIPTS_DIR / script_name
    command = [sys.executable, str(script_path)] + args
    print(f"\nRunning command: {' '.join(command)}") # For debugging
    # Ensure PROJECT_ROOT is in PYTHONPATH so imports like 'from craft... ' work
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{str(PROJECT_ROOT)}{os.pathsep}{pythonpath}"
    env["HYDRA_FULL_ERROR"] = "1" # Ensure full Hydra errors
    env["PYTHONIOENCODING"] = "utf-8" # Ensure consistent encoding

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False, # Don't automatically raise exception on non-zero exit
        cwd=PROJECT_ROOT, # Run from project root
        env=env,
        encoding='utf-8' # Specify encoding for text mode
    )
    if result.returncode != 0:
        print(f"Script Error Output ({script_name}):\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")
    return result

# NEW Helper for CLI commands
def run_craft_cli(args: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Helper function to run the craft CLI using 'python -m craft.cli.run'."""
    command = [sys.executable, "-m", CRAFT_CLI_MODULE] + args
    print(f"\nRunning command: {' '.join(command)}") # For debugging
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    # Ensure PROJECT_ROOT is in PYTHONPATH so imports like 'from craft...' work
    env['PYTHONPATH'] = f"{str(PROJECT_ROOT)}{os.pathsep}{pythonpath}"
    env["HYDRA_FULL_ERROR"] = "1" # Ensure full Hydra errors
    env["PYTHONIOENCODING"] = "utf-8" # Ensure consistent encoding

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False, # Don't automatically raise exception on non-zero exit
        cwd=cwd, # Run from specified CWD
        env=env,
        encoding='utf-8' # Specify encoding for text mode
    )
    if result.returncode != 0:
        print(f"CLI Command Error ({' '.join(args[:2])}):\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")
    return result

# --- Fixtures ---

# @pytest.fixture(scope="module")
# def minimal_train_config_path():
#     """Provides the path to the minimal training config file.""" # Replaced by overrides
#     config_path = PROJECT_ROOT / "tests" / "assets" / "configs" / "test_cli_train.yaml"
#     if not config_path.exists():
#         pytest.skip(f"Minimal train config not found at {config_path}")
#     return config_path

# @pytest.fixture
# def tmp_output_dir(tmp_path):
#     """Provides a temporary directory for training output.""" # Replaced by e2e_train_env
#     output_dir = tmp_path / "cli_train_output"
#     # Don't create it here, let the script/Hydra handle it
#     # But return the path for assertion checks
#     return output_dir

@pytest.fixture(scope="function")
def e2e_train_env(tmp_path):
    """Fixture to create a temporary environment for e2e training tests."""
    # Create directories
    input_dir = tmp_path / "input"
    tokenizer_dir = tmp_path / "tokenizers"
    processed_data_dir = tmp_path / "processed_data" # Directory for CLI output
    training_output_dir = tmp_path / "training_output"
    input_dir.mkdir()
    tokenizer_dir.mkdir()
    processed_data_dir.mkdir() # Create this directory now
    # training_output_dir will be created by hydra

    # Create a simple input text file
    input_file = input_dir / "sample_text.txt"
    input_file.write_text(
        """This is line one.\n"""
        """This is line two, repeated.\n"""
        """This is line two, repeated.\n"""
        """This is line three.\n"""
    )

    # REMOVE dummy pickle file creation
    # train_data = np.array([...], dtype=np.uint16)
    # val_data = np.array([...], dtype=np.uint16)
    # train_pkl_path = simple_pkl_dir / "train.pkl"
    # val_pkl_path = simple_pkl_dir / "val.pkl"
    # ... (pickle dump logic removed) ...

    yield {
        "tmp_path": tmp_path,
        "input_dir": input_dir,
        "input_file": input_file,
        "tokenizer_dir": tokenizer_dir,
        "processed_data_dir": processed_data_dir, # Pass the new path
        "training_output_dir": training_output_dir
    }
    # Teardown is handled automatically by tmp_path fixture

# --- Test Function ---

def test_cli_train_e2e_workflow(e2e_train_env):
    """Tests the end-to-end workflow: train tokenizer -> prepare data (CLI) -> train language."""

    # --- Phase 1a: Train Tokenizer ---
    tokenizer_model_prefix = "test_sp"
    tokenizer_output_path_prefix = e2e_train_env["tokenizer_dir"] / tokenizer_model_prefix
    train_tok_args = [
        "--tokenizer_type", "sentencepiece",
        "--input_files", str(e2e_train_env["input_file"]),
        "--output_dir", str(e2e_train_env["tokenizer_dir"]),
        "--vocab_size", "50",
        "--model_prefix", tokenizer_model_prefix
    ]
    train_tok_result = run_script("train_tokenizer.py", train_tok_args)
    assert train_tok_result.returncode == 0, "Tokenizer training script failed"
    tokenizer_model_file = tokenizer_output_path_prefix.parent / f"{tokenizer_model_prefix}.model"
    assert tokenizer_model_file.is_file()
    tokenizer_config_file = tokenizer_output_path_prefix.parent / "tokenizer_config.json"
    assert tokenizer_config_file.is_file()
    try:
        with open(tokenizer_config_file, 'r') as f:
            tok_config_data = json.load(f)
        actual_vocab_size = tok_config_data.get('vocab_size')
        assert actual_vocab_size is not None, "Could not read vocab_size from tokenizer_config.json"
        print(f"Actual vocab size from tokenizer: {actual_vocab_size}")
    except Exception as e:
        pytest.fail(f"Failed to read tokenizer config {tokenizer_config_file}: {e}")


    # --- Phase 1b: Prepare Data (using CLI command) ---
    processed_data_dir = e2e_train_env["processed_data_dir"]
    prepare_args = [
        "dataset", "prepare",
        "--input-file", str(e2e_train_env["input_file"]),
        "--output-dir", str(processed_data_dir),
        "--processing-type", "subword",
        "--tokenizer-path", str(tokenizer_model_file.parent), # Pass the *directory* containing the tokenizer model/config
        "--force", # Overwrite if needed
        # Add split ratios if needed, default is 90/10/0
        # "--split-ratios", "0.8,0.1,0.1"
    ]
    prepare_result = run_craft_cli(prepare_args)
    assert prepare_result.returncode == 0, f"CLI dataset prepare failed: {prepare_result.stderr}"
    # Check for output files
    train_pkl = processed_data_dir / "train.pkl"
    val_pkl = processed_data_dir / "val.pkl"
    meta_pkl = processed_data_dir / "meta.pkl" # Check for meta file too
    assert train_pkl.is_file(), f"train.pkl not found in {processed_data_dir}"
    assert val_pkl.is_file(), f"val.pkl not found in {processed_data_dir}"
    assert meta_pkl.is_file(), f"meta.pkl not found in {processed_data_dir}"
    print(f"Successfully prepared data using CLI in {processed_data_dir}")


    # --- Phase 3: Run Training (Minimal) ---
    training_output_dir = e2e_train_env["training_output_dir"]
    # simple_pkl_dir = e2e_train_env["simple_pkl_dir"] # REMOVED

    # Construct Hydra overrides for a minimal run
    # Assumes a 'conf/experiment/test_e2e_train.yaml' exists.
    # It uses 'conf/data/test_pickle_loader.yaml' which targets PickleDataset.
    overrides = [
        f"experiment=test_e2e_train",
        # Update data path to point to the *actual* processed data
        f"experiment.data.datasets.train.dataset_params.path={processed_data_dir}",
        f"experiment.data.datasets.val.dataset_params.path={processed_data_dir}",
        f"experiment.model.vocab_size={actual_vocab_size}", # Crucial override from trained tokenizer
        f"experiment.training.max_steps=2", # Run only 2 steps
        f"experiment.training.log_interval=1",
        f"experiment.training.eval_interval=0", # Disable eval for simplicity
        f"experiment.training.save_interval=1", # Save checkpoint quickly
        f"hydra.run.dir={training_output_dir}" # Explicitly set output dir
    ]

    train_args = ["train", "language"] + overrides
    train_process = run_craft_cli(train_args)

    # --- Assertions (Training Run) ---

    # 1. Check return code
    assert train_process.returncode == 0, f"CLI train script failed with exit code {train_process.returncode}\nSTDERR: {train_process.stderr[:1000]}"

    # 2. Check for output directory creation
    assert training_output_dir.is_dir(), f"Training output directory {training_output_dir} was not created."

    # 3. Check for expected output files
    log_files = list(training_output_dir.glob("*.log"))
    assert len(log_files) > 0, f"No log file found in {training_output_dir}"
    print(f"Found log file: {log_files[0]}")

    checkpoint_dir = training_output_dir / "checkpoints"
    assert checkpoint_dir.is_dir(), f"Checkpoints directory {checkpoint_dir} was not created."

    # Check for at least one checkpoint file (save_interval=1)
    ckpt_files = list(checkpoint_dir.glob("*.pt"))
    assert len(ckpt_files) > 0, f"No checkpoint .pt file found in {checkpoint_dir}"
    print(f"Found checkpoint file: {ckpt_files[0]}")

    # Check for marker file
    marker_files = list(checkpoint_dir.glob("*.pt.marker"))
    assert len(marker_files) > 0, f"No checkpoint .pt.marker file found in {checkpoint_dir}"
    print(f"Found marker file: {marker_files[0]}")

    print("\nE2E CLI train command executed successfully and created expected outputs.") 