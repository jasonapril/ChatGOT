import pytest
import subprocess
import sys
import os
from pathlib import Path
import json # Added for reading vocab size

# Add project root to sys.path to allow importing src modules
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Assuming your CLI script is executable or run via python -m
# Adjust the path as necessary
CLI_SCRIPT_NAME = "run.py" # Adjust if your main script has a different name
CRAFT_CLI_MODULE = "craft.cli.run" # Module path for python -m

# --- Import helpers from other integration tests --- #
# Need run_script and potentially the e2e_train_env fixture logic
# For simplicity, let's redefine run_script here, assuming similar needs.
# Adjust imports if these are moved to a shared conftest.py
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

def run_script(script_name: str, args: list[str], cwd=PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Helper function to run a script using subprocess."""
    script_path = SCRIPTS_DIR / script_name
    command = [sys.executable, str(script_path)] + args
    print(f"\nRunning command: {' '.join(command)}") # For debugging
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{str(PROJECT_ROOT)}{os.pathsep}{pythonpath}"
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=env,
        encoding='utf-8'
    )
    if result.returncode != 0:
        print(f"Script Error Output ({script_name}):\n--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}")
    return result

# --- Fixtures ---

@pytest.fixture(scope="function") # Use function scope for isolation
def trained_run_output_dir(tmp_path) -> Path:
    """Fixture that runs the E2E training to produce a run output directory."""
    # --- Create Env --- #
    input_dir = tmp_path / "input"
    tokenizer_dir = tmp_path / "tokenizers"
    processed_data_dir = tmp_path / "processed_data"
    training_output_dir = tmp_path / "training_run_output" # This will be the run dir
    input_dir.mkdir()
    tokenizer_dir.mkdir()
    processed_data_dir.mkdir()

    input_file = input_dir / "sample_text.txt"
    input_file.write_text("A line for training.\nAnother line.\n")

    # --- Phase 1a: Train Tokenizer ---
    tokenizer_model_prefix = "test_sp_gen"
    tokenizer_output_path = tokenizer_dir / tokenizer_model_prefix
    train_tok_args = [
        "--tokenizer_type", "sentencepiece",
        "--input_files", str(input_file),
        "--output_dir", str(tokenizer_dir),
        "--vocab_size", "30", # Very small
        "--model_prefix", tokenizer_model_prefix
    ]
    train_tok_result = run_script("train_tokenizer.py", train_tok_args)
    assert train_tok_result.returncode == 0, "Setup failed: Tokenizer training script failed"
    tokenizer_config_file = tokenizer_dir / "tokenizer_config.json"
    assert tokenizer_config_file.is_file(), "Setup failed: tokenizer_config.json not found"
    try:
        with open(tokenizer_config_file, 'r') as f: tok_config = json.load(f)
        vocab_size = tok_config['vocab_size']
    except Exception as e:
        pytest.fail(f"Setup failed: Could not read vocab size from tokenizer config: {e}")

    # --- Phase 1b: Prepare Data ---
    prepare_data_args = [
        "--type", "subword",
        "--input-path", str(input_file),
        "--output-dir", str(processed_data_dir),
        "--tokenizer-path", str(tokenizer_output_path),
    ]
    prepare_data_result = run_script("prepare_data.py", prepare_data_args)
    assert prepare_data_result.returncode == 0, "Setup failed: Data preparation script failed"
    assert (processed_data_dir / "train.pkl").is_file(), "Setup failed: train.pkl not found"

    # --- Phase 3: Run Training (Minimal) ---
    # Use overrides to configure a minimal run
    overrides = [
        f"experiment=test_e2e_train", # Assumes this config group exists
        f"experiment.data.datasets.train.dataset_params.path={processed_data_dir}",
        f"experiment.data.datasets.val.dataset_params.path={processed_data_dir}", # Use same for val
        f"experiment.model.vocab_size={vocab_size}",
        f"experiment.training.max_steps=1", # Just one step to create checkpoint
        f"experiment.training.log_interval=1",
        f"experiment.training.eval_interval=0",
        f"experiment.training.save_interval=1",
        f"hydra.run.dir={training_output_dir}"
    ]
    train_command = [
        sys.executable, "-m", CRAFT_CLI_MODULE, "train", "language"
    ] + overrides
    # Run from project root
    process = subprocess.run(
        train_command, capture_output=True, text=True, cwd=PROJECT_ROOT, env=os.environ.copy(), encoding='utf-8'
    )
    if process.returncode != 0:
         print(f"SETUP STDERR:\n{process.stderr}")
         pytest.fail(f"Setup failed: Training script failed with code {process.returncode}")

    # Verify checkpoint and tokenizer were created
    checkpoint_dir = training_output_dir / "checkpoints"
    final_tokenizer_dir = training_output_dir / "tokenizer"
    assert checkpoint_dir.is_dir(), f"Setup failed: Checkpoint dir {checkpoint_dir} not created."
    assert any(checkpoint_dir.glob("*.pt")), f"Setup failed: No .pt checkpoint found in {checkpoint_dir}"
    assert final_tokenizer_dir.is_dir(), f"Setup failed: Tokenizer dir {final_tokenizer_dir} not created."
    assert any(final_tokenizer_dir.glob("*.model")), f"Setup failed: No .model found in {final_tokenizer_dir}"

    return training_output_dir # Return the path to the completed run directory

# --- Test Function (Refactored) ---

def test_cli_generate_command(trained_run_output_dir):
    """Tests running the 'generate' command using a generated training run output."""
    prompt = "The"
    max_tokens = 10
    run_dir_path = trained_run_output_dir

    # Find the checkpoint file created (assuming step 1 based on max_steps=1, save_interval=1)
    checkpoint_files = list((run_dir_path / "checkpoints").glob("checkpoint_step_*.pt"))
    assert len(checkpoint_files) > 0, "No checkpoint file found in the generated run directory."
    checkpoint_name = checkpoint_files[0].name # Get the name relative to checkpoints dir

    # Run generate using Hydra config and overrides
    overrides = [
        f"load_from_run={run_dir_path}", # Provide path to the run dir
        f"checkpoint_name={checkpoint_name}", # Specify which checkpoint
        f"generation.start_prompt={prompt}",
        f"generation.max_new_tokens={max_tokens}",
        f"device=cpu" # Ensure test runs on CPU
    ]

    command = [
        sys.executable,
        "-m", CRAFT_CLI_MODULE, # Execute as a module
        "generate", # Hydra will load generate.yaml by default from conf/
    ] + overrides

    print(f"\nRunning command: {' '.join(command)}") # For debugging test runs

    # Run the generation script
    process = run_script("../" + CRAFT_CLI_MODULE.replace('.', '/') + ".py", command[2:]) # Use helper

    # Print output for debugging failures
    print(f"--- Subprocess STDOUT ---\n{process.stdout}")
    print(f"--- Subprocess STDERR ---\n{process.stderr}")

    # 1. Check return code
    assert process.returncode == 0, f"CLI generate script failed with exit code {process.returncode}\nSTDERR: {process.stderr[:1000]}"

    # 2. Check if stdout contains generated text markers
    assert "Generation Complete" in process.stdout, "Expected 'Generation Complete' marker not found"
    assert f"Prompt: {repr(prompt)}" in process.stdout, "Prompt marker not found or mismatch"
    assert "Generated Text:" in process.stdout, "Generated text marker not found"

    # 3. Check if the generated text seems plausible (e.g., contains more than just the prompt)
    output_lines = process.stdout.strip().split('\n')
    generated_line = ""
    for line in reversed(output_lines):
        if line.startswith("Generated Text:"):
            generated_line = line.replace("Generated Text:", "").strip()
            break
    assert len(generated_line) > 0, "Generated text seems empty."
    # Note: Since it's trained for 1 step, output might be garbage, so don't check quality

    print("\nCLI generate command executed successfully using generated run output.")

# Add more tests if needed for different options or scenarios 