# tests/training/integration/test_e2e_training.py

import pytest
import subprocess
import sys
import os
import time
import shutil
import logging
from pathlib import Path
import re
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Ensure root logger is configured (useful for seeing logs in tests)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine the project root based on the test file location
# __file__ -> C:/.../craft/tests/training/integration/test_e2e_training.py
TEST_FILE_DIR = Path(__file__).parent # .../tests/training/integration
TESTS_DIR = TEST_FILE_DIR.parent.parent # .../tests
PROJECT_ROOT = TESTS_DIR.parent # .../craft/
ABSOLUTE_CONF_DIR = (PROJECT_ROOT / "conf").resolve() # .../craft/conf/

logger.info(f"PROJECT_ROOT calculated as: {PROJECT_ROOT}")
logger.info(f"ABSOLUTE_CONF_DIR calculated as: {ABSOLUTE_CONF_DIR}")

# List of minimal experiment configurations to test
# Ensure these YAML files exist in `conf/experiment/`
TEST_EXPERIMENTS = ["test_got_char", "test_got_subword"]

# --- Helper function removed: find_latest_hydra_run_dir ---


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Ensure the base outputs directory exists before tests."""
    # Ensure base directory for test outputs exists
    Path("outputs/experiments/test_e2e_training").mkdir(parents=True, exist_ok=True)
    yield
    # Teardown handled within the test function (directory removal)


@pytest.mark.parametrize("experiment_config", TEST_EXPERIMENTS)
def test_minimal_training_run_e2e(experiment_config: str):
    """
    Runs the main training script as a subprocess for a minimal configuration
    and checks for successful completion and expected output artifacts (like checkpoints).
    """
    # # Get paths relative to the determined project root - script_path no longer needed
    # script_path = PROJECT_ROOT / "src/craft/main.py"
    # assert script_path.exists(), f"Training script not found at {script_path}" # Removed assertion
    assert ABSOLUTE_CONF_DIR.exists(), f"Absolute config dir not found at {ABSOLUTE_CONF_DIR}"

    # Define the specific output directory for this test run
    target_run_dir = PROJECT_ROOT / f"outputs/experiments/test_e2e_training/{experiment_config}"
    run_dir_path_to_clean = target_run_dir # Track for cleanup

    # Clean up any previous run directory for this specific experiment before starting
    if target_run_dir.exists():
        logger.warning(f"Removing previous test output directory: {target_run_dir}")
        shutil.rmtree(target_run_dir)

    try:
        # Construct the command to run train_commands.py directly with Hydra args
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["HYDRA_FULL_ERROR"] = "1" # Ensure full Hydra errors are shown

        # Path to the train_commands.py script
        train_script_path = PROJECT_ROOT / "src/craft/cli/train_commands.py"
        assert train_script_path.exists(), f"Train commands script not found at {train_script_path}"

        # Command uses module execution of the hydra-decorated script
        command = [
            sys.executable,
            "-m",
            "craft.cli.train_commands", # Module path to the hydra script
            # --- Hydra Overrides --- #
            f"experiment={experiment_config}",
            f"hydra.run.dir={str(target_run_dir).replace('\\\\', '/')}",
        ]

        # Note: hydra.job.chdir is managed by @hydra.main itself based on config/defaults

        logger.info(f"Running command: {' '.join(command)}")
        start_time = time.time()

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            encoding='utf-8', # Explicitly set encoding
            stderr=subprocess.STDOUT # <<< Add this line
        )

        # Always log stdout/stderr for debugging
        final_global_step = -1 # Initialize
        if process.stdout: # Log merged output if non-empty
            # logger.info(f"--- {experiment_config} STDOUT/STDERR (merged) ---\\n{process.stdout}") # DEBUG: Log all output
            pass # Avoid excessive logging unless debugging

            # Try to parse final global step from logs (Updated Regex)
            # Match: "[2025-04-06 23:33:53,823][Trainer][INFO] - [Trainer Train End] Final self.global_step: 10"
            log_pattern = re.compile(r"\[Trainer Train End\] Final self\.global_step:\s*(\d+)") # Match the actual log format
            for line in reversed(process.stdout.strip().split('\\n')): # Search from end
                match = log_pattern.search(line)
                if match:
                    try:
                        final_global_step = int(match.group(1))
                        logger.info(f"[E2E TEST PARSE] Parsed final_global_step: {final_global_step}")
                        break # Found the last occurrence
                    except (ValueError, IndexError) as parse_err:
                         logger.warning(f"[E2E TEST PARSE] Failed to parse final_global_step from matched group: '{match.group(1)}'. Error: {parse_err}")

            if final_global_step == -1 and process.returncode == 0: # Only warn if process succeeded but parsing failed
                logger.error(f"--- {experiment_config} STDOUT/STDERR (merged) for PARSE FAILURE ---\\n{process.stdout}") # Log output on parse failure
                logger.warning("[E2E TEST PARSE] Could not find 'final_global_step' in logs.")


        # 1. Check return code
        assert process.returncode == 0, f"Training script failed with exit code {process.returncode} for experiment '{experiment_config}'.\\nOutput:\\n{process.stdout}"

        # Load config to get expected max_steps
        # Initialize Hydra manually within the test to load the specific experiment config
        # Use the calculated absolute path based on __file__
        logger.info(f"Using absolute config dir for check steps: {ABSOLUTE_CONF_DIR}")

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(ABSOLUTE_CONF_DIR), version_base=None, job_name="e2e_test_check_steps"):
            cfg = compose(config_name="config", overrides=[f"experiment={experiment_config}"])
        expected_max_steps = cfg.experiment.training.get("max_steps")
        expected_save_interval = cfg.experiment.training.get("save_steps_interval")

        # 2. Check final global step (if found and expected)
        if expected_max_steps is not None:
            assert final_global_step != -1, f"Could not parse final_global_step from logs for {experiment_config}. Check log format and regex."
            assert final_global_step >= expected_max_steps, f"Final global step {final_global_step} is less than expected max_steps {expected_max_steps} for {experiment_config}."
            logger.info(f"[E2E TEST CHECK] final_global_step ({final_global_step}) >= expected_max_steps ({expected_max_steps}) - PASSED")
        else:
            logger.warning(f"Skipping final_global_step check for {experiment_config} as max_steps not defined.")


        # 3. Check if a checkpoint marker exists for the LAST expected save interval step
        # Initialize Hydra again to get the config for the specific run's output dir
        logger.info(f"Using absolute config dir for checkpoint check: {ABSOLUTE_CONF_DIR}")
        GlobalHydra.instance().clear() # Clear previous Hydra state
        with initialize_config_dir(config_dir=str(ABSOLUTE_CONF_DIR), version_base=None, job_name="e2e_test_checkpoint_check"):
            cfg_for_paths = compose(config_name="config", overrides=[f"experiment={experiment_config}"])

        # Construct the expected checkpoint path using the *actual* run directory used by the subprocess
        checkpoint_dir = target_run_dir / "checkpoints"

        if expected_max_steps is not None and expected_save_interval is not None and expected_save_interval > 0:
            # Find the last step that *should* have triggered a save
            last_save_step = (expected_max_steps // expected_save_interval) * expected_save_interval
            if last_save_step > 0: # Only check if a save was expected
                # Construct the marker name based on the actual checkpoint file name convention
                checkpoint_base_name = f"checkpoint_step_{last_save_step:06d}.pt"
                checkpoint_marker_name = f"{checkpoint_base_name}._SAVED"
                expected_marker_path = checkpoint_dir / checkpoint_marker_name
                logger.info(f"Checking for checkpoint marker: {expected_marker_path}")

                assert expected_marker_path.exists(), (
                    f"Checkpoint marker file '{checkpoint_marker_name}' not found in {checkpoint_dir} "
                    f"for experiment '{experiment_config}' after {expected_max_steps} steps "
                    f"(expected save interval: {expected_save_interval}, last expected save step: {last_save_step})."
                    f"\\nDirectory contents: {[p.name for p in checkpoint_dir.iterdir() if checkpoint_dir.exists()]}"
                )
                logger.info(f"[E2E TEST CHECK] Checkpoint marker '{checkpoint_marker_name}' found - PASSED")
            else:
                logger.info(f"Skipping checkpoint marker check for {experiment_config} as no save step <= max_steps was expected.")
        else:
            logger.warning(f"Skipping checkpoint marker check for {experiment_config} due to missing max_steps or save_steps_interval.")


    finally:
        #pass # Optionally skip cleanup for debugging
        # Clean up the specific run directory created by this test
        # Add a small delay just in case file handles are still open
        time.sleep(0.1)
        if run_dir_path_to_clean is not None and run_dir_path_to_clean.exists():
            try:
                logger.warning(f"Cleaning up test output directory: {run_dir_path_to_clean}")
                shutil.rmtree(run_dir_path_to_clean)
            except OSError as e:
                logger.error(f"Error removing directory {run_dir_path_to_clean}: {e}")
        elif run_dir_path_to_clean is not None:
             logger.warning(f"Test output directory already cleaned up or not created: {run_dir_path_to_clean}")

# If you add more tests or helpers, place them below 