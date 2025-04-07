import pytest
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
import sys
import os

# Ensure the config directory is discoverable by Hydra
# Assuming tests are run from the project root
CONF_DIR = Path(__file__).parent.parent.parent.parent / "conf"
CONFIG_DIR_PATH = str(CONF_DIR.resolve())

# Add project root to sys.path to allow importing src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.craft.config.schemas import AppConfig
# from src.craft.data.tokenizers.char import CharTokenizer # Keep if minimal test remains

# Test experiments defined in the E2E tests
TEST_EXPERIMENTS = ["test_got_char", "test_got_subword"]

@pytest.mark.parametrize("experiment_name", TEST_EXPERIMENTS)
def test_stage1_config_loading_and_validation(experiment_name):
    """Tests Stage 1: Loading experiment config with Hydra and validating with Pydantic."""
    # Clear Hydra global state before initializing
    GlobalHydra.instance().clear()
    try:
        # Use initialize_config_dir for robustness
        with initialize_config_dir(config_dir=CONFIG_DIR_PATH, job_name=f"test_stage1_{experiment_name}", version_base="1.3"):
            # Compose the configuration using the experiment name
            cfg = compose(config_name="config", overrides=[f"experiment={experiment_name}"])

            # Convert OmegaConf to a Python dictionary for Pydantic
            resolved_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

            # Validate with Pydantic
            validated_cfg = AppConfig(**resolved_cfg_dict)

            # If validation passes, the stage is successful
            assert validated_cfg is not None
            assert validated_cfg.experiment is not None
            assert validated_cfg.experiment.experiment_name == experiment_name

    except Exception as e:
        resolved_cfg_dict_str = OmegaConf.to_yaml(cfg) if 'cfg' in locals() else 'Error before config composition'
        pytest.fail(f"Stage 1: Config loading/validation failed for experiment '{experiment_name}' with error: {e}\nResolved Config:\n{resolved_cfg_dict_str}")


# --- Removed Stage 2 Test ---
# The successful E2E tests in test_e2e_training.py now cover dataloader and subsequent stages implicitly.
# Keeping Stage 1 as a fast, isolated check for config loading/validation.

# --- Minimal Test for CharTokenizer Loading --- 
# Keep this? It tests CharTokenizer.load directly, which is useful and independent of factory/Hydra.
def test_minimal_char_tokenizer_load():
    """Directly tests CharTokenizer.load without Hydra/factory."""
    tokenizer_dir = os.path.join(PROJECT_ROOT, "data", "processed", "got", "char", "tokenizer")
    print(f"\n[Minimal Test] Attempting CharTokenizer.load with path: '{tokenizer_dir}' (Type: {type(tokenizer_dir)})")
    
    # Ensure the directory and required files exist before calling load
    assert os.path.isdir(tokenizer_dir), f"Tokenizer directory not found: {tokenizer_dir}"
    assert os.path.exists(os.path.join(tokenizer_dir, 'tokenizer_config.json')), "tokenizer_config.json missing"
    assert os.path.exists(os.path.join(tokenizer_dir, 'vocab.json')), "vocab.json missing"

    try:
        from src.craft.data.tokenizers.char import CharTokenizer # Import locally
        tokenizer = CharTokenizer.load(tokenizer_dir) # Pass the string path directly
        
        assert tokenizer is not None, "CharTokenizer.load returned None"
        assert isinstance(tokenizer, CharTokenizer), f"Loaded object is not a CharTokenizer, but {type(tokenizer)}"
        assert tokenizer.vocab_size > 0, "Loaded tokenizer vocab size is not positive"
        print("[Minimal Test] CharTokenizer.load successful!")

    except Exception as e:
        pytest.fail(f"Minimal CharTokenizer.load test failed with error: {type(e).__name__}: {e}", pytrace=True) 