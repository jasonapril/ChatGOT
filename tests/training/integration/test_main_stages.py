import pytest
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
import sys
import os
import tempfile
import json
import shutil
from craft.data.tokenizers.char import CharTokenizer

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
    """Test loading a minimal pre-saved CharTokenizer."""
    # Setup: Create a minimal tokenizer setup manually
    temp_dir = Path(tempfile.mkdtemp(prefix="test_char_tok_"))
    config_path = temp_dir / "tokenizer_config.json"
    vocab_path = temp_dir / "vocab.json"
    
    char_to_idx = {'a': 0, 'b': 1, '<unk>': 2}
    config_data = {
        "model_type": "char",
        "vocab_size": 3,
        "special_tokens": {"unk_token": "<unk>", "unk_id": 2}
    }
    
    with open(config_path, 'w') as f: json.dump(config_data, f)
    with open(vocab_path, 'w') as f: json.dump(char_to_idx, f)
    
    try:
        # Load using the class method
        tokenizer = CharTokenizer.load_from_dir(str(temp_dir))
        assert isinstance(tokenizer, CharTokenizer)
        assert tokenizer.vocab_size == 3
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.unk_token_id == 2
    except Exception as e:
        pytest.fail(f"Minimal CharTokenizer.load test failed with error: {e}")
    finally:
        shutil.rmtree(temp_dir) 