# tests/config/test_config_loading.py

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import json # For debug printing
import sys # For potential debug exit

from src.craft.config.schemas import AppConfig # Adjust import path if necessary

# Helper to ensure Hydra is initialized safely for tests
@pytest.fixture(autouse=True)
def hydra_init_fixture():
    if not GlobalHydra().is_initialized():
        initialize(config_path="../../conf", job_name="test_config_loading")
    yield
    # Optional: Clear Hydra instance if it causes issues between tests
    # GlobalHydra.instance().clear()

def test_load_and_validate_got_char_config():
    """Tests loading and validating the test_got_char experiment config."""
    try:
        # Compose configuration using Hydra
        # Relative path from this test file to the 'conf' directory
        cfg = compose(config_name="config", overrides=["experiment=test_got_char"])
        
        # Convert OmegaConf object to a standard Python dict for Pydantic validation
        # Resolve interpolations before converting
        OmegaConf.resolve(cfg)
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        
        # --- ADDED DEBUG --- #
        print("\nDEBUG (got_char): config_dict passed to AppConfig:")
        print(json.dumps(config_dict, indent=2))
        # sys.exit(1) # Uncomment to halt execution here and inspect the dict
        # --- END DEBUG --- #
        
        # Validate against Pydantic schema
        validated_config = AppConfig(**config_dict)
        
        # --- DEBUG PRINT --- # Removed, was for checking structure before
        # print("\nDEBUG (got_char): validated_config.experiment.data.datasets")
        # print(validated_config.experiment.data.datasets)
        # print(type(validated_config.experiment.data.datasets)) # Check type
        # --- END DEBUG PRINT --- #

        # Basic checks on the validated config (Updated assertions)
        assert validated_config is not None
        assert validated_config.experiment.data.type == "char_pickle"
        # Access model config directly
        assert validated_config.experiment.model.architecture == "transformer"
        # Expect block_size from /data: got_char default (1024)
        assert validated_config.experiment.data.block_size == 1024
        # Check interpolation for nested dataset block_size (should also be 1024 now)
        assert validated_config.experiment.data.datasets['train'].dataset_params['block_size'] == 1024

    except Exception as e:
        pytest.fail(f"Config loading/validation failed for test_got_char: {e}")

def test_load_and_validate_got_subword_config():
    """Tests loading and validating the test_got_subword experiment config."""
    try:
        cfg = compose(config_name="config", overrides=["experiment=test_got_subword"])
        OmegaConf.resolve(cfg)
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        # --- ADDED DEBUG --- #
        print("\nDEBUG (got_subword): config_dict passed to AppConfig:")
        print(json.dumps(config_dict, indent=2))
        # sys.exit(1) # Uncomment to halt execution here and inspect the dict
        # --- END DEBUG --- #

        validated_config = AppConfig(**config_dict)

        # --- DEBUG PRINT --- # Removed
        # print("\nDEBUG (got_subword): validated_config.experiment.data.datasets")
        # print(validated_config.experiment.data.datasets)
        # print(type(validated_config.experiment.data.datasets)) # Check type
        # --- END DEBUG PRINT --- #

        # Updated assertions
        assert validated_config is not None
        assert validated_config.experiment.data.type == "subword_pickle"
        assert validated_config.experiment.model.architecture == "transformer"
        assert validated_config.experiment.data.block_size == 256
        assert validated_config.experiment.data.datasets['train'].dataset_params['block_size'] == 256

    except Exception as e:
        pytest.fail(f"Config loading/validation failed for test_got_subword: {e}") 