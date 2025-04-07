# tests/config/test_config_loading.py

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

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
        
        # Validate against Pydantic schema
        validated_config = AppConfig(**config_dict)
        
        # --- DEBUG PRINT --- #
        print("\nDEBUG (got_char): validated_config.experiment.data.datasets")
        print(validated_config.experiment.data.datasets)
        print(type(validated_config.experiment.data.datasets)) # Check type
        # --- END DEBUG PRINT --- #

        # Basic checks on the validated config
        assert validated_config is not None
        assert validated_config.experiment.data.type == "char_pickle"
        # Access nested model config via dictionary keys
        assert validated_config.experiment.model.config['architecture'] == "transformer"
        assert validated_config.experiment.data.block_size == 256 # Check resolved value (interpolated from model.config.max_seq_length)
        # Check if interpolation worked for nested dataset block_size
        # Access datasets via dictionary keys
        assert validated_config.experiment.data.datasets['train'].dataset['block_size'] == 256

    except Exception as e:
        pytest.fail(f"Config loading/validation failed for test_got_char: {e}")

def test_load_and_validate_got_subword_config():
    """Tests loading and validating the test_got_subword experiment config."""
    try:
        cfg = compose(config_name="config", overrides=["experiment=test_got_subword"])
        OmegaConf.resolve(cfg)
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        validated_config = AppConfig(**config_dict)

        # --- DEBUG PRINT --- #
        print("\nDEBUG (got_subword): validated_config.experiment.data.datasets")
        print(validated_config.experiment.data.datasets)
        print(type(validated_config.experiment.data.datasets)) # Check type
        # --- END DEBUG PRINT --- #

        assert validated_config is not None
        assert validated_config.experiment.data.type == "subword_pickle"
        # Access nested model config via dictionary keys
        assert validated_config.experiment.model.config['architecture'] == "transformer"
        assert validated_config.experiment.data.block_size == 256 # Check resolved value
        # Access datasets via dictionary keys
        assert validated_config.experiment.data.datasets['train'].dataset['block_size'] == 256

    except Exception as e:
        pytest.fail(f"Config loading/validation failed for test_got_subword: {e}") 