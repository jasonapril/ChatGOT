"""
Unit tests for the model factory function.
"""
import pytest
from pydantic import ValidationError
from omegaconf import OmegaConf
import hydra # Import hydra for its exception type

from craft.models.base import LanguageModelConfig, BaseModelConfig
from craft.models.factory import create_model_from_config
from craft.models.transformer import TransformerModel

# Assuming TransformerModel is the expected target for the config
TRANSFORMER_TARGET = "craft.models.transformer.TransformerModel" 

# Assume 'mock_transformer' is registered via entry points or tests/conftest.py
# If not, we might need to register it here for the tests.

class TestModelCreation:
    """Tests for the model creation factory function using the registry (pytest style)."""

    def test_create_language_transformer_model(self):
        """Test creating a registered language transformer model via _target_."""
        # Config defining the model parameters
        config_params = {
            # "model_type": "language", # Type info now goes top-level for factory routing
            "vocab_size": 50257,
            "d_model": 768,
            "n_head": 12,
            "n_layers": 12,
            "max_seq_length": 1024
        }
        # Create the Pydantic config object (still useful for defining params)
        lm_config = LanguageModelConfig(**config_params)
        
        # Structure expected by factory: dict with _target_, model_type, and nested config dict
        factory_dict = {
            "_target_": TRANSFORMER_TARGET,
            "model_type": "language", # Expected at top level by factory
            "config": lm_config.model_dump() # Pass nested parameters as dict
        }
        # Convert to OmegaConf DictConfig
        factory_config = OmegaConf.create(factory_dict)
        
        model = create_model_from_config(factory_config)
        
        assert isinstance(model, TransformerModel)
        # The model should have the Pydantic config object reconstructed internally
        assert isinstance(model.config, LanguageModelConfig)
        assert model.config.vocab_size == 50257
        assert model.config.d_model == 768

    def test_invalid_config_validation(self):
        """Test Pydantic validation before factory and factory missing _target_."""
        # 1. Pydantic validation error when creating config object
        invalid_config_dict = {
            "model_type": "language",
            "d_model": 768, 
        }
        with pytest.raises(ValidationError):
            LanguageModelConfig(**invalid_config_dict)
        
        # 2. Factory error if _target_ is missing
        base_cfg_obj = BaseModelConfig(model_type="test")
        config_dict_without_target = {
             "model_type": "test", # Needs type for potential validation routing
             "config": base_cfg_obj.model_dump()
        }
        factory_config_no_target = OmegaConf.create(config_dict_without_target)
        with pytest.raises(ValueError, match="Model configuration must include '_target_' or 'target'."):
             create_model_from_config(factory_config_no_target)

    def test_factory_instantiation_errors(self):
        """Test error handling for incorrect target or config mismatch."""
        base_cfg_obj = BaseModelConfig(model_type="test")
        
        # 1. Incorrect _target_ path (leads to ImportError wrapped by Hydra)
        invalid_target_dict = {
             "_target_": "non.existent.path.Model",
             "model_type": "test", # Include model_type
             "config": base_cfg_obj.model_dump()
        }
        invalid_target_config = OmegaConf.create(invalid_target_dict)
        # Expect Hydra's InstantiationException wrapping ImportError/ModuleNotFoundError
        # Update: Factory catches Hydra exception and raises generic Exception
        # with pytest.raises(hydra.errors.InstantiationException):
        with pytest.raises(Exception, match="Error instantiating model"):
             create_model_from_config(invalid_target_config)

        # 2. Correct target, but incompatible args (missing 'config')
        mismatched_dict = {
             "_target_": TRANSFORMER_TARGET,
             "model_type": "language", # Include model_type
             "some_other_param": 123 
             # Missing the 'config' key expected by TransformerModel
        }
        mismatched_config = OmegaConf.create(mismatched_dict)
        # Expect TypeError during instantiation within the factory call (likely from model __init__)
        # Update: Factory catches internal errors and raises generic Exception
        # with pytest.raises(TypeError):
        with pytest.raises(Exception, match="Error instantiating model"):
            create_model_from_config(mismatched_config)
        
        # 3. Correct target, but nested config dict has invalid field type
        wrong_config_content_dict = {
             "_target_": TRANSFORMER_TARGET,
             "model_type": "language", # Include model_type
             # Config dict has invalid type for d_model
             # Also include other required fields to ensure validation targets the type error
             "config": {
                 "vocab_size": 100, 
                 "d_model": "not_an_int", # Invalid type
                 "n_head": 4,
                 "n_layers": 2,
                 "max_seq_length": 128
             }
        }
        wrong_config_content_config = OmegaConf.create(wrong_config_content_dict)
        # Expect Pydantic validation error inside the factory's validation step
        with pytest.raises(ValidationError): 
            create_model_from_config(wrong_config_content_config)

    # def test_unregistered_model_type_or_architecture(self):
    # This test is likely obsolete if the factory primarily relies on _target_
    # and doesn't fall back to a registry based on model_type/architecture.
    # If a registry fallback exists, this test would need to be adapted.
    # For now, let's remove it or comment it out, replaced by test_factory_instantiation_errors.
    #    pass 

    # Add more tests here for other model types (e.g., 'generative' if applicable) 
    # and architectures as they are implemented and registered. 