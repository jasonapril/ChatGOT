"""
Unit tests for configuration schemas defined in src/craft/config/schemas.py.
"""
import pytest
from pydantic import ValidationError
from typing import Annotated # Required for AnyModelConfig

# Schemas to test
from craft.config.schemas import (
    AppConfig,
    ExperimentConfig,
    LanguageModelConfig,
    SimpleRNNConfig,
    DataConfig,
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    AnyModelConfig
)

# --- Test Data Fixtures ---

@pytest.fixture
def valid_transformer_config_dict():
    """Provides a valid configuration dictionary for a Transformer model experiment."""
    return {
        'experiment': {
            'model': {
                'architecture': 'transformer',
                'vocab_size': 10000,
                'd_model': 256,
                'n_head': 8,
                'n_layers': 6,
                'dropout': 0.1,
                'max_seq_length': 512,
            },
            'training': {'batch_size': 32, 'max_steps': 10000, 'use_amp': True, 'log_interval': 10, 'eval_interval': 500, 'save_interval': 1000},
            'data': {
                'batch_size': 32,
                'num_workers': 4,
                'block_size': 512,
                'datasets': {
                    'train': {'dataset': {'_target_': 'craft.data.MyDataset', 'path': '/data/train'}},
                    'val': {'dataset': {'_target_': 'craft.data.MyDataset', 'path': '/data/val'}}
                }
            },
            'optimizer': {'_target_': 'torch.optim.AdamW', 'lr': 3e-4},
            'scheduler': {'_target_': 'torch.optim.lr_scheduler.CosineAnnealingLR', 'T_max': 10000},
        },
        'project_name': 'TestProject',
        'seed': 99
    }

@pytest.fixture
def valid_rnn_config_dict():
    """Provides a valid configuration dictionary for a SimpleRNN model experiment."""
    return {
        'experiment': {
            'model': {
                'architecture': 'simple_rnn',
                'hidden_size': 128,
                'num_layers': 2,
                'vocab_size': 5000,
                'd_model': 64,
                'dropout': 0.2,
                'max_seq_length': 256,
            },
            'training': {'batch_size': 64, 'num_epochs': 5},
            'data': {
                'batch_size': 64,
                'num_workers': 0,
                'block_size': 256,
                'datasets': {
                    'train': {'dataset': {'_target_': 'craft.data.AnotherDataset', 'file': 'train.txt'}},
                    'val': None
                 }
            },
             'optimizer': {'_target_': 'torch.optim.Adam', 'lr': 1e-3},
        },
         'project_name': 'TestRNN',
    }

# --- Test Cases ---

def test_valid_transformer_config(valid_transformer_config_dict):
    """Test successful validation of a complete, valid Transformer config."""
    try:
        config = AppConfig(**valid_transformer_config_dict)
        assert isinstance(config, AppConfig)
        assert isinstance(config.experiment.model, LanguageModelConfig)
        assert config.experiment.model.architecture == 'transformer'
        assert config.experiment.model.n_layers == 6
        assert config.experiment.data.block_size == 512
        assert config.experiment.optimizer.lr == 3e-4
    except ValidationError as e:
        pytest.fail(f"Valid Transformer config failed validation: {e}")

def test_valid_rnn_config(valid_rnn_config_dict):
    """Test successful validation of a complete, valid SimpleRNN config."""
    try:
        config = AppConfig(**valid_rnn_config_dict)
        assert isinstance(config, AppConfig)
        assert isinstance(config.experiment.model, SimpleRNNConfig)
        assert config.experiment.model.architecture == 'simple_rnn'
        assert config.experiment.model.hidden_size == 128
        assert config.experiment.training.num_epochs == 5
        assert config.experiment.data.datasets['val'] is None
    except ValidationError as e:
        pytest.fail(f"Valid RNN config failed validation: {e}")

def test_missing_required_field(valid_transformer_config_dict):
    """Test that validation fails if a required field is missing."""
    invalid_dict = valid_transformer_config_dict.copy()
    # Example: remove required optimizer learning rate
    del invalid_dict['experiment']['optimizer']['lr']
    with pytest.raises(ValidationError):
        AppConfig(**invalid_dict)

def test_incorrect_discriminator(valid_transformer_config_dict):
    """Test validation fails if model config doesn't match discriminator."""
    invalid_dict = valid_transformer_config_dict.copy()
    # Keep architecture='transformer' but provide RNN fields
    invalid_dict['experiment']['model']['hidden_size'] = 128 # Invalid for LanguageModelConfig
    with pytest.raises(ValidationError):
        AppConfig(**invalid_dict)

def test_unknown_architecture(valid_transformer_config_dict):
    """Test validation fails if the architecture value is not in the Union."""
    invalid_dict = valid_transformer_config_dict.copy()
    invalid_dict['experiment']['model']['architecture'] = "unknown_model"
    with pytest.raises(ValidationError):
        AppConfig(**invalid_dict)

def test_wrong_field_type(valid_transformer_config_dict):
    """Test validation fails if a field has the wrong type."""
    invalid_dict = valid_transformer_config_dict.copy()
    # Set learning rate to a string instead of float
    invalid_dict['experiment']['optimizer']['lr'] = "not-a-float"
    with pytest.raises(ValidationError):
        AppConfig(**invalid_dict)

def test_data_split_config(valid_transformer_config_dict):
    """Test the nested DataConfig and DatasetSplitConfig validation."""
    config = AppConfig(**valid_transformer_config_dict)
    assert config.experiment.data.datasets['train'] is not None
    assert config.experiment.data.datasets['train'].dataset_target == 'craft.data.MyDataset'
    assert config.experiment.data.datasets['train'].dataset_params == {'path': '/data/train'}

def test_optional_scheduler(valid_transformer_config_dict):
    """Test config validates correctly when optional scheduler is missing."""
    config_dict = valid_transformer_config_dict.copy()
    del config_dict['experiment']['scheduler'] # Remove optional scheduler
    try:
        config = AppConfig(**config_dict)
        assert config.experiment.scheduler is None
    except ValidationError as e:
        pytest.fail(f"Config without optional scheduler failed validation: {e}")

# Add more tests as needed for edge cases, specific validators (like d_hid), callbacks etc. 