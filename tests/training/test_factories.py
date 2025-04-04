"""
Tests for optimizer and scheduler factory functions.
"""
import pytest
import torch
import torch.optim as optim
from torch import nn
from omegaconf import OmegaConf, DictConfig

from craft.training.optimizers import create_optimizer
from craft.training.schedulers import create_scheduler

# --- Fixtures ---

@pytest.fixture
def dummy_model():
    """Provides a simple nn.Module for optimizer tests."""
    return nn.Linear(10, 2) # Simple model with parameters

@pytest.fixture
def base_optimizer(dummy_model):
    """Provides a basic optimizer instance for scheduler tests."""
    return optim.AdamW(dummy_model.parameters(), lr=1e-3)

# --- Tests for create_optimizer --- 

def test_create_optimizer_adamw_success(dummy_model):
    """Test creating AdamW optimizer successfully."""
    cfg = OmegaConf.create({
        "_target_": "torch.optim.AdamW",
        "lr": 0.002,
        "weight_decay": 0.05
    })
    optimizer = create_optimizer(dummy_model, cfg)
    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.defaults['lr'] == 0.002
    assert optimizer.defaults['weight_decay'] == 0.05

def test_create_optimizer_adamw_short_name_success(dummy_model):
    """Test creating AdamW using the short name."""
    cfg = OmegaConf.create({
        "_target_": "AdamW",
        "lr": 0.001
    })
    optimizer = create_optimizer(dummy_model, cfg)
    assert isinstance(optimizer, optim.AdamW)

def test_create_optimizer_missing_target(dummy_model):
    """Test error when _target_ is missing."""
    cfg = OmegaConf.create({"lr": 0.001})
    with pytest.raises(ValueError, match="must specify 'target' or '_target_'"):
        create_optimizer(dummy_model, cfg)

def test_create_optimizer_adamw_missing_lr(dummy_model):
    """Test error when AdamW config misses the required 'lr' parameter."""
    cfg = OmegaConf.create({"_target_": "AdamW", "weight_decay": 0.01})
    with pytest.raises(ValueError, match="must include 'lr'"):
        create_optimizer(dummy_model, cfg)

def test_create_optimizer_unsupported_type(dummy_model):
    """Test error for an explicitly unsupported optimizer type."""
    cfg = OmegaConf.create({"_target_": "torch.optim.SGD", "lr": 0.01})
    with pytest.raises(ValueError, match="Unsupported optimizer type: torch.optim.SGD"):
        create_optimizer(dummy_model, cfg)

def test_create_optimizer_invalid_param_type(dummy_model):
    """Test TypeError propagation for invalid parameter types."""
    cfg = OmegaConf.create({
        "_target_": "AdamW",
        "lr": "invalid_lr_type" # Pass string instead of float
    })
    with pytest.raises(ValueError, match="Invalid parameters for optimizer"):
         # The inner error is TypeError, but we wrap it in ValueError
        create_optimizer(dummy_model, cfg)

# --- Tests for create_scheduler --- 

def test_create_scheduler_cosine_success(base_optimizer):
    """Test creating CosineAnnealingLR successfully."""
    cfg = OmegaConf.create({
        "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 0.0001
    })
    scheduler = create_scheduler(base_optimizer, cfg)
    assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 100
    assert scheduler.eta_min == 0.0001

def test_create_scheduler_cosine_short_name_success(base_optimizer):
    """Test creating CosineAnnealingLR using the short name."""
    cfg = OmegaConf.create({
        "_target_": "CosineAnnealingLR",
        "T_max": 50
    })
    scheduler = create_scheduler(base_optimizer, cfg)
    assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

def test_create_scheduler_none_config(base_optimizer):
    """Test that None is returned when config is None."""
    scheduler = create_scheduler(base_optimizer, None)
    assert scheduler is None

def test_create_scheduler_missing_target(base_optimizer):
    """Test error when _target_ is missing."""
    cfg = OmegaConf.create({"T_max": 100})
    with pytest.raises(ValueError, match="must specify 'target' or '_target_'"):
        create_scheduler(base_optimizer, cfg)

def test_create_scheduler_cosine_missing_tmax(base_optimizer):
    """Test error when CosineAnnealingLR misses required 'T_max'."""
    cfg = OmegaConf.create({"_target_": "CosineAnnealingLR", "eta_min": 0.01})
    with pytest.raises(ValueError, match="CosineAnnealingLR config must include 'T_max'"):
        create_scheduler(base_optimizer, cfg)

def test_create_scheduler_unsupported_type(base_optimizer):
    """Test error for an explicitly unsupported scheduler type."""
    cfg = OmegaConf.create({"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 10})
    with pytest.raises(ValueError, match="Unsupported or unrecognized scheduler type: torch.optim.lr_scheduler.StepLR"):
        create_scheduler(base_optimizer, cfg)

def test_create_scheduler_invalid_param_type(base_optimizer):
    """Test TypeError propagation for invalid parameter types."""
    cfg = OmegaConf.create({
        "_target_": "CosineAnnealingLR",
        "T_max": "invalid_tmax_type" # Pass string instead of int
    })
    with pytest.raises(ValueError, match="parameter 'T_max' must be an integer"):
        create_scheduler(base_optimizer, cfg) 