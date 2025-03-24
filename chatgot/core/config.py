"""
Configuration management utilities for ChatGoT.

This module provides functions to load, validate, and access configurations 
using Hydra and OmegaConf.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")


def get_config_path() -> str:
    """Get the path to the configuration directory.
    
    Returns:
        Path to the configuration directory.
    """
    return CONFIG_PATH


def register_configs() -> None:
    """Register structured configs with Hydra."""
    cs = ConfigStore.instance()
    # Register structured configs here when we define them
    # cs.store(name="config_schema", node=ConfigSchema)


def load_config(config_name: str = "default", overrides: Optional[list] = None) -> DictConfig:
    """Load a configuration from the config directory.
    
    Args:
        config_name: Name of the configuration file (without extension)
        overrides: List of configuration overrides
        
    Returns:
        Loaded configuration as a DictConfig object
    """
    with hydra.initialize_config_module(config_module=get_config_path()):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
    return cfg


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save a configuration to a file.
    
    Args:
        config: Configuration to save
        save_path: Path to save the configuration to
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
    
    logger.info(f"Configuration saved to {save_path}")


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert a configuration to a dictionary.
    
    Args:
        config: Configuration to convert
        
    Returns:
        Dictionary representation of the configuration
    """
    return OmegaConf.to_container(config, resolve=True)


def get_full_path(relative_path: str, config: Optional[DictConfig] = None) -> Path:
    """Convert a relative path to an absolute path based on the configuration.
    
    Args:
        relative_path: Relative path
        config: Configuration to use for base paths. If None, paths are relative to CWD.
        
    Returns:
        Absolute path
    """
    if os.path.isabs(relative_path):
        return Path(relative_path)
    
    # If path starts with a reference like ${paths.data_dir}, resolve it
    if relative_path.startswith("${") and config is not None:
        resolved_path = OmegaConf.select(config, relative_path)
        if resolved_path is not None:
            return Path(resolved_path)
    
    # Otherwise, use the CWD as base
    return Path(os.getcwd()) / relative_path 