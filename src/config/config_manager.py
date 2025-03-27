"""
Configuration management for Craft.

This module provides utility functions for loading, merging, and validating
configuration files for the Craft project.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from ..utils.common import set_seed

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for Craft.
    
    Handles loading, merging, and validating configuration files.
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.default_config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration for Craft.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'project': {
                'name': 'Craft',
                'version': '0.1.0',
                'description': 'A modular framework for AI models'
            },
            'system': {
                'seed': 42,
                'log_level': 'INFO',
                'device': 'auto'
            },
            'paths': {
                'data_dir': 'data',
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models_dir': 'models',
                'logs_dir': 'logs',
                'output_dir': 'runs'
            },
            'experiment': {
                'name': 'default',
                'tags': [],
                'description': 'Default experiment'
            }
        }
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            configs: List of configuration dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        merged_config = self.default_config.copy()
        
        for config in configs:
            merged_config = self._deep_merge(merged_config, config)
        
        return merged_config
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - check for required sections
        required_sections = ['project', 'system', 'paths']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section in config: {section}")
                return False
        
        return True
    
    def load_experiment_config(self, experiment_config_path: str) -> Dict[str, Any]:
        """
        Load a complete experiment configuration.
        
        Args:
            experiment_config_path: Path to the experiment configuration file
            
        Returns:
            Complete configuration dictionary
        """
        # Load experiment config
        experiment_config = self.load_config(experiment_config_path)
        
        # Check for model, data, and training configs
        config_paths = []
        
        if 'model_config' in experiment_config:
            model_config_path = experiment_config['model_config']
            config_paths.append(self.load_config(model_config_path))
        
        if 'data_config' in experiment_config:
            data_config_path = experiment_config['data_config']
            config_paths.append(self.load_config(data_config_path))
        
        if 'training_config' in experiment_config:
            training_config_path = experiment_config['training_config']
            config_paths.append(self.load_config(training_config_path))
        
        # Add the experiment config itself
        config_paths.append(experiment_config)
        
        # Merge configs
        merged_config = self.merge_configs(config_paths)
        
        # Validate
        if not self.validate_config(merged_config):
            logger.warning("Config validation failed, but proceeding with merged config")
        
        return merged_config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_manager = ConfigManager()
    return config_manager.load_config(config_path)


def load_experiment_config(experiment_config_path: str) -> Dict[str, Any]:
    """
    Load a complete experiment configuration.
    
    Args:
        experiment_config_path: Path to the experiment configuration file
        
    Returns:
        Complete configuration dictionary
    """
    config_manager = ConfigManager()
    config = config_manager.load_experiment_config(experiment_config_path)
    
    # Set seed from config
    if 'system' in config and 'seed' in config['system']:
        set_seed(config['system']['seed'])
    
    return config 