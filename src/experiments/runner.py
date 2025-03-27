"""Experiment runner for comparing different model configurations."""
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from omegaconf import OmegaConf
from mlflow import start_run, log_metrics, log_params, log_artifacts

from ..models.transformer import create_transformer_model
from ..data.dataset import prepare_dataloaders_from_config
from ..training.trainer import train_with_samples
from ..utils.metrics import calculate_metrics
from ..utils.logging import get_logger, setup_logging
from ..utils.performance import setup_monitoring, get_resource_metrics

logger = get_logger(__name__)

def load_experiment_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment(exp_config: Dict, output_dir: str) -> Path:
    """Set up experiment directory and logging."""
    exp_dir = Path(output_dir) / exp_config['name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(log_file=exp_dir / 'experiment.log')
    
    # Set up monitoring if enabled
    if exp_config.get('monitoring', {}).get('enabled', False):
        setup_monitoring(exp_config['monitoring'])
    
    return exp_dir

def run_model_training(
    model_config: Dict,
    train_config: Dict,
    exp_dir: Path,
    run_name: str,
) -> Dict:
    """Run training for a single model configuration."""
    # Create model
    model = create_transformer_model(**model_config['params'])
    
    # Prepare data
    train_dataloader, val_dataloader = prepare_dataloaders_from_config(
        OmegaConf.create(train_config)
    )
    
    # Set up MLflow run
    with start_run(run_name=run_name) as run:
        # Log parameters
        log_params(model_config['params'])
        log_params(train_config)
        
        # Train model
        metrics = train_with_samples(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            **train_config
        )
        
        # Log metrics
        log_metrics(metrics)
        
        # Save model and artifacts
        model_path = exp_dir / f"{run_name}_model.pt"
        torch.save(model.state_dict(), model_path)
        log_artifacts(str(model_path))
        
        return metrics

def run_experiment(config_path: str, output_dir: str):
    """Run a complete experiment comparing different model configurations."""
    # Load experiment configuration
    exp_config = load_experiment_config(config_path)
    
    # Set up experiment directory
    exp_dir = setup_experiment(exp_config, output_dir)
    
    # Run each model configuration
    results = {}
    for model_config in exp_config['models']:
        run_name = f"{exp_config['name']}_{model_config['name']}"
        logger.info(f"Running experiment: {run_name}")
        
        try:
            metrics = run_model_training(
                model_config=model_config,
                train_config=exp_config['training'],
                exp_dir=exp_dir,
                run_name=run_name,
            )
            results[model_config['name']] = metrics
        except Exception as e:
            logger.error(f"Error running {run_name}: {e}")
            continue
    
    # Save experiment results
    results_path = exp_dir / 'results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    logger.info(f"Experiment completed. Results saved to {results_path}")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to experiment configuration file')
    parser.add_argument('--output_dir', type=str, default='runs/experiments',
                      help='Directory to save experiment results')
    args = parser.parse_args()
    
    run_experiment(args.config_path, args.output_dir) 