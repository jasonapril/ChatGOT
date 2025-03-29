"""Experiment runner for comparing different model configurations."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import yaml
import hydra
from omegaconf import OmegaConf, DictConfig
from mlflow import start_run, log_metrics, log_params, log_artifacts, end_run
from transformers import AutoTokenizer

from ..models.base import create_model_from_config
from ..data.base import prepare_dataloaders_from_config
from ..training.base import LanguageModelTrainer
from ..training.callbacks import SampleGenerationCallback
from ..utils import logging as logging_utils
from ..utils.performance import get_resource_metrics
from ..utils.common import set_seed, setup_device

# Import the standard logging module
import logging

# Get logger using the standard logging module
logger = logging.getLogger(__name__)

def load_experiment_config(config_path: str) -> DictConfig:
    """Load experiment configuration using OmegaConf."""
    return OmegaConf.load(config_path)

def setup_experiment(exp_config: DictConfig, output_dir: str) -> Path:
    """Set up experiment directory and logging."""
    exp_name = exp_config.get('name', 'default_experiment')
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_level = exp_config.get("logging", {}).get("level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging_utils.setup_logger(
        name=None,
        level=numeric_level, 
        log_file=exp_dir / 'experiment.log'
    )
    logger.info(f"Logging setup complete for experiment: {exp_name}")
    
    return exp_dir

def run_model_training(
    model_config: DictConfig,
    train_config: DictConfig,
    data_config: DictConfig,
    exp_dir: Path,
    run_name: str,
) -> Dict:
    """Run training for a single model configuration using the Trainer framework."""
    set_seed(train_config.get("seed", 42))
    device = setup_device(train_config.get("device", "auto"))

    logger.info("Loading tokenizer...")
    tokenizer = None
    try:
        tokenizer_name_or_path = data_config.tokenizer_name
        logger.info(f"Attempting to load tokenizer: {tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        logger.info(f"Successfully loaded tokenizer: {tokenizer_name_or_path}")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token ({tokenizer.eos_token})")

    except Exception as e:
        logger.error(f"Failed to load tokenizer '{data_config.get('tokenizer_name', 'N/A')}': {e}", exc_info=True)

    if tokenizer:
        if 'vocab_size' in model_config:
            model_config.vocab_size = len(tokenizer)
            logger.info(f"Set model_config.vocab_size to {len(tokenizer)} based on tokenizer.")
    else:
        if 'vocab_size' not in model_config or model_config.vocab_size is None:
            logger.error("Tokenizer failed to load and model_config.vocab_size is not set!")
            raise ValueError("Cannot proceed without a tokenizer or an explicit vocab_size in model config.")

    logger.info("Preparing data loaders...")
    train_dataloader, val_dataloader = prepare_dataloaders_from_config(
        config=data_config,
        tokenizer=tokenizer
    )

    logger.info("Creating model...")
    model = create_model_from_config(model_config)
    model.to(device)

    logger.info("Instantiating callbacks...")
    callbacks = []
    if train_config.get("callbacks"):
        try:
            callbacks = [hydra.utils.instantiate(cb_conf, _convert_="partial") for cb_conf in train_config.callbacks]
        except Exception as e:
            logger.error(f"Failed to instantiate callbacks: {e}", exc_info=True)
            callbacks = []

    logger.info("Creating trainer...")
    try:
        checkpoint_dir = exp_dir / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.set_struct(train_config, False)
        train_config.checkpoint_dir = str(checkpoint_dir)
        train_config.device = str(device)
        OmegaConf.set_struct(train_config, True)

        trainer = LanguageModelTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=train_config,
            callbacks=callbacks
        )
    except Exception as e:
        logger.error(f"Failed to create Trainer: {e}", exc_info=True)
        raise

    logger.info("Wiring tokenizer/device to callbacks...")
    for callback in trainer.callbacks:
        if isinstance(callback, SampleGenerationCallback):
            if tokenizer:
                callback.tokenizer = tokenizer
                logger.info(f"Injected tokenizer into {callback.__class__.__name__}")
            else:
                logger.warning(f"Tokenizer not available, {callback.__class__.__name__} might not function.")

    logger.info("Starting MLflow run...")
    with start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        if config.mlflow.log_params:
             log_params(OmegaConf.to_container(config.model, resolve=True), prefix="model")
             log_params(OmegaConf.to_container(config.training, resolve=True), prefix="training")
             log_params(OmegaConf.to_container(config.data, resolve=True), prefix="data")

        logger.info("Starting training...")
        metrics = trainer.train()
        logger.info("Training finished.")

        if config.mlflow.log_metrics:
            final_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, list) and v:
                    final_metrics[f"final_{k}"] = v[-1]
                elif isinstance(v, (int, float)):
                     final_metrics[f"final_{k}"] = v
            if final_metrics:
                log_metrics(final_metrics)
                logger.info(f"Logged final metrics to MLflow: {final_metrics}")

        logger.info("Saving final model checkpoint...")
        final_model_path = Path(trainer.checkpoint_dir) / "final_model.pt"
        trainer.save_checkpoint(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        log_artifacts(str(Path(trainer.checkpoint_dir).parent))

    end_run()
    logger.info("MLflow run ended.")

    return final_metrics

def run_experiment(config: DictConfig, output_dir: str):
    """Run a complete experiment comparing different model configurations."""
    exp_dir = setup_experiment(config, output_dir)
    logger.info(f"Experiment directory: {exp_dir}")

    resolved_config_path = exp_dir / "resolved_config.yaml"
    OmegaConf.save(config, resolved_config_path)

    results = {}
    for model_variant_name, model_config in config.get("models", {}).items():
        run_name = f"{config.get('name', 'exp')}_{model_variant_name}"
        logger.info(f"--- Running Model Variant: {run_name} ---")

        try:
            final_metrics = run_model_training(
                model_config=model_config,
                train_config=config.get("training"),
                data_config=config.get("data"),
                exp_dir=exp_dir,
                run_name=run_name,
            )
            results[model_variant_name] = final_metrics
            logger.info(f"--- Finished Model Variant: {run_name} --- Metrics: {final_metrics}")

        except Exception as e:
            logger.error(f"Error running variant {run_name}: {e}", exc_info=True)
            results[model_variant_name] = {"status": "failed", "error": str(e)}
            continue

    results_path = exp_dir / 'results.yaml'
    try:
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Experiment summary saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save experiment summary: {e}")

    logger.info("=== Experiment Completed ===")
    return results 