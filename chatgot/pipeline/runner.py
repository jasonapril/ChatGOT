"""Pipeline runner for the ChatGoT project."""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from chatgot.core.config import get_full_path
from chatgot.data.processor import process_data
from chatgot.models.generate import generate_text
from chatgot.training.trainer import train_model
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


def run_process_stage(cfg: DictConfig) -> bool:
    """
    Run the data processing stage.
    
    Args:
        cfg: Configuration
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting data processing stage")
    
    if not cfg.pipeline.process.enabled:
        logger.info("Data processing stage is disabled, skipping")
        return True
    
    start_time = time.time()
    result = process_data(cfg, save_analysis=True)
    elapsed = time.time() - start_time
    
    if result == 0:
        logger.info(f"Data processing completed successfully in {elapsed:.2f} seconds")
        return True
    else:
        logger.error("Data processing failed")
        return False


def run_train_stage(cfg: DictConfig) -> bool:
    """
    Run the training stage.
    
    Args:
        cfg: Configuration
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting training stage")
    
    if not cfg.pipeline.train.enabled:
        logger.info("Training stage is disabled, skipping")
        return True
    
    start_time = time.time()
    result = train_model(cfg, resume=cfg.pipeline.resume)
    elapsed = time.time() - start_time
    
    if result == 0:
        logger.info(f"Training completed successfully in {elapsed:.2f} seconds")
        return True
    else:
        logger.error("Training failed")
        return False


def run_generate_stage(cfg: DictConfig) -> bool:
    """
    Run the generation stage.
    
    Args:
        cfg: Configuration
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting generation stage")
    
    if not cfg.pipeline.generate.enabled:
        logger.info("Generation stage is disabled, skipping")
        return True
    
    try:
        start_time = time.time()
        
        # Get parameters from config
        checkpoint_path = cfg.pipeline.generate.get("checkpoint_path", "${paths.models_dir}/model_best.pt")
        checkpoint_path = get_full_path(checkpoint_path, cfg)
        
        num_samples = cfg.pipeline.generate.get("num_samples", 5)
        max_new_tokens = cfg.pipeline.generate.get("max_new_tokens", 500)
        prompt = cfg.pipeline.generate.get("prompt", "In the Game of Thrones, ")
        output_file = cfg.pipeline.generate.get("output_file", "${paths.outputs_dir}/generated_text.txt")
        output_file = get_full_path(output_file, cfg)
        
        # Generate text
        generated_texts = generate_text(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            output_file=output_file,
            num_samples=num_samples,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Text generation completed successfully in {elapsed:.2f} seconds")
        logger.info(f"Generated {len(generated_texts)} samples, saved to {output_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_optimize_stage(cfg: DictConfig) -> bool:
    """
    Run the optimization stage.
    
    Args:
        cfg: Configuration
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Starting optimization stage")
    
    if not cfg.pipeline.optimize.enabled:
        logger.info("Optimization stage is disabled, skipping")
        return True
    
    try:
        # For now, this stage is just a placeholder
        # In the future, this could implement hyperparameter optimization
        logger.info("Optimization stage is not yet implemented")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def run_pipeline(cfg: DictConfig) -> int:
    """
    Run the entire pipeline.
    
    Args:
        cfg: Configuration
        
    Returns:
        0 for success, non-zero for failure
    """
    logger.info("Starting ChatGoT pipeline")
    
    # Log pipeline configuration
    pipeline_cfg = cfg.pipeline
    stages = pipeline_cfg.stages
    
    logger.info(f"Pipeline stages: {', '.join(stages)}")
    logger.info(f"Resume enabled: {pipeline_cfg.resume}")
    logger.info(f"Force restart: {pipeline_cfg.force_restart}")
    
    # Run stages in order
    results = {}
    
    for stage in stages:
        if stage == "process":
            results[stage] = run_process_stage(cfg)
        elif stage == "optimize":
            results[stage] = run_optimize_stage(cfg)
        elif stage == "train":
            results[stage] = run_train_stage(cfg)
        elif stage == "generate":
            results[stage] = run_generate_stage(cfg)
        else:
            logger.warning(f"Unknown pipeline stage: {stage}")
            results[stage] = False
        
        # Stop pipeline if a stage fails
        if not results[stage]:
            logger.error(f"Pipeline failed at stage '{stage}'")
            return 1
    
    # Log results summary
    logger.info("Pipeline completed successfully")
    for stage, result in results.items():
        logger.info(f"Stage '{stage}': {'Success' if result else 'Failed'}")
    
    return 0 