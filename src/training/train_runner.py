#!/usr/bin/env python
"""
Training Runner Module
=========================

This module provides the main entry point for model training with all optimizations.
It coordinates the entire training process including:

1. Data loading and preparation
2. Model creation and optimization
3. Training loop execution
4. Checkpoint saving and loading
5. Progress monitoring and reporting

Usage:
    python -m src.training.train_runner --data_path processed_data/got_char_data.pkl
"""

import argparse
import logging
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional, List
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
import subprocess
import json

# Hydra and OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig # Import HydraConfig

# Import local modules
from src.models.factory import create_model_from_config
from src.data.base import prepare_dataloaders_from_config
from src.utils import (
    set_seed,
    setup_device,
    get_latest_checkpoint,
    create_output_dir,
    setup_logging,
    log_section_header,
    format_time,
    ensure_directory,
    load_json,
    save_json
)
# Import force_flush_logs directly from its module
from src.utils.logging import force_flush_logs
from src.training.optimizations import (
    setup_mixed_precision,
    setup_torch_compile,
    configure_activation_checkpointing
)
from src.training.trainer import Trainer

# --- Main Execution Block --- #

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_training(cfg: DictConfig) -> None:
    """Hydra-managed function to run the training process."""
    
    # Get the logger for the current module 
    # Hydra configures the root logger, so getting __name__ logger should inherit handlers
    logger = logging.getLogger(__name__)

    # --- Save Experiment Metadata --- #
    try:
        # Use HydraConfig to get output directory reliably
        output_dir = HydraConfig.get().runtime.output_dir 
        metadata = {
            # Use .get() for safety in case keys are missing
            "model_architecture": cfg.model.get('architecture', 'unknown'),
            "dataset_target": cfg.data.get('train', {}).get('_target_', 'unknown'),
            "experiment_name": cfg.get('experiment_name', 'unnamed'),
            "start_timestamp": datetime.now().isoformat()
        }
        metadata_path = os.path.join(output_dir, "experiment_info.json")
        save_json(metadata, metadata_path)
        logger.info(f"Saved experiment metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save experiment metadata: {e}", exc_info=True)
    # --- End Save Metadata --- #

    logger.info("\n" + "="*80 + "\n" + "INITIAL SETUP".center(80) + "\n" + "="*80)
    # Seed for reproducibility
    set_seed(cfg.seed)

    # Setup device
    device_arg = "cpu" if cfg.force_cpu else "auto"
    device = setup_device(device_arg)

    # --- Prepare DataLoaders --- #
    log_section_header(logger, "DATA PREPARATION")
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Hydra output directory: {hydra_output_dir}")
    original_cwd = hydra.utils.get_original_cwd()
    logger.info(f"Original working directory: {original_cwd}")
    train_loader, val_loader, test_loader = prepare_dataloaders_from_config(
        data_config=cfg.data, 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers, 
        original_cwd=original_cwd
    )
    if train_loader is None:
        logger.error("Training dataloader could not be created. Exiting.")
        sys.exit(1)

    # --- Build Model --- #
    log_section_header(logger, "BUILDING MODEL")
    model = create_model_from_config(cfg.model)
    logger.info(f"Model created: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    if cfg.training.activation_checkpointing:
         configure_activation_checkpointing(model)
    model.to(device)
    
    # --- Optimizer and Scheduler --- #
    log_section_header(logger, "SETTING UP OPTIMIZER & SCHEDULER")
    if cfg.optimizer.type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimizer.type}")
    logger.info(f"Optimizer created: {type(optimizer).__name__}")
    scheduler = None
    if cfg.scheduler:
        if cfg.scheduler.type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            steps_per_epoch = len(train_loader) // cfg.training.gradient_accumulation_steps
            total_steps = steps_per_epoch * cfg.training.epochs
            if total_steps <= 0:
                 logger.warning(f"Calculated total_steps <= 0 ({total_steps}). Check epochs/dataloader/grad_accum. Disabling scheduler.")
            else:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=cfg.scheduler.warmup_steps,
                    num_training_steps=total_steps
                )
                logger.info(f"Scheduler created: Cosine with {cfg.scheduler.warmup_steps} warmup steps, {total_steps} total steps")
        else:
             logger.warning(f"Unsupported scheduler type: {cfg.scheduler.type}. No scheduler will be used.")

    # --- Callbacks --- #
    log_section_header(logger, "SETTING UP CALLBACKS")
    callbacks = []
    if cfg.callbacks:
        for cb_conf in cfg.callbacks.values():
            try:
                callbacks.append(hydra.utils.instantiate(cb_conf))
            except Exception as e:
                 logger.error(f"Error instantiating callback {cb_conf.get('_target_')}: {e}", exc_info=True)
    logger.info(f"Initialized {len(callbacks)} callbacks.")

    # --- Trainer Initialization --- #
    log_section_header(logger, "INITIALIZING TRAINER")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        # Pass training config args directly
        epochs=cfg.training.epochs,
        config=cfg.training, # Pass sub-config for other params like time_save_interval
        callbacks=callbacks,
        use_amp=cfg.training.use_amp,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        # Construct checkpoint dir from hydra output + sub-config
        checkpoint_dir=os.path.join(hydra_output_dir, cfg.training.get('checkpoint_subdir', 'checkpoints')), 
        log_interval=cfg.training.log_interval,
        # Pass resolved absolute vocab_path
        vocab_path=os.path.join(original_cwd, cfg.data.vocab_path) if cfg.data.vocab_path and not os.path.isabs(cfg.data.vocab_path) else cfg.data.vocab_path 
    )
    logger.info("Trainer initialized successfully.")

    # --- Checkpoint Loading (Resume) --- #
    checkpoint_path_cfg = cfg.get('resume_from')
    absolute_checkpoint_path = None

    if checkpoint_path_cfg == "latest":
        # Call script to find latest checkpoint matching current config
        script_path = "scripts/get_latest_checkpoint.py"
        try:
            # Convert relevant parts of config to JSON string for script argument
            # Filter sensitive or overly complex parts if necessary
            config_subset = OmegaConf.to_container(cfg, resolve=True) 
            # Select keys relevant for matching checkpoints (e.g., model type, dataset)
            match_keys = ['model', 'data', 'experiment_name']
            match_config = {k: config_subset.get(k) for k in match_keys if k in config_subset}
            config_json = json.dumps(match_config)
            
            result = subprocess.run([
                sys.executable, # Use the current Python interpreter
                script_path,
                "--config_json", config_json,
                "--base_dir", os.path.join(original_cwd, "outputs") # Base search dir
            ], capture_output=True, text=True, check=True)
            
            latest_checkpoint = result.stdout.strip()
            if latest_checkpoint:
                absolute_checkpoint_path = latest_checkpoint # Script should return absolute path
                logger.info(f"Found latest checkpoint: {absolute_checkpoint_path}")
            else:
                logger.warning(f"'latest' specified but script {script_path} did not return a path.")
        except FileNotFoundError:
            logger.error(f"Script {script_path} not found. Cannot resume from latest.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running script {script_path}: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error running script {script_path}: {e}")

    elif checkpoint_path_cfg:
         # Handle explicitly provided path (relative or absolute)
         if not os.path.isabs(checkpoint_path_cfg):
             # Get original working directory if Hydra changed it
             try:
                original_cwd = hydra.utils.get_original_cwd()
                absolute_checkpoint_path = os.path.join(original_cwd, checkpoint_path_cfg)
             except ValueError:
                 logger.warning("Could not determine original CWD, assuming checkpoint path is relative to current (Hydra) CWD.")
                 absolute_checkpoint_path = os.path.abspath(checkpoint_path_cfg)
         else:
             absolute_checkpoint_path = checkpoint_path_cfg

    if absolute_checkpoint_path and os.path.exists(absolute_checkpoint_path):
        logger.info(f"Checkpoint found at {absolute_checkpoint_path}. Attempting to load...")
        trainer.load_checkpoint(absolute_checkpoint_path)
        # Generate a sample immediately after resuming
        logger.info("Generating initial sample after loading checkpoint...")
        trainer._generate_sample_and_log()
    elif checkpoint_path_cfg and not absolute_checkpoint_path:
        pass # Already logged warning if 'latest' failed
    elif checkpoint_path_cfg:
        logger.warning(f"Specified checkpoint path {checkpoint_path_cfg} (resolved to {absolute_checkpoint_path}) not found. Starting fresh.")
    else:
        logger.info("No checkpoint specified or found. Starting fresh training.") 

    # --- Optional: Torch Compile --- #
    if cfg.training.torch_compile:
        logger.info("Applying torch.compile() to the model...")
        setup_torch_compile(model)

    # --- Training --- #
    log_section_header(logger, "STARTING TRAINING")
    try:
        # Start training
        start_time = time.time()
        training_metrics = trainer.train()
        end_time = time.time()
        total_training_time = end_time - start_time
        logger.info(f"Training finished in {format_time(total_training_time)}.")
        logger.info(f"Final Training Metrics: {training_metrics}")

        # --- Optional: Evaluation --- #
        if val_loader:
            log_section_header(logger, "STARTING EVALUATION (Validation Set)")
            eval_metrics = trainer.evaluate(val_loader)
            logger.info(f"Final Validation Metrics: {eval_metrics}")
        else:
            logger.info("No validation set provided, skipping final evaluation.")
        
        # Save final model explicitly? (Optional, trainer might save best/last)
        # final_model_path = os.path.join(hydra_output_dir, "final_model.pt")
        # torch.save(model.state_dict(), final_model_path)
        # logger.info(f"Final model state saved to {final_model_path}")

    except Exception as e:
        logger.error("An error occurred during training:", exc_info=True)
        # Force flush logs in case of crash
        force_flush_logs()
        sys.exit(1) # Exit with error code
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        # Force flush logs after interruption
        force_flush_logs()
        sys.exit(130) # Standard exit code for Ctrl+C
    finally:
        # Ensure logs are flushed even on normal exit
        force_flush_logs()

    logger.info("\n" + "="*80 + "\n" + "TRAINING COMPLETE".center(80) + "\n" + "="*80)

# Main execution guard
if __name__ == "__main__":
    run_training()

    # Placeholder for the removed main function
    # main()

    # Placeholder for the removed TrainRunner class
    # train_runner = TrainRunner(config_path="path_to_config.json")
    # train_runner.run()

    # Placeholder for the removed main function
    # main() 