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
    # --- Basic Logging Setup (Configure Trainer logger) ---
    # Ensure file handler captures INFO from Trainer
    log_file = "train_runner.log" # Default Hydra log file name
    # Get the specific logger used by Trainer
    trainer_logger = logging.getLogger("Trainer")
    trainer_logger.setLevel(logging.INFO) # Set level for this specific logger
    # Check if a file handler already exists from Hydra
    # This is a bit hacky, assumes Hydra adds a FileHandler to the root
    root_logger = logging.getLogger()
    file_handler = None
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler) and log_file in h.baseFilename:
            file_handler = h
            break
    # If Hydra provided a file handler, add it to Trainer logger too
    if file_handler:
        # Ensure handler level is low enough
        file_handler.setLevel(logging.INFO) 
        trainer_logger.addHandler(file_handler)
        trainer_logger.propagate = False # Prevent duplicate messages in root logger
        logging.info("Added Hydra file handler to Trainer logger.")
    else:
        logging.warning("Could not find Hydra file handler to attach to Trainer logger.")
    # --- End Basic Logging Setup ---

    # Get the logger for the current module AFTER setting up handlers
    logger = logging.getLogger(__name__)

    print("--- DEBUG: Entered run_training function ---") # Keep this for console confirmation
    logger.info("\n" + "="*80 + "\n" + "INITIAL SETUP".center(80) + "\n" + "="*80)
    # Seed for reproducibility
    set_seed(cfg.seed)

    # Setup device
    device_arg = "cpu" if cfg.force_cpu else "auto"
    device = setup_device(device_arg)

    # Save final config (after Hydra composition and potential overrides)
    # Hydra automatically saves the config in its output directory (.hydra)
    # We can optionally save it again in a more prominent location if desired.
    # config_save_path = os.path.join(cfg.hydra.run.dir, "resolved_config.yaml")
    # with open(config_save_path, 'w') as f:
    #     f.write(OmegaConf.to_yaml(cfg))
    # logger.info(f"Final resolved configuration saved to {config_save_path}")

    log_section_header(logger, "DATA PREPARATION") # Pass logger object

    # Get Hydra's current working directory (output directory)
    # hydra_output_dir = os.getcwd() # INCORRECT: This is the launch directory
    hydra_output_dir = HydraConfig.get().runtime.output_dir # CORRECT way
    logger.info(f"Hydra output directory: {hydra_output_dir}")
    # Get original working directory BEFORE hydra changes it
    try:
        original_cwd = hydra.utils.get_original_cwd()
        logger.info(f"Original working directory: {original_cwd}")
    except ValueError:
        logger.warning("Could not determine original CWD. Assuming it's the same as the current CWD.")
        original_cwd = hydra_output_dir # Fallback, might be incorrect if Hydra changed dir

    # Construct and ensure checkpoint directory
    checkpoint_dir = os.path.join(hydra_output_dir, "checkpoints")
    ensure_directory(checkpoint_dir)
    logger.info(f"Checkpoint directory ensured: {checkpoint_dir}")

    # --- Data Loading --- #
    log_section_header(logger, "LOADING DATA")
    # Remove separate dataset creation
    # train_dataset = create_dataset_from_config(cfg.data)
    # val_dataset = None # Placeholder
    # logger.info(f"Training dataset created: {type(train_dataset).__name__}")
    # if val_dataset:
    #     logger.info(f"Validation dataset created: {type(val_dataset).__name__}")

    # Create DataLoaders using the factory function from base.py
    # This assumes prepare_dataloaders_from_config handles dataset creation internally
    # and returns (train_loader, val_loader, test_loader)
    logger.info("Preparing dataloaders...")
    try:
        train_dataloader, val_dataloader, _ = prepare_dataloaders_from_config(
            data_config=cfg.data,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            original_cwd=original_cwd # Pass original CWD
        )
        logger.info("Dataloaders prepared.")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True)
        sys.exit(1)

    # Check if dataloaders were created successfully
    if train_dataloader:
        logger.info(f"Train DataLoader created successfully.")
    else:
        logger.error("Failed to create Train DataLoader.")
        sys.exit(1)

    if val_dataloader:
        logger.info(f"Validation DataLoader created successfully.")
    else:
        logger.info("No Validation DataLoader created (this might be intended based on config).")

    # Remove old separate dataloader creation calls
    # train_dataloader = create_dataloader(train_dataset, cfg.data, split='train')
    # val_dataloader = create_dataloader(val_dataset, cfg.data, split='validation') if val_dataset else None
    # logger.info(f"Train DataLoader created with batch size {cfg.data.batch_size}")
    # if val_dataloader:
    #     logger.info(f"Validation DataLoader created with batch size {cfg.data.batch_size}")

    # --- Model Creation --- #
    log_section_header(logger, "CREATING MODEL")
    # Pass the model sub-config directly (OmegaConf DictConfig works)
    model = create_model_from_config(cfg.model)
    logger.info(f"Model created: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # Apply activation checkpointing if requested
    if cfg.training.activation_checkpointing:
         configure_activation_checkpointing(model)

    model.to(device)

    # --- Optimizer and Scheduler --- #
    log_section_header(logger, "SETTING UP OPTIMIZER & SCHEDULER")
    # TODO: Implement flexible optimizer/scheduler creation using factories/registry
    if cfg.optimizer.type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    # Add other optimizer types here
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimizer.type}")
    logger.info(f"Optimizer created: {type(optimizer).__name__}")

    scheduler = None
    if cfg.scheduler:
        if cfg.scheduler.type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            # Calculate total steps
            steps_per_epoch = len(train_dataloader) // cfg.training.gradient_accumulation_steps
            total_steps = steps_per_epoch * cfg.training.epochs
            if total_steps <= 0:
                 logger.warning(f"Calculated total_steps <= 0 ({total_steps}). Disabling scheduler.")
            else:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=cfg.scheduler.warmup_steps,
                    num_training_steps=total_steps
                )
                logger.info(f"Scheduler created: Cosine with {cfg.scheduler.warmup_steps} warmup steps, {total_steps} total steps")
        # Add other scheduler types here (linear, onecycle etc.)
        elif cfg.scheduler.type:
             logger.warning(f"Scheduler type '{cfg.scheduler.type}' not implemented or invalid, proceeding without scheduler.")

    # --- Callbacks --- #
    log_section_header(logger, "SETTING UP CALLBACKS")
    callbacks = []
    # TODO: Instantiate callbacks based on cfg.callbacks using hydra.utils.instantiate
    logger.info(f"Initialized {len(callbacks)} callbacks.")


    # --- Trainer Initialization --- #
    log_section_header(logger, "INITIALIZING TRAINER")
    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            scheduler=scheduler,
            device=device,
            epochs=cfg.training.epochs,
            config=OmegaConf.to_container(cfg.training, resolve=True), # Pass training config as dict
            callbacks=callbacks,
            use_amp=cfg.training.use_amp,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            checkpoint_dir=checkpoint_dir, # Pass constructed checkpoint_dir path
            log_interval=cfg.training.log_interval,
            vocab_path=cfg.data.vocab_path # Pass vocab path
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Trainer: {e}", exc_info=True)
        sys.exit(1)

    # --- Checkpoint Loading (Resume) --- #
    checkpoint_path_cfg = cfg.resume_from
    absolute_checkpoint_path = None
    if checkpoint_path_cfg == "latest":
        latest_path = get_latest_checkpoint(checkpoint_dir) # Use constructed checkpoint_dir
        if latest_path:
             logger.info(f"Found latest checkpoint: {latest_path}")
             absolute_checkpoint_path = os.path.abspath(latest_path)
        else:
             logger.warning(f"Resume set to 'latest' but no checkpoint found in {checkpoint_dir}. Starting fresh.") # Use constructed checkpoint_dir
    elif checkpoint_path_cfg:
         # Handle potentially relative path from original CWD
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

    # DEBUG: Log the resolved path before checking existence
    if absolute_checkpoint_path:
        logger.info(f"DEBUG: Resolved checkpoint path to check: {absolute_checkpoint_path}")
        logger.info(f"DEBUG: Does path exist? {os.path.exists(absolute_checkpoint_path)}")

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
        logger.info("No checkpoint specified or found. Starting fresh training.") # Add log for clarity

    # --- Optional: Torch Compile --- #
    if cfg.training.compile_model:
        log_section_header(logger, "COMPILING MODEL")
        logger.info("Attempting to compile model with torch.compile()...")
        try:
            # Note: compilation might happen inside Trainer if needed later
            trainer.model = torch.compile(trainer.model)
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.error(f"torch.compile failed: {e}", exc_info=True)
            # Decide whether to continue without compilation or exit
            logger.warning("Proceeding without model compilation.")

    # --- Start Training --- #
    log_section_header(logger, "STARTING TRAINING")
    try:
        training_metrics = trainer.train()
        logger.info("Training completed successfully.")
        # Save final metrics
        metrics_path = os.path.join(hydra_output_dir, "training_metrics.json") # Use hydra_output_dir
        save_json(training_metrics, metrics_path)
        logger.info(f"Training metrics saved to {metrics_path}")

    except Exception as e:
        logger.error("An error occurred during training:", exc_info=True)
        # Potentially save a final checkpoint on error?
        # error_checkpoint_path = os.path.join(checkpoint_dir, 'error_checkpoint.pt') # Use constructed checkpoint_dir
        # logger.info(f"Attempting to save error checkpoint to {error_checkpoint_path}")
        # trainer.save_checkpoint(error_checkpoint_path)
        sys.exit(1) # Exit with error code

    log_section_header(logger, "TRAINING FINISHED")
    # --- Optional: Final Evaluation / Text Generation --- #

if __name__ == "__main__":
    run_training() # Call the hydra-decorated function

    # Placeholder for the removed main function
    # main()

    # Placeholder for the removed TrainRunner class
    # train_runner = TrainRunner(config_path="path_to_config.json")
    # train_runner.run()

    # Placeholder for the removed main function
    # main() 