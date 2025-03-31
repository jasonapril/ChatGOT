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

# Add hydra.utils import for instantiate
import hydra.utils
# Add copy for deep copying config sections
import copy

# --- Main Execution Block --- #

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run_training(cfg: DictConfig) -> None:
    """Hydra-managed function to run the training process."""
    # --- Remove VERY EARLY DEBUG --- 
    # print("--- DEBUG: Entered run_training function ---", file=sys.stderr)
    # try:
    #     resume_val = cfg.get("resume_from", "<NOT FOUND>") # Corrected access
    #     print(f"--- DEBUG: cfg.resume_from = {resume_val} (Type: {type(resume_val)}) ---", file=sys.stderr)
    # except Exception as e:
    #     print(f"--- DEBUG: Error accessing cfg.resume_from: {e} ---", file=sys.stderr)
    # --- END VERY EARLY DEBUG ---

    # Get the logger for the current module 
    logger = logging.getLogger(__name__)
    # print(f"--- DEBUG: Logger {logger.name} effective level: {logger.getEffectiveLevel()} ---", file=sys.stderr) # Remove this too

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

    # --- Determine vocab_size from training dataset --- #
    vocab_size = None
    if hasattr(train_loader.dataset, 'vocab_size'):
        vocab_size = getattr(train_loader.dataset, 'vocab_size')
        if isinstance(vocab_size, int):
            logger.info(f"Determined vocab_size from dataset: {vocab_size}")
        else:
            logger.warning(f"Dataset has 'vocab_size' but it's not an int ({type(vocab_size)}). Cannot set automatically.")
            vocab_size = None
    else:
        logger.warning("Training dataset does not have 'vocab_size' attribute. Cannot set automatically.")
    
    if vocab_size is None:
        logger.error("Failed to determine vocab_size. Check dataset implementation and config.")
        sys.exit(1)

    # --- Build Model --- #
    log_section_header(logger, "BUILDING MODEL")
    # Inject vocab_size into model config before instantiation
    try:
        OmegaConf.set_struct(cfg.model.config, False) # Allow adding vocab_size
        cfg.model.config.vocab_size = vocab_size
        OmegaConf.set_struct(cfg.model.config, True)
        logger.info(f"Injected vocab_size={vocab_size} into model config.")
    except Exception as e:
        logger.error(f"Failed to inject vocab_size into cfg.model.config: {e}")
        sys.exit(1)

    logger.info(f"Instantiating model using hydra: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)
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
    logger.info(f"Scheduler created: {cfg.scheduler.type}")

    # --- Determine Checkpoint Path --- #
    logger.debug(f"Attempting to access resume_from config...")
    checkpoint_path_cfg = cfg.get("resume_from")
    logger.debug(f"Value of cfg.resume_from: '{checkpoint_path_cfg}' (Type: {type(checkpoint_path_cfg)})")
    absolute_checkpoint_path = None
    loaded_config_from_checkpoint = None # Initialize to None

    if checkpoint_path_cfg == "latest":
        # Call script to find latest checkpoint matching current config
        logger.info("'latest' specified for resume_from. Running script to find checkpoint...")
        script_path = os.path.join(hydra.utils.get_original_cwd(), "scripts", "get_latest_checkpoint.py")
        try:
            # Construct arguments for the script based on current config
            script_args = [
                sys.executable, # Use the current Python interpreter
                script_path,
                # Add filters based on current config
                # Ensure these keys exist in your config structure
                "--model-architecture", cfg.model.get('architecture', 'unknown'),
                "--dataset-target", cfg.data.get('train', {}).get('_target_', 'unknown'),
                "--experiment-name", cfg.get('experiment_name', 'unknown'),
            ]
            
            # Base directory for searching runs
            base_search_dir = os.path.join(original_cwd, "outputs/hydra") # Search within outputs/hydra
            script_args.extend(["--base-dir", base_search_dir])

            logger.info(f"Running script to find latest checkpoint: {' '.join(script_args)}")
            result = subprocess.run(script_args, capture_output=True, text=True, check=False) # Use check=False initially

            if result.returncode == 0:
                latest_checkpoint = result.stdout.strip()
                if latest_checkpoint and os.path.exists(latest_checkpoint): # Check if path exists
                    absolute_checkpoint_path = latest_checkpoint # Assume script returns absolute path
                    logger.info(f"Found latest checkpoint via script: {absolute_checkpoint_path}")
                elif latest_checkpoint: # Script returned path but it doesn't exist
                     logger.warning(f"Script returned path '{latest_checkpoint}' but it does not exist.")
                else: # Script succeeded but returned empty string
                    logger.warning(f"'latest' specified but script {script_path} did not return a path.")
            else: # Script failed
                logger.error(f"Script {script_path} failed with return code {result.returncode}. Cannot resume from latest.\nStderr: {result.stderr}")

        except FileNotFoundError:
            logger.error(f"Script {script_path} not found. Cannot resume from latest.")
        except Exception as e:
            logger.error(f"Unexpected error running script {script_path}: {e}", exc_info=True)

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

    # --- Instantiate Callbacks --- #
    callbacks = []
    if "callbacks" in cfg:
        # Deep copy the callbacks config section to allow modification
        callbacks_cfg = copy.deepcopy(cfg.callbacks)

        # Instantiate callbacks using the potentially modified config
        # Iterate over the list associated with the 'callbacks_list' key
        for cb_conf in callbacks_cfg.get('callbacks_list', []):
            try:
                # Inject tokenizer if it's the SampleGenerationCallback
                if isinstance(cb_conf, DictConfig) and cb_conf.get('_target_') == 'src.training.callbacks.SampleGenerationCallback':
                    # Pass the instantiated tokenizer object
                    callbacks.append(hydra.utils.instantiate(cb_conf, tokenizer=tokenizer))
                else:
                    callbacks.append(hydra.utils.instantiate(cb_conf))
                logger.info(f"Initialized callback: {cb_conf.get('_target_') if isinstance(cb_conf, DictConfig) else 'Unknown Callback Config Format'}")
            except Exception as e:
                target_name = cb_conf.get('_target_', 'Unknown') if isinstance(cb_conf, DictConfig) else 'Invalid Config'
                logger.error(f"Failed to instantiate callback {target_name}: {e}", exc_info=True)

    # --- Initialize Trainer --- #
    log_section_header(logger, "INITIALIZING TRAINER")
    # Make sure checkpoint dir exists relative to hydra output
    hydra_output_dir = os.getcwd() # Hydra sets current working dir to output dir
    checkpoint_subdir = cfg.training.get('checkpoint_subdir', 'checkpoints')
    trainer_checkpoint_dir = os.path.join(hydra_output_dir, checkpoint_subdir)

    # Get original CWD for resolving relative paths like vocab_path if needed
    original_cwd = "./" # Default if original_cwd cannot be obtained
    try:
        original_cwd = hydra.utils.get_original_cwd()
    except ValueError:
        logger.warning("Could not determine original CWD, assuming paths in config are relative to Hydra CWD or absolute.")

    # Resolve vocab path
    resolved_vocab_path = None
    # if cfg.data.vocab_path:
    #     if not os.path.isabs(cfg.data.vocab_path):
    #          resolved_vocab_path = os.path.join(original_cwd, cfg.data.vocab_path)
    #     else:
    #          resolved_vocab_path = cfg.data.vocab_path

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        epochs=cfg.training.epochs,
        config=cfg, # Pass the *current* config for now
        callbacks=callbacks, # Pass initially instantiated callbacks
        use_amp=cfg.training.use_amp,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        checkpoint_dir=trainer_checkpoint_dir, 
        log_interval=cfg.training.log_interval,
        vocab_path=resolved_vocab_path
    )
    logger.info("Trainer initialized successfully.")
    
    # --- Load Checkpoint State (if path determined earlier) --- #
    if absolute_checkpoint_path and os.path.exists(absolute_checkpoint_path):
        logger.info(f"Checkpoint found at {absolute_checkpoint_path}. Attempting to load...")
        # Now load the checkpoint using the initialized trainer
        loaded_config_from_checkpoint = trainer.load_checkpoint(absolute_checkpoint_path)
        
        if loaded_config_from_checkpoint:
            logger.info(f"Trainer state loaded from {absolute_checkpoint_path}")
            # *** Now update callbacks if needed ***
            resumed_experiment_id = loaded_config_from_checkpoint.get('experiment_id')
            if resumed_experiment_id:
                logger.info(f"Resuming experiment ID: {resumed_experiment_id}. Updating TensorBoard logger.")
                for cb in trainer.callbacks:
                    if isinstance(cb, TensorBoardLogger):
                        # NOTE: Assuming TensorBoardLogger has a method to update its log_dir or logger object
                        # If not, this might need adjustment based on the callback's implementation.
                        # For simplicity, let's assume it re-initializes its writer based on config.
                        # We might need to update the trainer's config reference as well if callbacks rely on it directly.
                        # A cleaner way might be to re-instantiate callbacks after loading config.
                        resumed_log_dir = f"outputs/tensorboard/{resumed_experiment_id}"
                        cb.log_dir = resumed_log_dir # Assuming direct modification is possible/intended
                        cb._configure_writer() # Assuming a method to reconfigure the writer
                        logger.info(f"Updated TensorBoard log_dir to: {resumed_log_dir}")
                        break
            
            # Generate a sample immediately after resuming
            logger.info("Generating initial sample after loading checkpoint...")
            trainer._generate_sample_and_log()
        else:
            logger.error("Failed to load checkpoint state, cannot resume. Starting fresh.")
            absolute_checkpoint_path = None # Prevent further attempts

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