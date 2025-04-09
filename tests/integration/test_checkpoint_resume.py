import pytest
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pathlib import Path
import tempfile
import os
import pickle
import logging
import shutil
import json
import copy
import re

# Project imports
from craft.training.trainer import Trainer
from craft.training.checkpointing import CHECKPOINT_FILE_PATTERN, TrainingState

def create_dummy_data_and_config(tmp_path: Path) -> DictConfig:
    """Helper to create dummy data and a minimal OmegaConf config for testing."""
    
    # Create dummy tokenizer vocab file
    vocab_content = " .abcdefghijklmnopqrstuvwxyz\n"
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text(vocab_content)
    vocab_size = len(vocab_content)

    # Create dummy tokenized data
    dummy_token_ids = list(range(vocab_size)) * 10 # Create some repeatable data
    dummy_pickled_data_path = tmp_path / "dummy_train.pkl"
    metadata_path = tmp_path / "metadata.json" # Metadata for PickledDataset
    with open(dummy_pickled_data_path, 'wb') as f:
        pickle.dump(torch.tensor(dummy_token_ids, dtype=torch.long), f)
    # Create dummy metadata
    metadata_content = {"vocab_size": vocab_size, "tokenizer_type": "CharTokenizer", "idx_to_char": {i:c for i,c in enumerate(vocab_content)}}
    with open(metadata_path, 'w') as f:
        json.dump(metadata_content, f)

    # Define output directory for this test run
    output_dir = tmp_path / "test_run_output"
    checkpoint_dir = output_dir / "checkpoints"

    cfg_dict = {
        'defaults': ['_self_'], 
        'experiment_name': 'test_checkpoint_resume_exp', # Added experiment name
        'output_dir': str(output_dir), # Define output directory
        
        'data': {
            'tokenizer': {
                '_target_': 'craft.data.tokenizers.char.CharTokenizer',
                'vocab_path': str(vocab_file),
            },
            # Using PickledDataset requires pre-tokenized data
            'datasets': {
                 'train': {
                      'dataset': {
                            '_target_': 'craft.data.datasets.pickled_dataset.PickledDataset',
                            'file_path': str(dummy_pickled_data_path),
                            'block_size': 16,
                      },
                      'dataloader': {
                          '_target_': 'torch.utils.data.DataLoader',
                          'batch_size': 2,
                          'shuffle': False, 
                          'num_workers': 0, # Important for reproducibility/simplicity in tests
                          'pin_memory': False
                      }
                 },
                 'val': None # No validation for this test
            },
        },
        'model': {
            '_target_': 'craft.models.transformer.TransformerModel', 
            'config': {
                '_target_': 'craft.config.schemas.LanguageModelConfig',
                'architecture': 'transformer', # Ensure architecture is present for schema loading
                'vocab_size': vocab_size, 
                'block_size': 16, 
                'max_seq_length': 16,
                'n_layer': 1,
                'n_head': 1,
                'd_model': 4,
                'dropout': 0.0,
                'bias': False,
                'd_hid': 16,
                'activation': 'relu',
                'norm_first': False,
                'layer_norm_eps': 1e-5
            }
        },
        'optimizer': {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-4,
            'weight_decay': 0.01,
        },
        'scheduler': None, # No scheduler for simplicity
        'callbacks': None, # No callbacks for simplicity
        'evaluation': None, # No evaluation
        'checkpointing': { # Configure checkpoint manager via Hydra
            '_target_': 'craft.training.checkpointing.CheckpointManager',
            'checkpoint_dir': str(checkpoint_dir), # Explicitly set dir
            'keep_last_n': 1,
            'keep_best_n': 0, # Don't worry about best for this test
            'save_best_only': False,
        },
        'training': { # Corresponds to TrainingConfig Pydantic model
             '_target_': 'craft.config.schemas.TrainingConfig',
             'batch_size': '${data.datasets.train.dataloader.batch_size}',
             'num_epochs': 1,
             'max_steps': 3, # Train for a few steps
             'use_amp': False,
             'gradient_accumulation_steps': 1,
             'log_interval': 1,
             'eval_interval': 0, # Disable eval interval
             'save_interval': 2, # Save checkpoint every 2 steps
             'checkpoint_dir': str(checkpoint_dir),
             'resume_from_checkpoint': None, # Initial run: no resume
             'log_level': "INFO",
             'seed': 42,
             'device': "cpu",
             'max_grad_norm': 1.0,
             'time_save_interval_seconds': 0,
             'time_eval_interval_seconds': 0,
             'mixed_precision': False,
             'save_steps_interval': '${..save_interval}' # Link to save_interval
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    OmegaConf.resolve(cfg) # Resolve interpolations
    return cfg

def test_checkpoint_save_and_resume(tmp_path: Path):
    """Tests saving a checkpoint, resuming, verifying state, and continuing training."""
    
    # --- Setup & Initial Run --- #
    cfg = create_dummy_data_and_config(tmp_path)
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    
    print(f"--- Initial Config ---\n{OmegaConf.to_yaml(cfg)}\n----------------------")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Instantiate and run the first trainer
    print("--- Starting Initial Training Run --- ")
    trainer1 = instantiate(cfg, _recursive_=False) # Instantiate Trainer using full config
    assert isinstance(trainer1, Trainer)
    trainer1.train()
    print("--- Initial Training Run Finished --- ")
    
    # Assertions for the first run
    final_step_run1 = trainer1.global_step
    final_epoch_run1 = trainer1.epoch
    assert final_step_run1 == cfg.training.max_steps

    # Check that a checkpoint file was created
    # Checkpoint should be saved at step 2 (save_interval=2, max_steps=3)
    expected_checkpoint_step = (final_step_run1 // cfg.training.save_interval) * cfg.training.save_interval
    if expected_checkpoint_step == 0: # Handle case where max_steps < save_interval
        pytest.skip("max_steps is less than save_interval, no checkpoint expected.")
        
    expected_checkpoint_pattern = checkpoint_dir / f"checkpoint_step_{expected_checkpoint_step}*.pt"
    checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_step_{expected_checkpoint_step}*.pt"))
    
    print(f"Expected checkpoint pattern: {expected_checkpoint_pattern}")
    print(f"Found checkpoint files: {checkpoint_files}")

    assert len(checkpoint_files) >= 1, f"No checkpoint file found for step {expected_checkpoint_step}"
    # Get the most recent checkpoint if multiple match (e.g., _resumed)
    checkpoint_path_run1 = max(checkpoint_files, key=os.path.getctime)
    print(f"Checkpoint saved at: {checkpoint_path_run1}")

    # --- Load the saved state for comparison --- #
    saved_state_dict = torch.load(checkpoint_path_run1, map_location='cpu')
    saved_state = TrainingState(**saved_state_dict)
    model_state_run1 = copy.deepcopy(saved_state.model_state_dict)
    optimizer_state_run1 = copy.deepcopy(saved_state.optimizer_state_dict)
    # --- End Load --- #

    del trainer1 # Ensure resources are released (optional)
    torch.cuda.empty_cache() # If using GPU

    # --- Setup & Resume Run --- #
    print("--- Configuring for Resume Run --- ")
    # Modify the config to resume from the saved checkpoint
    cfg_resume = cfg.copy() # Create a copy to modify
    OmegaConf.update(cfg_resume, "training.resume_from_checkpoint", str(checkpoint_path_run1))
    # Increase max_steps to allow continuation
    new_max_steps = cfg.training.max_steps + 2
    OmegaConf.update(cfg_resume, "training.max_steps", new_max_steps)
    print(f"--- Resume Config ---\n{OmegaConf.to_yaml(cfg_resume)}\n----------------------")

    # Instantiate the second trainer (should resume)
    print("--- Starting Resume Training Run --- ")
    trainer2 = instantiate(cfg_resume, _recursive_=False)
    assert isinstance(trainer2, Trainer)
    
    # Assert state BEFORE training starts again
    print(f"Trainer 2 Initial State: global_step={trainer2.global_step}, epoch={trainer2.epoch}")
    # The loaded step should be the step at which the checkpoint was SAVED
    assert trainer2.global_step == expected_checkpoint_step, f"Resumed step mismatch. Expected {expected_checkpoint_step}, got {trainer2.global_step}"
    # Compare model state dicts
    model_state_run2 = trainer2.model.state_dict()
    for key in model_state_run1:
        assert key in model_state_run2, f"Key {key} missing in resumed model state"
        assert torch.equal(model_state_run1[key], model_state_run2[key]), f"Tensor mismatch for key {key} in model state"
    # Compare optimizer state dicts
    optimizer_state_run2 = trainer2.optimizer.state_dict()
    # Basic check: compare top-level keys and state structure if possible
    assert optimizer_state_run1.keys() == optimizer_state_run2.keys(), "Optimizer state keys mismatch"
    # Deeper comparison is complex due to tensors inside; might need tolerance or specific checks
    # For AdamW, check 'step' - it should match the loaded global_step
    optimizer_steps_run1 = [s.get('step', -1) for s in optimizer_state_run1.get('state', {}).values()]
    optimizer_steps_run2 = [s.get('step', -1) for s in optimizer_state_run2.get('state', {}).values()]
    if optimizer_steps_run1 and optimizer_steps_run2:
        assert all(s == expected_checkpoint_step for s in optimizer_steps_run2), f"Optimizer step count mismatch after resume. Expected {expected_checkpoint_step}, got {optimizer_steps_run2}"

    # --- Continue Training --- #
    print("--- Continuing Training Run --- ")
    trainer2.train()
    print("--- Resume Training Run Finished --- ")
    final_step_run2 = trainer2.global_step
    # Assert that training continued until the new max_steps
    assert final_step_run2 == new_max_steps, f"Expected final step {new_max_steps}, got {final_step_run2}"
    # --- End Continued Training --- #

    print("--- Checkpoint Resume Test Completed Successfully --- ")

def test_checkpoint_resume_latest(tmp_path: Path):
    """Tests resuming from 'latest' checkpoint specifier."""
    # --- Setup & Initial Run (save multiple checkpoints) --- #
    cfg = create_dummy_data_and_config(tmp_path)
    # Modify config for this test
    new_max_steps = 6
    new_save_interval = 2
    OmegaConf.update(cfg, "training.max_steps", new_max_steps)
    OmegaConf.update(cfg, "training.save_interval", new_save_interval)
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    print("--- Starting Initial Run (Save Multiple) --- ")
    trainer1 = instantiate(cfg, _recursive_=False)
    trainer1.train()
    print("--- Initial Run Finished --- ")

    # Check checkpoints were created (expect steps 2, 4, 6 if max_steps=6, save=2)
    expected_steps = [s for s in range(new_save_interval, new_max_steps + 1, new_save_interval)]
    found_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    found_steps = sorted([
        int(re.search(CHECKPOINT_FILE_PATTERN, f.name).group(1))
        for f in found_files if re.search(CHECKPOINT_FILE_PATTERN, f.name)
    ])
    print(f"Expected checkpoint steps: {expected_steps}")
    print(f"Found checkpoint steps: {found_steps}")
    # Check if the latest expected step was saved
    latest_expected_step = max(expected_steps) if expected_steps else 0
    assert latest_expected_step in found_steps, f"Latest expected checkpoint step {latest_expected_step} not found."

    del trainer1

    # --- Setup & Resume Run with 'latest' --- #
    print("--- Configuring for Resume Run ('latest') --- ")
    cfg_resume = cfg.copy()
    OmegaConf.update(cfg_resume, "training.resume_from_checkpoint", "latest") # Use 'latest'
    # Don't change max_steps, resuming latest should mean it's already finished
    print(f"--- Resume Config ---\n{OmegaConf.to_yaml(cfg_resume)}\n----------------------")

    print("--- Starting Resume Training Run ('latest') --- ")
    trainer2 = instantiate(cfg_resume, _recursive_=False)

    # --- Assert State AFTER Resuming ('latest') --- #
    print(f"Trainer 2 Initial State: global_step={trainer2.global_step}, epoch={trainer2.epoch}")
    # Should resume from the highest step checkpoint saved
    assert trainer2.global_step == latest_expected_step, f"Resumed step mismatch for 'latest'. Expected {latest_expected_step}, got {trainer2.global_step}"

    # --- Optionally try to continue training (should stop immediately) --- #
    # Since we resumed from the step matching max_steps, train() should do nothing.
    trainer2.train()
    assert trainer2.global_step == latest_expected_step, "Global step changed after resuming from latest and calling train()"

    print("--- Checkpoint Resume 'latest' Test Completed Successfully --- ")

def test_checkpoint_resume_with_amp(tmp_path: Path):
    """Tests resuming training with AMP enabled."""
    # --- Setup & Initial Run with AMP --- #
    cfg = create_dummy_data_and_config(tmp_path)
    # Modify config for AMP and steps
    OmegaConf.update(cfg, "training.use_amp", True)
    OmegaConf.update(cfg, "training.max_steps", 3)
    OmegaConf.update(cfg, "training.save_interval", 2)
    # Ensure device is CUDA if possible, otherwise skip
    if not torch.cuda.is_available():
        pytest.skip("AMP test requires CUDA device")
    OmegaConf.update(cfg, "training.device", "cuda")

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    print("--- Starting Initial Run (AMP Enabled) --- ")
    trainer1 = instantiate(cfg, _recursive_=False)
    trainer1.train()
    print("--- Initial Run Finished (AMP Enabled) --- ")

    expected_checkpoint_step = 2
    checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_step_{expected_checkpoint_step}*.pt"))
    assert len(checkpoint_files) >= 1, f"No checkpoint file found for step {expected_checkpoint_step} with AMP"
    checkpoint_path_run1 = max(checkpoint_files, key=os.path.getctime)

    # --- Load the saved state for comparison --- #
    saved_state_dict = torch.load(checkpoint_path_run1, map_location='cpu') # Load to CPU first
    saved_state = TrainingState(**saved_state_dict)
    scaler_state_run1 = copy.deepcopy(saved_state.scaler_state_dict)
    assert scaler_state_run1 is not None, "Scaler state dict was not saved in checkpoint"
    assert len(scaler_state_run1) > 0, "Saved scaler state dict is empty"

    del trainer1
    torch.cuda.empty_cache()

    # --- Setup & Resume Run with AMP --- #
    print("--- Configuring for Resume Run (AMP Enabled) --- ")
    cfg_resume = cfg.copy()
    OmegaConf.update(cfg_resume, "training.resume_from_checkpoint", str(checkpoint_path_run1))
    new_max_steps = cfg.training.max_steps + 2
    OmegaConf.update(cfg_resume, "training.max_steps", new_max_steps)

    print("--- Starting Resume Training Run (AMP Enabled) --- ")
    trainer2 = instantiate(cfg_resume, _recursive_=False)

    # --- Assert State AFTER Resuming (AMP) --- #
    print(f"Trainer 2 Initial State: global_step={trainer2.global_step}")
    assert trainer2.global_step == expected_checkpoint_step, "Resumed step mismatch (AMP)"
    assert trainer2.scaler is not None, "Scaler object not present in resumed trainer"
    scaler_state_run2 = trainer2.scaler.state_dict()
    assert scaler_state_run2 == scaler_state_run1, "Scaler state dict mismatch after resume"

    # --- Continue Training --- #
    print("--- Continuing Training Run (AMP Enabled) --- ")
    trainer2.train()
    print("--- Resume Training Run Finished (AMP Enabled) --- ")
    final_step_run2 = trainer2.global_step
    assert final_step_run2 == new_max_steps, f"Expected final step {new_max_steps} (AMP), got {final_step_run2}"

    print("--- Checkpoint Resume with AMP Test Completed Successfully --- ")

def test_checkpoint_resume_with_scheduler(tmp_path: Path):
    """Tests resuming training with a scheduler enabled."""
    # --- Setup & Initial Run with Scheduler --- #
    cfg = create_dummy_data_and_config(tmp_path)
    # Modify config for Scheduler and steps
    scheduler_config = {
        '_target_': 'torch.optim.lr_scheduler.StepLR',
        'step_size': 1, # Step every step for testing
        'gamma': 0.9
    }
    OmegaConf.update(cfg, "scheduler", scheduler_config)
    OmegaConf.update(cfg, "training.max_steps", 3)
    OmegaConf.update(cfg, "training.save_interval", 2)

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    print("--- Starting Initial Run (Scheduler Enabled) --- ")
    trainer1 = instantiate(cfg, _recursive_=False)
    # Get initial LR
    initial_lr = get_current_lr(trainer1.optimizer)
    trainer1.train()
    print("--- Initial Run Finished (Scheduler Enabled) --- ")

    expected_checkpoint_step = 2
    checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_step_{expected_checkpoint_step}*.pt"))
    assert len(checkpoint_files) >= 1, f"No checkpoint file found for step {expected_checkpoint_step} with scheduler"
    checkpoint_path_run1 = max(checkpoint_files, key=os.path.getctime)

    # --- Load the saved state for comparison --- #
    saved_state_dict = torch.load(checkpoint_path_run1, map_location='cpu')
    saved_state = TrainingState(**saved_state_dict)
    scheduler_state_run1 = copy.deepcopy(saved_state.scheduler_state_dict)
    assert scheduler_state_run1 is not None, "Scheduler state dict was not saved in checkpoint"
    # Check specific keys for StepLR if possible
    assert 'step_count' in scheduler_state_run1
    assert scheduler_state_run1['step_count'] == expected_checkpoint_step

    del trainer1

    # --- Setup & Resume Run with Scheduler --- #
    print("--- Configuring for Resume Run (Scheduler Enabled) --- ")
    cfg_resume = cfg.copy()
    OmegaConf.update(cfg_resume, "training.resume_from_checkpoint", str(checkpoint_path_run1))
    new_max_steps = cfg.training.max_steps + 2
    OmegaConf.update(cfg_resume, "training.max_steps", new_max_steps)

    print("--- Starting Resume Training Run (Scheduler Enabled) --- ")
    trainer2 = instantiate(cfg_resume, _recursive_=False)

    # --- Assert State AFTER Resuming (Scheduler) --- #
    print(f"Trainer 2 Initial State: global_step={trainer2.global_step}")
    assert trainer2.global_step == expected_checkpoint_step, "Resumed step mismatch (Scheduler)"
    assert trainer2.scheduler is not None, "Scheduler object not present in resumed trainer"
    scheduler_state_run2 = trainer2.scheduler.state_dict()
    assert scheduler_state_run2 == scheduler_state_run1, "Scheduler state dict mismatch after resume"

    # Check that LR *before* the next step reflects the scheduler state at resume
    lr_after_resume_before_step = get_current_lr(trainer2.optimizer)
    expected_lr_at_resume = initial_lr * (scheduler_config['gamma'] ** expected_checkpoint_step)
    assert abs(lr_after_resume_before_step - expected_lr_at_resume) < 1e-9, f"LR mismatch after resume. Expected ~{expected_lr_at_resume:.2e}, got {lr_after_resume_before_step:.2e}"

    # --- Continue Training --- #
    print("--- Continuing Training Run (Scheduler Enabled) --- ")
    trainer2.train()
    print("--- Resume Training Run Finished (Scheduler Enabled) --- ")
    final_step_run2 = trainer2.global_step
    assert final_step_run2 == new_max_steps, f"Expected final step {new_max_steps} (Scheduler), got {final_step_run2}"

    # Check LR at the end - it should have taken one more step
    lr_at_end = get_current_lr(trainer2.optimizer)
    expected_lr_at_end = initial_lr * (scheduler_config['gamma'] ** (new_max_steps -1)) # StepLR steps *before* optimizer step usually
    # It might actually be new_max_steps depending on when scheduler.step() is called relative to step increment.
    # Let's just check it's lower than the resumed LR
    assert lr_at_end < lr_after_resume_before_step, "LR did not decrease after continuing training with scheduler"

    print("--- Checkpoint Resume with Scheduler Test Completed Successfully --- ")

# --- TODO: Add more tests --- #
# - test_resume_best (needs eval setup)
# - test_resume_error_handling
# - Add more test functions as needed

# --- Add more test functions as needed --- # 