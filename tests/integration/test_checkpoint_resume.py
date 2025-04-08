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

# Project imports
from craft.training.trainer import Trainer
from craft.training.checkpointing import CHECKPOINT_FILE_PATTERN

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
    with open(dummy_pickled_data_path, 'wb') as f:
        pickle.dump(torch.tensor(dummy_token_ids, dtype=torch.long), f)

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
             'keep_last': '${checkpointing.keep_last_n}',
             'checkpoint_dir': str(checkpoint_dir),
             'resume_from_checkpoint': None, # Initial run: no resume
             'log_level': "INFO",
             'seed': 42,
             'device': "cpu",
             'learning_rate': '${optimizer.lr}', 
             'max_grad_norm': 1.0,
             'torch_compile': False,
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
    """Tests saving a checkpoint during a short run and resuming from it."""
    
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

    del trainer1 # Ensure resources are released (optional)
    torch.cuda.empty_cache() # If using GPU

    # --- Setup & Resume Run --- #
    print("--- Configuring for Resume Run --- ")
    # Modify the config to resume from the saved checkpoint
    cfg_resume = cfg.copy() # Create a copy to modify
    OmegaConf.update(cfg_resume, "training.resume_from_checkpoint", str(checkpoint_path_run1))
    # Optionally, increase max_steps slightly to see if it continues
    # OmegaConf.update(cfg_resume, "training.max_steps", cfg.training.max_steps + 2)
    print(f"--- Resume Config ---\n{OmegaConf.to_yaml(cfg_resume)}\n----------------------")

    # Instantiate the second trainer (should resume)
    print("--- Starting Resume Training Run --- ")
    trainer2 = instantiate(cfg_resume, _recursive_=False)
    assert isinstance(trainer2, Trainer)
    
    # Assert state BEFORE training starts again
    print(f"Trainer 2 Initial State: global_step={trainer2.global_step}, epoch={trainer2.epoch}")
    # The loaded step should be the step at which the checkpoint was SAVED
    assert trainer2.global_step == expected_checkpoint_step, f"Resumed step mismatch. Expected {expected_checkpoint_step}, got {trainer2.global_step}"
    # Epoch depends on when the checkpoint was saved relative to epoch boundaries.
    # For this simple test, step 2 is likely within epoch 0.
    # assert trainer2.epoch == expected_epoch_at_checkpoint 

    # trainer2.train() # Optionally run train again
    # print("--- Resume Training Run Finished --- ")
    # final_step_run2 = trainer2.global_step
    # assert final_step_run2 == cfg_resume.training.max_steps

    # Basic check passed if instantiation and state loading worked
    print("--- Checkpoint Resume Test Completed Successfully --- ") 