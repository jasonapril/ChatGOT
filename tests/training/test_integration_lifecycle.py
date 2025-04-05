#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for the full training lifecycle, focusing on:
- Subword tokenization
- Checkpointing and Resuming
- Sample Generation callback
- TensorBoard Logging callback
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
import sentencepiece as spm
from pathlib import Path
import tempfile
import os
import logging
import hydra
import pickle

# Import necessary components from the craft library
from craft.config.schemas import TrainingConfig
# Import TransformerModel directly from its module
from craft.models import create_model_from_config, LanguageModelConfig
from craft.models.transformer import TransformerModel
# Import correct data factory function name
from craft.data.factory import prepare_dataloaders_from_config
# from craft.data.configs import TokenizerConfig, DataConfig # Assuming DataConfig exists - REMOVED
from craft.data.tokenizers import SentencePieceTokenizer # Assuming this exists
from craft.training import Trainer
from craft.training.callbacks import SampleGenerationCallback, TensorBoardLogger

# Configure basic logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Fixtures --- #

@pytest.fixture(scope="module") # Scope module to train tokenizer once per module
def tiny_corpus(tmp_path_factory):
    """Creates a tiny text file for training the tokenizer and dataset."""
    content = "hello world.\nthis is a test sentence.\nsubword tokenizers work."
    filepath = tmp_path_factory.mktemp("data") / "tiny_corpus.txt"
    filepath.write_text(content)
    return filepath

@pytest.fixture(scope="module")
def trained_sp_tokenizer(tiny_corpus, tmp_path_factory):
    """Trains a SentencePiece model on the tiny corpus."""
    model_prefix = tmp_path_factory.mktemp("sp_model") / "tiny_sp"
    try:
        spm.SentencePieceTrainer.train(
            f'--input={str(tiny_corpus)} --model_prefix={str(model_prefix)} ' 
            f'--vocab_size=50 --model_type=bpe --character_coverage=1.0 ' 
            f'--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3' # Ensure standard special tokens
        )
    except Exception as e:
        pytest.fail(f"SentencePiece training failed: {e}")
        
    model_path = Path(f"{model_prefix}.model")
    if not model_path.exists():
        pytest.fail(f"SentencePiece model file not found after training: {model_path}")
        
    # Return the prefix path, tokenizer can load from this
    return model_prefix

@pytest.fixture
def integration_config(trained_sp_tokenizer, tmp_path):
    """Creates a DictConfig for the integration test."""
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "output"

    conf = OmegaConf.create({
        "data": {
            "datasets": {
                "train": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": "dummy_train.pkl", # Path set later
                        "block_size": 32,
                        "vocab_path": str(trained_sp_tokenizer) + ".vocab" # Path to .vocab file
                    }
                },
                "val": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        "file_path": "dummy_val.pkl", # Path set later
                        "block_size": 32,
                        "vocab_path": str(trained_sp_tokenizer) + ".vocab" # Path to .vocab file
                    }
                }
            },
            "tokenizer": {
                 "_target_": "craft.data.tokenizers.SentencePieceTokenizer",
                 "model_path": str(trained_sp_tokenizer),
            },
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "_target_": "craft.models.transformer.TransformerModel",
            "config": { 
                # Smallest possible transformer config
                "vocab_size": 50, # Needs to match trained tokenizer
                "d_model": 16,
                "n_layers": 1,
                "n_head": 1,
                "dropout": 0.0,
                "bias": False,
                "max_seq_length": 32
            }
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-3
        },
        "scheduler": None,
        "training": {
            "device": "cpu",
            "seed": 1337,
            "num_epochs": 2,
            "gradient_accumulation_steps": 1,
            "use_amp": False,
            "max_grad_norm": 1.0,
            "log_interval": 1,
            "eval_interval": 2, # Evaluate every 2 steps
            "sample_interval": 2, # Sample every 2 steps
            "max_steps": 4, # Run only a few steps total
            "log_level": "DEBUG",
            "torch_compile": False, 
            "sample_max_new_tokens": 10,
            "sample_temperature": 0.8,
            "sample_start_text": "this is"
        },
        "callbacks": [
            {
                "_target_": "craft.training.callbacks.SampleGenerationCallback",
                "prompt": "hello", # Override default
                "max_new_tokens": 5,
                "temperature": 0.7,
                "step_interval": 2, # Match training.sample_interval 
            },
            {
                "_target_": "craft.training.callbacks.TensorBoardLogger",
                "log_dir": str(log_dir),
            }
        ],
        "checkpoints": {
            "checkpoint_dir": str(checkpoint_dir),
            "save_steps_interval": 2, # Save every 2 steps
            "keep_last": 1,
            "resume_from_checkpoint": None # Initially no resume
        },
        "output_dir": str(output_dir),
    })
    # --- Update paths AFTER creating the config dict --- #
    # We need the output_dir which is derived from tmp_path
    output_dir = Path(conf.output_dir)
    conf.data.datasets.train.dataset.file_path = str(output_dir / "dummy_train.pkl")
    conf.data.datasets.val.dataset.file_path = str(output_dir / "dummy_val.pkl")
    # Update vocab path too if it relies on tmp_path indirectly via tokenizer fixture
    conf.data.datasets.train.dataset.vocab_path = str(trained_sp_tokenizer) + ".vocab"
    conf.data.datasets.val.dataset.vocab_path = str(trained_sp_tokenizer) + ".vocab"

    return conf


# --- Test Function --- #

def test_subword_lifecycle(integration_config: DictConfig, tiny_corpus: Path):
    """Runs a short training loop, checkpoints, resumes, and checks outputs."""
    
    # --- Setup: Prepare Data (Simulated) ---
    output_dir = Path(integration_config.output_dir)
    # Use the correct key 'file_path' from the config
    train_path = Path(integration_config.data.datasets.train.dataset.file_path)
    val_path = Path(integration_config.data.datasets.val.dataset.file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save *only* the list/array of IDs using standard pickle, as expected by PickledDataset
    dummy_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 10
    with open(train_path, 'wb') as f:
        pickle.dump(dummy_ids, f)
    with open(val_path, 'wb') as f:
        pickle.dump(dummy_ids, f)

    # No need to update config paths here anymore, they are set correctly in the fixture

    # --- Instantiate components from config --- #
    logger.info("Instantiating components...")
    try:
        # 1. Dataloaders and Tokenizer
        # Use prepare_dataloaders_from_config which handles tokenizer init
        train_loader, val_loader, _, tokenizer = prepare_dataloaders_from_config(integration_config)
        assert train_loader is not None, "Train loader failed to initialize"
        assert val_loader is not None, "Validation loader failed to initialize"
        assert tokenizer is not None, "Tokenizer failed to initialize"
        vocab_size = tokenizer.get_vocab_size()
        # Update model vocab size just in case it differs from the one in config
        # (though we set it explicitly in the fixture)
        integration_config.model.config.vocab_size = vocab_size

        # 2. Model
        model = create_model_from_config(integration_config.model)
        
        # 3. Optimizer
        # Basic AdamW setup for the test
        optimizer = torch.optim.AdamW(model.parameters(), lr=integration_config.optimizer.lr)

        # 4. Scheduler (Optional)
        scheduler = None # Configured as None in fixture
        
        # 5. Training Config (Pydantic object)
        # Combine training and checkpoint args for Pydantic model
        flat_training_config = OmegaConf.to_container(integration_config.training, resolve=True)
        flat_training_config.update(OmegaConf.to_container(integration_config.checkpoints, resolve=True))
        flat_training_config["log_level"] = integration_config.training.log_level # Ensure log level is included
        # Add batch_size from the data config section
        flat_training_config["batch_size"] = integration_config.data.batch_size
        training_config = TrainingConfig(**flat_training_config)

        # 6. Callbacks (Instantiate from config)
        callbacks_list = []
        if "callbacks" in integration_config and integration_config.callbacks:
            for cb_conf in integration_config.callbacks:
                try:
                     # Pass trainer later using set_trainer
                     callback_instance = hydra.utils.instantiate(cb_conf, _convert_="partial") 
                     callbacks_list.append(callback_instance)
                except Exception as e:
                     logger.error(f"Failed to instantiate callback {cb_conf.get('_target_')}: {e}")
                     pytest.fail(f"Callback instantiation failed: {e}")

    except Exception as e:
        logger.exception("Component instantiation failed.")
        pytest.fail(f"Failed to instantiate components: {e}")

    # --- Part 1: Initial Training --- #
    logger.info("--- Starting Initial Training (Part 1) ---")
    trainer1 = Trainer(
        config=training_config, 
        model=model, 
        optimizer=optimizer, 
        train_dataloader=train_loader, 
        val_dataloader=val_loader,
        scheduler=scheduler,
        tokenizer=tokenizer,
        callbacks=callbacks_list # Pass instantiated list
    )
    try:
        trainer1.train()
    except Exception as e:
        logger.exception("Initial training failed.")
        pytest.fail(f"Initial trainer.train() failed: {e}")
        
    # Assertions for Part 1 (paths relative to tmp_path)
    checkpoint_dir = Path(integration_config.checkpoints.checkpoint_dir)
    log_dir = Path(integration_config.callbacks[1].log_dir) # Index 1 is TensorBoard
    # Remove assertion for sample log path as we removed log_to_file from config
    # sample_log_path = Path(integration_config.callbacks[0].log_to_file) 

    expected_ckpt_pattern = "checkpoint_step_*.pt"
    checkpoints = list(checkpoint_dir.glob(expected_ckpt_pattern))
    assert len(checkpoints) >= 1, f"Expected >=1 checkpoint, found {len(checkpoints)} in {checkpoint_dir}"
    # Check for the step 4 checkpoint specifically due to max_steps=4 and save_steps_interval=2
    step4_ckpt = checkpoint_dir / "checkpoint_step_000004.pt"
    assert step4_ckpt.exists(), f"Checkpoint for step 4 not found: {step4_ckpt}"
    latest_checkpoint_path = step4_ckpt # Use the specific checkpoint we expect

    assert log_dir.exists() and log_dir.is_dir(), f"TensorBoard log directory missing: {log_dir}"
    assert len(list(log_dir.glob("events.out.tfevents.*"))) > 0, f"No TensorBoard event files found in {log_dir}"

    # Remove assertion for sample log path
    # assert sample_log_path.exists(), f"Sample generation log file missing: {sample_log_path}"
    # assert sample_log_path.stat().st_size > 0, f"Sample generation log file is empty: {sample_log_path}"
    
    # --- Part 2: Resume Training --- #
    logger.info("--- Starting Resumed Training (Part 2) ---")
    # Create new components for the second trainer instance
    try:
        model2 = create_model_from_config(integration_config.model)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=integration_config.optimizer.lr)
        scheduler2 = None
        
        # Update config for resume
        resume_flat_config = flat_training_config.copy()
        resume_flat_config["resume_from_checkpoint"] = str(latest_checkpoint_path)
        resume_flat_config["max_steps"] = 6 # Increase max steps
        resume_training_config = TrainingConfig(**resume_flat_config)
        
        # Re-instantiate callbacks (state should be loaded by Trainer via CheckpointManager)
        callbacks_list_2 = []
        if "callbacks" in integration_config and integration_config.callbacks:
            for cb_conf in integration_config.callbacks:
                 callbacks_list_2.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

    except Exception as e:
        logger.exception("Component instantiation for resume failed.")
        pytest.fail(f"Failed to instantiate components for resume: {e}")

    trainer2 = Trainer(
        config=resume_training_config, 
        model=model2, 
        optimizer=optimizer2, 
        train_dataloader=train_loader, # Reuse dataloaders
        val_dataloader=val_loader,
        scheduler=scheduler2,
        tokenizer=tokenizer,
        callbacks=callbacks_list_2
    )
    try:
        trainer2.train()
    except Exception as e:
        logger.exception("Resumed training failed.")
        pytest.fail(f"Resumed trainer.train() failed: {e}")
        
    # Assertions for Part 2
    expected_ckpt_pattern_resume = "checkpoint_step_*.pt"
    checkpoints_resume = list(checkpoint_dir.glob(expected_ckpt_pattern_resume))
    # keep_last=1 means only step 6 checkpoint should remain
    assert len(checkpoints_resume) == 1, f"Expected 1 checkpoint after resume (keep_last=1), found {len(checkpoints_resume)} in {checkpoint_dir}"
    assert "step_000006" in checkpoints_resume[0].name, f"Expected checkpoint at step 6 after resume, found {checkpoints_resume[0].name}"
    assert not step4_ckpt.exists(), f"Checkpoint from step 4 should have been deleted (keep_last=1)"

    # Remove assertion for sample log path
    # assert sample_log_path.exists(), f"Sample generation log file missing after resume: {sample_log_path}"
    # Ideally, check content or size increased, but existing & non-empty is a basic check
    # assert sample_log_path.stat().st_size > 0, f"Sample generation log file empty after resume: {sample_log_path}"

    logger.info("Integration lifecycle test completed successfully.") 