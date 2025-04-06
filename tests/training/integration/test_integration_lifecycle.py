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
    """Creates a DictConfig for the integration test, nested under 'experiment'.
       Uses tmp_path for output directories to ensure cleanup."""
    # Use tmp_path for test-specific output directories
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    output_dir = tmp_path / "output"

    # Ensure the base output directory exists for file paths defined below
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the core configuration
    core_conf = {
        "data": {
            "datasets": {
                "train": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        # Use tmp_path for dummy data files
                        "file_path": str(output_dir / "dummy_train.pkl"),
                        "block_size": 32,
                        "vocab_path": str(trained_sp_tokenizer) + ".vocab" # Path to .vocab file
                    }
                },
                "val": {
                    "dataset": {
                        "_target_": "craft.data.dataset.PickledDataset",
                        # Use tmp_path for dummy data files
                        "file_path": str(output_dir / "dummy_val.pkl"),
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
            "time_save_interval_seconds": 0, # ADDED - Disable time-based saving for this test
            "log_level": "DEBUG",
            "torch_compile": False,
            "sample_max_new_tokens": 10,
            "sample_temperature": 0.8,
            "sample_start_text": "this is",
            "mixed_precision": False, # Keep simple for integration test
        },
        "callbacks": [
            {
                "_target_": "craft.training.callbacks.SampleGenerationCallback",
                "max_new_tokens": 5,
                "temperature": 0.7,
                "step_interval": 2, # Match training.sample_interval
            },
            {
                "_target_": "craft.training.callbacks.TensorBoardLogger",
                # Use tmp_path log_dir
                "log_dir": str(log_dir),
            }
        ],
        "checkpoints": {
            # Use tmp_path checkpoint_dir
            "checkpoint_dir": str(checkpoint_dir),
            "save_steps_interval": 2, # Save every 2 steps
            "keep_last": 2, # Keep step 2 and step 4 checkpoints
            "resume_from_checkpoint": None # Initially no resume
        },
        # Use tmp_path output_dir
        "output_dir": str(output_dir),
    }

    # Wrap the core config under an 'experiment' key
    conf = OmegaConf.create({"experiment": core_conf})

    # No need to update paths again, they were defined correctly above
    return conf

# --- Test Function --- #

def test_subword_lifecycle(integration_config: DictConfig, tiny_corpus: Path):
    """Runs a short training loop, checkpoints, resumes, and checks outputs."""
    
    # --- Setup: Prepare Data (Simulated) ---
    # Access paths relative to the nested structure
    output_dir = Path(integration_config.experiment.output_dir)
    train_path = Path(integration_config.experiment.data.datasets.train.dataset.file_path)
    val_path = Path(integration_config.experiment.data.datasets.val.dataset.file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save *only* the list/array of IDs using standard pickle, as expected by PickledDataset
    dummy_ids = [1, 2, 3, 4, 5, 6, 7, 8] * 10
    with open(train_path, 'wb') as f:
        pickle.dump(dummy_ids, f)
    with open(val_path, 'wb') as f:
        pickle.dump(dummy_ids, f)

    # --- Instantiate components from config --- #
    logger.info("Instantiating components...")
    try:
        # 1. Dataloaders and Tokenizer
        # Use prepare_dataloaders_from_config which handles tokenizer init
        # Pass the full integration_config object
        train_loader, val_loader, _, tokenizer = prepare_dataloaders_from_config(integration_config)

        assert train_loader is not None, "Train loader failed to initialize"
        assert val_loader is not None, "Validation loader failed to initialize"
        assert tokenizer is not None, "Tokenizer failed to initialize"
        vocab_size = tokenizer.get_vocab_size()
        integration_config.experiment.model.config.vocab_size = vocab_size

        # # --- Explicitly set tokenizer on datasets --- #
        # # This is needed because TextGenerator inside the callback needs access via dataset
        train_dataset_id = None
        val_dataset_id = None
        tokenizer_id = id(tokenizer)
        if hasattr(train_loader.dataset, 'tokenizer'):
            train_loader.dataset.tokenizer = tokenizer
            train_dataset_id = id(train_loader.dataset)
            logger.debug(f"[Test Setup] Set tokenizer (ID: {tokenizer_id}) on train_loader.dataset (ID: {train_dataset_id})")
            logger.debug(f"[Test Setup] train_loader.dataset.__dict__ after set: {getattr(train_loader.dataset, '__dict__', 'N/A')}")
        if hasattr(val_loader.dataset, 'tokenizer'):
            val_loader.dataset.tokenizer = tokenizer
            val_dataset_id = id(val_loader.dataset)
            logger.debug(f"[Test Setup] Set tokenizer on val_loader.dataset (ID: {val_dataset_id})")
        # # --- End explicit set --- #

        # 2. Model
        model = create_model_from_config(integration_config.experiment.model)
        
        # 3. Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=integration_config.experiment.optimizer.lr)

        # 4. Scheduler (Optional)
        scheduler = None # Configured as None in fixture
        
        # 5. Training Config (Pydantic object)
        # Combine training and checkpoint args from the nested structure
        flat_training_config = OmegaConf.to_container(integration_config.experiment.training, resolve=True)
        flat_training_config.update(OmegaConf.to_container(integration_config.experiment.checkpoints, resolve=True))
        flat_training_config["log_level"] = integration_config.experiment.training.log_level # Ensure log level is included
        flat_training_config["batch_size"] = integration_config.experiment.data.batch_size
        training_config = TrainingConfig(**flat_training_config)

        # 6. Callbacks (Instantiate from config)
        callbacks_list = []
        if "callbacks" in integration_config.experiment and integration_config.experiment.callbacks:
            for cb_conf in integration_config.experiment.callbacks:
                try:
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
        callbacks=callbacks_list, # Pass the instantiated list
        experiment_name="lifecycle_test" # Provide an experiment name
    )
    trainer1.train()
    logger.info("--- Initial Training Finished (Part 1) ---")

    # --- Assertions after Part 1 --- #
    assert trainer1.global_step == integration_config.experiment.training.max_steps
    # Check if checkpoints were created (based on save_steps_interval = 2, max_steps = 4)
    checkpoint_dir = Path(integration_config.experiment.checkpoints.checkpoint_dir)
    step2_ckpt = checkpoint_dir / "checkpoint_step_000002.pt"
    step4_ckpt = checkpoint_dir / "checkpoint_step_000004.pt"
    assert step2_ckpt.exists(), "Checkpoint at step 2 should exist"
    assert step4_ckpt.exists(), "Checkpoint at step 4 should exist"
    # Check if TensorBoard logs exist
    log_dir = Path(integration_config.experiment.callbacks[1].log_dir) # Assuming TB logger is second callback
    assert any(log_dir.iterdir()), "TensorBoard log directory should not be empty"
    # Check if sample generation produced output (difficult to assert content precisely)
    # We can check if the callback logged something, but that requires capturing logs

    # Get state dicts before potentially resetting for resume
    model_state_after_part1 = model.state_dict()
    optimizer_state_after_part1 = optimizer.state_dict()

    # --- Part 2: Resuming Training --- #
    logger.info("--- Starting Resumed Training (Part 2) ---")
    # Modify config to resume from the last checkpoint (step 4)
    integration_config.experiment.checkpoints.resume_from_checkpoint = str(step4_ckpt)
    # Increase max_steps to allow further training
    new_max_steps = 6
    integration_config.experiment.training.max_steps = new_max_steps

    # Re-create components (simulate restarting the script)
    logger.info("Re-instantiating components for resume...")
    try:
        # Re-create using the modified config
        train_loader2, val_loader2, _, tokenizer2 = prepare_dataloaders_from_config(integration_config)
        model2 = create_model_from_config(integration_config.experiment.model)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=integration_config.experiment.optimizer.lr)
        scheduler2 = None
        # Re-create TrainingConfig with updated max_steps and resume path
        flat_training_config2 = OmegaConf.to_container(integration_config.experiment.training, resolve=True)
        flat_training_config2.update(OmegaConf.to_container(integration_config.experiment.checkpoints, resolve=True))
        flat_training_config2["log_level"] = integration_config.experiment.training.log_level
        flat_training_config2["batch_size"] = integration_config.experiment.data.batch_size
        training_config2 = TrainingConfig(**flat_training_config2)

        # # --- Explicitly set tokenizer on datasets (Part 2) --- #
        # if hasattr(train_loader2.dataset, 'tokenizer'):
        #     train_loader2.dataset.tokenizer = tokenizer2
        # if hasattr(val_loader2.dataset, 'tokenizer'):
        #     val_loader2.dataset.tokenizer = tokenizer2
        # # --- End explicit set --- #

        callbacks_list2 = []
        if "callbacks" in integration_config.experiment and integration_config.experiment.callbacks:
            for cb_conf in integration_config.experiment.callbacks:
                callbacks_list2.append(hydra.utils.instantiate(cb_conf, _convert_="partial"))

    except Exception as e:
        logger.exception("Component re-instantiation for resume failed.")
        pytest.fail(f"Failed to re-instantiate components for resume: {e}")

    trainer2 = Trainer(
        config=training_config2, # Use updated config
        model=model2, 
        optimizer=optimizer2, 
        train_dataloader=train_loader2, 
        val_dataloader=val_loader2,
        scheduler=scheduler2,
        tokenizer=tokenizer2,
        callbacks=callbacks_list2,
        experiment_name="lifecycle_test_resume", # Can use same or different
        resume_from_checkpoint=str(step4_ckpt) # <-- EXPLICITLY PASS RESUME PATH
    )
    # Trainer init should load the checkpoint specified in config
    assert trainer2.global_step == 4, "Trainer should resume from global_step 4"
    trainer2.train() 
    logger.info("--- Resumed Training Finished (Part 2) ---")

    # --- Assertions after Part 2 --- #
    assert trainer2.global_step == new_max_steps
    # Check if a new checkpoint was created at step 6 (save_steps_interval = 2)
    step6_ckpt = checkpoint_dir / "checkpoint_step_000006.pt"
    assert step6_ckpt.exists(), "Checkpoint at step 6 should exist after resume"
    # Due to keep_last=2, step 2 checkpoint should be deleted
    assert not step2_ckpt.exists(), "Checkpoint at step 2 should be deleted (keep_last=2)"

    # Assert that model/optimizer states are different after resuming and training further

# ... (Potentially add more specific assertions about model weights if needed) ...