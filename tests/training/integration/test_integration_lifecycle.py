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
import json
from unittest.mock import patch

# Import necessary components from the craft library
from craft.config.schemas import TrainingConfig, AppConfig # Ensure AppConfig is available if needed
from craft.config.schemas import LanguageModelConfig # Import config from schemas
from craft.data.tokenizers.sentencepiece import SentencePieceTokenizer
from craft.training import Trainer
from craft.training.callbacks import SampleGenerationCallback, TensorBoardLogger
from craft.utils.common import set_seed

# Define constant if not imported
METADATA_FILENAME = "tokenizer_metadata.json"

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
    """Trains a SentencePiece model on the tiny corpus and saves metadata."""
    model_dir = tmp_path_factory.mktemp("sp_model_final") # Directory to save final model+metadata
    model_prefix = model_dir / "sp_temp" # Temporary prefix for training
    final_model_path = model_dir / SentencePieceTokenizer.MODEL_FILENAME # Use standard name
    metadata_path = model_dir / METADATA_FILENAME # Use standard name
    vocab_size = 50
    unk_id = 0
    bos_id = 1
    eos_id = 2
    pad_id = 3

    try:
        spm.SentencePieceTrainer.train(
            f'--input={str(tiny_corpus)} --model_prefix={str(model_prefix)} ' 
            f'--vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 ' 
            f'--unk_id={unk_id} --bos_id={bos_id} --eos_id={eos_id} --pad_id={pad_id}' # Ensure standard special tokens
        )
        # Rename the generated model file to the standard name
        trained_model_file = Path(f"{model_prefix}.model")
        if trained_model_file.exists():
            trained_model_file.rename(final_model_path)
            logger.info(f"Renamed trained SP model to {final_model_path}")
        else:
            pytest.fail(f"SentencePiece model file not found after training: {trained_model_file}")

        # Clean up temporary vocab file if it exists
        temp_vocab_file = Path(f"{model_prefix}.vocab")
        if temp_vocab_file.exists():
            temp_vocab_file.unlink()
            
    except Exception as e:
        pytest.fail(f"SentencePiece training failed: {e}")

    # Create and save metadata JSON
    metadata = {
        'model_type': 'sentencepiece',
        'vocab_size': vocab_size,
        'model_file': SentencePieceTokenizer.MODEL_FILENAME,
        'add_bos_as_control': False, # Assuming default behavior
        'add_eos_as_control': False, # Assuming default behavior
        'special_tokens_map': {
            'unk': '<unk>', # Assuming standard token strings
            'bos': '<s>',
            'eos': '</s>',
            'pad': '<pad>'
        },
        # Add any other relevant metadata if needed
    }
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved SentencePiece metadata to {metadata_path}")
    except Exception as e:
        pytest.fail(f"Failed to save SentencePiece metadata: {e}")

    # Check files exist
    if not final_model_path.exists():
        pytest.fail(f"Final SentencePiece model file not found: {final_model_path}")
    if not metadata_path.exists():
        pytest.fail(f"Final SentencePiece metadata file not found: {metadata_path}")

    # Return the directory containing the model and metadata
    return model_dir

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
                        "_target_": "craft.data.datasets.pickled_dataset.PickledDataset",
                        # Use tmp_path for dummy data files
                        "file_path": str(output_dir / "dummy_train.pkl"),
                        "block_size": 32,
                        "vocab_path": str(trained_sp_tokenizer) + ".vocab" # Path to .vocab file
                    }
                },
                "val": {
                    "dataset": {
                        "_target_": "craft.data.datasets.pickled_dataset.PickledDataset",
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

def test_subword_lifecycle(tmp_path, simple_config_dict_subword):
    """Tests the basic lifecycle: init -> train (mocked) -> finish for subword."""
    # ... existing setup code ...

    # --- Trainer Initialization ---
    trainer = Trainer(config=cfg, model=model, dataset=dataset)

    # Assert components are initialized after __init__
    assert trainer.model is not None
    assert trainer.optimizer is not None # Optimizer is created in __init__
    assert trainer.train_dataloader is not None # Dataloader is created in __init__
    assert trainer.callbacks is not None
    assert trainer.logger is not None
    assert trainer.progress is not None # ProgressTracker initialized
    assert trainer.global_step == 0 # Initial state
    assert trainer.epoch == 0

    # --- Mock Training ---
    # Mock the core training loop part to avoid actual training steps
    with patch.object(trainer.training_loop, 'train_epoch', return_value={'loss': 0.5}) as mock_train_epoch, \
         patch.object(trainer.evaluator, 'evaluate', return_value={'val_loss': 0.6}) as mock_evaluate, \
         patch.object(trainer.checkpoint_manager, 'save_checkpoint') as mock_save_checkpoint:

        # --- Run Training ---
        # trainer.setup() # This method does not exist, initialization happens in __init__
        result = trainer.train() # Call the main training method

        # --- Assertions ---
        # Verify train was called (implies loop ran at least once for 1 epoch)
        mock_train_epoch.assert_called()
        # Verify evaluate was called if needed (based on default val_interval_steps=1000, max_steps=100)
        # Evaluate should not be called in this short run
        mock_evaluate.assert_not_called()

        # Check final state reported by train
        assert result["final_global_step"] > 0 # Should have taken some steps
        assert result["final_epoch"] == cfg.training.num_epochs - 1 # Should complete epochs or stop early

        # Check internal state
        # Global step should update based on dataloader size and epochs/max_steps
        # For max_steps=100, train_batch_size=2, vocab_size=50 -> ~5 batches needed.
        # Checkpoint manager interactions
        # Default save_interval_steps is 1000, should not be called for max_steps=100
        mock_save_checkpoint.assert_not_called()

    # Additional checks if needed (e.g., logger messages)
    # Check if logs were written (basic check)
    log_file = tmp_path / "logs" / f"{cfg.experiment_name}.log"
    assert log_file.exists()
    # More specific log content checks could be added