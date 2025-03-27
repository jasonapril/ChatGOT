#!/usr/bin/env python
"""
Command-line interface for Craft.

This module provides a unified CLI for working with different model types
and experiments.
"""
import os
import logging
import typer
from typing import Optional, List, Tuple, Dict, Any

import torch
from rich.console import Console
from rich.logging import RichHandler

from ..config.config_manager import ConfigManager, load_experiment_config
from ..models.base import create_model_from_config
from ..data.base import prepare_dataloaders_from_config
from ..training.base import create_trainer_from_config
from ..utils.common import set_seed, setup_device

# Create Typer app
app = typer.Typer(
    name="craft",
    help="A framework for developing AI models",
    add_completion=False,
)

# Create subcommands
train_app = typer.Typer(help="Commands for model training")
generate_app = typer.Typer(help="Commands for text generation")
evaluate_app = typer.Typer(help="Commands for model evaluation")
experiment_app = typer.Typer(help="Commands for running experiments")
dataset_app = typer.Typer(help="Commands for dataset operations")

# Add subcommands to the main app
app.add_typer(train_app, name="train")
app.add_typer(generate_app, name="generate")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(experiment_app, name="experiment")
app.add_typer(dataset_app, name="dataset")

# Global console for rich output
console = Console()


def setup_logging(log_level: str = "INFO"):
    """Set up logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Set log level"),
):
    """
    Craft - A framework for developing AI models.
    """
    # Set up logging
    if verbose:
        log_level = "DEBUG"
    setup_logging(log_level)


@train_app.command("language")
def train_language_model(
    config_path: str = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device to train on"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume training from checkpoint"),
    checkpoint_path: Optional[str] = typer.Option(None, "--checkpoint", help="Path to checkpoint"),
):
    """Train a language model."""
    # Load configuration
    config = load_experiment_config(config_path)
    
    # Override configuration
    if output_dir:
        config["paths"]["output_dir"] = output_dir
    if seed:
        config["system"]["seed"] = seed
    
    # Set seed for reproducibility
    set_seed(config.get("system", {}).get("seed", 42))
    
    # Set up device
    device_name = device or config.get("system", {}).get("device", "auto")
    device = setup_device(device_name)
    
    # Create output directory
    output_dir = config.get("paths", {}).get("output_dir", "runs/default")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Update config with paths
    config["training"]["checkpoint_dir"] = checkpoint_dir
    
    # Prepare data
    console.print(f"Preparing data from configuration...")
    train_dataloader, val_dataloader = prepare_dataloaders_from_config(config)
    
    # Create model
    console.print(f"Creating model from configuration...")
    model = create_model_from_config(config["model"])
    
    # Create trainer
    console.print(f"Setting up trainer...")
    trainer = create_trainer_from_config(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config["training"],
    )
    
    # Resume training if requested
    if resume and checkpoint_path:
        console.print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Train model
    console.print(f"Starting training...")
    metrics = trainer.train()
    
    console.print(f"Training completed!")
    console.print(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
    if val_dataloader:
        console.print(f"Final validation loss: {metrics['val_loss'][-1]:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    trainer.save_checkpoint(final_model_path)
    console.print(f"Final model saved to: {final_model_path}")


@generate_app.command("text")
def generate_text(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to model checkpoint"),
    prompt: str = typer.Option("", "--prompt", "-p", help="Prompt for generation"),
    max_length: int = typer.Option(100, "--max-length", "-l", help="Maximum length of generated text"),
    temperature: float = typer.Option(0.8, "--temperature", "-t", help="Sampling temperature"),
    top_k: int = typer.Option(40, "--top-k", "-k", help="Top-k sampling parameter"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling parameter"),
    repetition_penalty: float = typer.Option(1.0, "--repetition-penalty", "-r", help="Repetition penalty"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device to generate on"),
):
    """Generate text from a trained language model."""
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)
    
    # Set up device
    device = setup_device(device or "auto")
    
    # Load the model
    console.print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get("config", {}).get("model", {})
    if not config:
        # Try to extract from state dict
        config = {"model_type": "language", "architecture": "transformer"}
    
    # Create model
    model = create_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    if not prompt:
        # Start with a token or character from the vocabulary
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        # TODO: This is a placeholder for tokenization
        # This will need to be updated to use proper tokenization based on the model
        from ..data.dataset import CharDataset
        char_to_idx = checkpoint.get("char_to_idx", {})
        if char_to_idx:
            input_ids = torch.tensor(
                [[char_to_idx.get(c, 0) for c in prompt]],
                dtype=torch.long,
                device=device
            )
        else:
            raise ValueError("Cannot tokenize prompt: no character mapping found in checkpoint")
    
    # Generate text
    console.print(f"Generating text with parameters:")
    console.print(f"  - Max length: {max_length}")
    console.print(f"  - Temperature: {temperature}")
    console.print(f"  - Top-k: {top_k}")
    console.print(f"  - Top-p: {top_p}")
    console.print(f"  - Repetition penalty: {repetition_penalty}")
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        verbose=True
    )
    
    # Decode output
    idx_to_char = checkpoint.get("idx_to_char", {})
    if idx_to_char:
        generated_text = "".join([idx_to_char.get(str(idx), "?") for idx in output_ids[0].tolist()])
    else:
        raise ValueError("Cannot decode output: no character mapping found in checkpoint")
    
    console.print(f"\nGenerated text:")
    console.print(generated_text)


@experiment_app.command("run")
def run_experiment(
    config_path: str = typer.Option(..., "--config", "-c", help="Path to experiment configuration"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
):
    """Run an experiment from configuration."""
    from ..experiments.runner import run_experiment
    
    # Load configuration
    config = load_experiment_config(config_path)
    
    # Override output directory if provided
    if output_dir:
        config["paths"]["output_dir"] = output_dir
    
    # Run experiment
    output_dir = config.get("paths", {}).get("output_dir", "runs/experiments")
    run_experiment(config, output_dir)


@dataset_app.command("prepare")
def prepare_dataset(
    input_file: str = typer.Option(..., "--input", "-i", help="Input data file"),
    output_dir: str = typer.Option("data/processed", "--output-dir", "-o", help="Output directory"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Data processing configuration"),
):
    """Prepare a dataset for training."""
    from ..data.processors import prepare_data
    
    # Load configuration if provided
    config = None
    if config_path:
        config = load_experiment_config(config_path)
    
    # Prepare data
    prepare_data(input_file, output_dir, config)


if __name__ == "__main__":
    app() 