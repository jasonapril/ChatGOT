"""Command-line interface for ChatGoT."""
import click
from pathlib import Path
from typing import Optional

from ..training.trainer import train_with_samples
from ..utils.text_generation import generate_text
from ..data.dataset import prepare_data
from ..utils.logging import setup_logging

@click.group()
def cli():
    """ChatGoT CLI - Experimental AI Model Framework"""
    pass

@cli.command()
@click.option('--config-path', type=click.Path(exists=True), default='configs/merged_config.yaml',
              help='Path to configuration file')
@click.option('--num-epochs', type=int, default=50, help='Number of epochs to train')
@click.option('--sample-length', type=int, default=100, help='Length of generated samples')
@click.option('--sample-temperature', type=float, default=0.8, help='Temperature for sampling')
@click.option('--sample-top-k', type=int, default=40, help='Top-k sampling parameter')
@click.option('--sample-top-p', type=float, default=0.9, help='Top-p sampling parameter')
@click.option('--device', type=str, default=None, help='Device to train on')
def train(config_path: str, num_epochs: int, sample_length: int, sample_temperature: float,
          sample_top_k: int, sample_top_p: float, device: Optional[str]):
    """Train a model with sample text generation."""
    train_with_samples(
        config_path=config_path,
        num_epochs=num_epochs,
        sample_length=sample_length,
        sample_temperature=sample_temperature,
        sample_top_k=sample_top_k,
        sample_top_p=sample_top_p,
        device=device,
    )

@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model')
@click.option('--prompt', type=str, required=True, help='Text prompt for generation')
@click.option('--max-length', type=int, default=100, help='Maximum length of generated text')
@click.option('--temperature', type=float, default=0.8, help='Temperature for sampling')
@click.option('--top-k', type=int, default=40, help='Top-k sampling parameter')
@click.option('--top-p', type=float, default=0.9, help='Top-p sampling parameter')
@click.option('--device', type=str, default=None, help='Device to generate on')
def generate(model_path: str, prompt: str, max_length: int, temperature: float,
            top_k: int, top_p: float, device: Optional[str]):
    """Generate text from a trained model."""
    text = generate_text(
        model_path=model_path,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )
    click.echo(text)

@cli.command()
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='Input text file to process')
@click.option('--output-dir', type=click.Path(), default='processed_data',
              help='Directory to save processed data')
@click.option('--config-path', type=click.Path(exists=True), default='configs/data/default.yaml',
              help='Path to data processing configuration')
def prepare(input_file: str, output_dir: str, config_path: str):
    """Prepare data for training."""
    prepare_data(
        input_file=input_file,
        output_dir=output_dir,
        config_path=config_path,
    )

@cli.command()
@click.option('--config-path', type=click.Path(exists=True), required=True,
              help='Path to experiment configuration')
@click.option('--output-dir', type=click.Path(), default='runs/experiments',
              help='Directory to save experiment results')
def experiment(config_path: str, output_dir: str):
    """Run an experiment from configuration."""
    from ..experiments.runner import run_experiment
    run_experiment(
        config_path=config_path,
        output_dir=output_dir,
    )

if __name__ == '__main__':
    cli() 