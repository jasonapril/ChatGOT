# src/cli/generate_commands.py
import typer
import logging
from typing import Optional, Tuple, cast
# import yaml # Keep for potential future use? Or remove?
# from omegaconf import OmegaConf # Keep for potential future use?

import torch
from pathlib import Path
# import hydra # No longer needed directly for instantiation here

from ..utils.common import set_seed, setup_device
# Import the updated model loading utility
from ..utils.model_io import load_model_for_inference
# Import base classes for type checking
from ..models.base import Model, GenerativeModel
# Tokenizer imports might still be needed for type hinting or error handling
from ..data.tokenizers.base import Tokenizer
# Keep TextGenerator and manual sampling imports? 
# TextGenerator might be redundant if model.generate works well.
# Manual sampling is a fallback.
# from ..training.generation import TextGenerator
# from ..training.sampling import generate_text_manual_sampling

generate_app = typer.Typer(help="Commands for text generation")
logger = logging.getLogger(__name__)
console = typer.echo

@generate_app.command("text")
def generate_text(
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-c", help="Path to the model checkpoint (.pt file or directory containing model.safetensors/model.bin and config/tokenizer)", exists=True, resolve_path=True),
    prompt: str = typer.Option("", "--prompt", "-p", help="Text prompt to start generation from."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", "-l", help="Maximum number of new tokens to generate."),
    temperature: float = typer.Option(0.8, "--temperature", "-t", help="Sampling temperature."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Top-k sampling threshold (set k, e.g., 50). Disable = None."),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Top-p (nucleus) sampling threshold (set p, e.g., 0.9). Disable = None."),
    # Add repetition penalty CLI option
    repetition_penalty: float = typer.Option(1.0, "--repetition-penalty", "--rp", help="Repetition penalty (1.0 = no penalty)."),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device override (e.g., 'cpu', 'cuda'). Auto-detect if None."),
) -> None:
    """Generate text using a trained model checkpoint/directory."""
    
    console(f"Starting text generation using checkpoint/model path: {checkpoint_path}")
    
    # --- 1. Setup --- 
    if seed is not None:
        set_seed(seed)
        console(f"Using random seed: {seed}")
    
    resolved_device = setup_device(device or "auto")
    console(f"Using device: {resolved_device}")

    # --- 2. Load Model and Tokenizer using Utility --- 
    console("Loading model and tokenizer...")
    model: Model
    tokenizer: Tokenizer
    try:
        # Pass path as string
        loaded_model, loaded_tokenizer = load_model_for_inference(str(checkpoint_path), device=resolved_device)
        # Cast to specific types for type checker clarity
        model = cast(Model, loaded_model)
        tokenizer = cast(Tokenizer, loaded_tokenizer)

        model.eval() # Ensure model is in eval mode
        console(f"Model ({type(model).__name__}) and Tokenizer ({type(tokenizer).__name__}) loaded successfully.")
        
        # Check if the loaded model is actually a GenerativeModel
        if not isinstance(model, GenerativeModel):
             logger.error(f"Loaded model ({type(model).__name__}) is not a GenerativeModel subclass and lacks a required 'generate' method.")
             raise typer.Exit(code=1)
        
        # Check if tokenizer has necessary methods (basic check)
        if not (hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode')):
             logger.error(f"Loaded tokenizer ({type(tokenizer).__name__}) does not have required encode/decode methods.")
             raise typer.Exit(code=1)
             
    except FileNotFoundError:
        logger.error(f"Checkpoint path or necessary files not found: {checkpoint_path}")
        raise typer.Exit(code=1)
    except ValueError as ve:
         logger.error(f"Configuration or loading error: {ve}", exc_info=True)
         raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Failed to load model or tokenizer.")
        raise typer.Exit(code=1)

    # --- 3. Prepare Input --- 
    console(f"Encoding prompt: '{prompt}'")
    try:
        if not prompt:
             logger.warning("Empty prompt provided. Starting generation potentially from BOS token if available.")
             bos_token_id = getattr(tokenizer, 'bos_token_id', None)
             if bos_token_id is not None:
                  # Base tokenizer encode expects str, returns List[int]
                  # Create tensor manually
                  input_ids_list = [bos_token_id]
                  input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=resolved_device)
             else:
                  logger.error("Empty prompt provided, but tokenizer has no BOS token ID. Cannot start generation.")
                  raise typer.Exit(code=1)
        else:
             # Base tokenizer encode expects str, returns List[int]
             encoded_list = tokenizer.encode(prompt)
             # Create tensor manually
             input_ids = torch.tensor([encoded_list], dtype=torch.long, device=resolved_device)
        
        logger.info(f"Input tensor shape: {input_ids.shape}")

    except Exception as e:
        logger.exception("Failed to encode prompt.")
        raise typer.Exit(code=1)

    # --- 4. Generate --- 
    console("Generating text...")
    try:
        with torch.no_grad():
             eos_token_id = getattr(tokenizer, 'eos_token_id', None)
             if eos_token_id is None:
                  logger.warning("Tokenizer does not have an 'eos_token_id'. Generation might not stop correctly.")
                  
             generative_model = cast(GenerativeModel, model)
             generated_ids = generative_model.generate(
                 input_ids,
                 max_new_tokens=max_new_tokens,
                 temperature=temperature,
                 top_k=top_k,
                 top_p=top_p,
                 repetition_penalty=repetition_penalty,
                 eos_token_id=eos_token_id,
                 verbose=True
             )
        logger.info(f"Output tensor shape: {generated_ids.shape}")

    except NotImplementedError:
         logger.error(f"The loaded model '{type(model).__name__}' has not implemented the required 'generate' method.")
         logger.error("Ensure the model class implements 'generate', potentially by calling 'autoregressive_generate' from craft.utils.generation.")
         raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Error during text generation.")
        raise typer.Exit(code=1)

    # --- 5. Decode and Output --- 
    console("Decoding output...")
    try:
        # Base tokenizer decode expects List[int]
        output_ids_list = generated_ids[0].tolist()
        # Base tokenizer decode does not have skip_special_tokens
        generated_text = tokenizer.decode(output_ids_list)
        
        console("--- Generated Text ---")
        console(generated_text)
        console("----------------------")

    except Exception as e:
        logger.exception("Failed to decode generated tokens.")
        raise typer.Exit(code=1)

    console("Text generation finished.")

# Example usage from terminal:
# python -m craft.cli.run generate text -c /path/to/model/dir_or_checkpoint.pt -p "Hello world" -l 100 -t 0.7 