# src/cli/generate_commands.py
import typer
import logging
from typing import Optional

import torch
from pathlib import Path

# Assuming these utils might be useful
from ..utils.common import set_seed, setup_device
# Need a way to load model and tokenizer based on checkpoint/config
# Placeholder imports - these will need refinement
from ..models.factory import create_model_from_config
from ..utils.io import load_json # Example, might load config from checkpoint
from transformers import AutoTokenizer # Example if using HF tokenizers

# Create Typer app for generation commands
generate_app = typer.Typer(help="Commands for text generation")
logger = logging.getLogger(__name__)
console = typer.echo # Use typer.echo for simple console output in commands

@generate_app.command("text")
def generate_text(
    checkpoint_path: Path = typer.Option(..., "--checkpoint", "-c", help="Path to the model checkpoint (.pt file)", exists=True, file_okay=True, dir_okay=False, readable=True),
    prompt: str = typer.Option("", "--prompt", "-p", help="Text prompt to start generation from."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", "-l", help="Maximum number of new tokens to generate."),
    temperature: float = typer.Option(0.8, "--temperature", "-t", help="Sampling temperature."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Top-k sampling threshold (set k, e.g., 50). Disable = None."),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Top-p (nucleus) sampling threshold (set p, e.g., 0.9). Disable = None."),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility."),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device override (e.g., 'cpu', 'cuda'). Auto-detect if None."),
):
    """Generate text using a trained model checkpoint."""
    
    console(f"Starting text generation using checkpoint: {checkpoint_path}")
    
    # --- 1. Setup --- 
    if seed is not None:
        set_seed(seed)
        console(f"Using random seed: {seed}")
    
    resolved_device = setup_device(device or "auto")
    console(f"Using device: {resolved_device}")

    # --- 2. Load Model and Tokenizer --- 
    # This is the critical part that needs proper implementation
    # We need to load the checkpoint, extract config, load model, AND load the correct tokenizer
    
    try:
        console("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        
        # Attempt to load model config from checkpoint
        # TODO: Refine config loading - might be under 'config', 'model_config', etc.
        model_cfg_dict = checkpoint.get("config", {}).get("model", None)
        if model_cfg_dict is None:
             # Fallback or error - depends on checkpoint save format
             logger.error("Could not find model configuration within the checkpoint.")
             raise typer.Exit(code=1)
             
        console("Creating model...")
        model = create_model_from_config(model_cfg_dict)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)
        model.eval()
        console("Model loaded successfully.")

        # Load Tokenizer / Vocabulary
        # TODO: Standardize how tokenizer/vocab is loaded based on model config
        tokenizer = None
        vocab_path = None # Find vocab path from config if possible
        # Placeholder: logic to determine if it should be AutoTokenizer or vocab file
        if vocab_path:
             logger.warning(f"Loading vocab from {vocab_path} not fully implemented here yet.")
             # Need to load char_to_idx, idx_to_char if character model
             char_to_idx = {} # Placeholder
             idx_to_char = {} # Placeholder
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) # Try loading from checkpoint dir
                logger.info(f"Loaded AutoTokenizer from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load AutoTokenizer from checkpoint dir {checkpoint_path}: {e}. Trying default 'gpt2'.")
                try:
                    tokenizer = AutoTokenizer.from_pretrained('gpt2')
                    logger.info("Loaded default 'gpt2' tokenizer.")
                except Exception as e_gpt2:
                     logger.error(f"Could not load default 'gpt2' tokenizer: {e_gpt2}. Cannot proceed with generation.")
                     return
            # If using AutoTokenizer, we likely don't need char_to_idx/idx_to_char separately
            char_to_idx = None
            idx_to_char = None

        console("Tokenizer placeholder loaded.")

    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"Failed to load model or tokenizer from checkpoint.")
        raise typer.Exit(code=1)

    # --- 3. Prepare Input --- 
    console(f"Encoding prompt: '{prompt}'")
    # Use the loaded tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(resolved_device) if prompt else torch.zeros((1, 1), dtype=torch.long, device=resolved_device)
    if input_ids.nelement() == 0 and prompt: # Handle case where tokenizer returns empty for valid prompt
         logger.error(f"Tokenizer returned empty tensor for non-empty prompt: '{prompt}'")
         raise typer.Exit(code=1)
    if input_ids.nelement() == 0 and not prompt:
         logger.info("No prompt provided, starting with default initial token(s).")
         # TODO: Define a default start token based on tokenizer maybe? Using 0 for now.
         input_ids = torch.tensor([[0]], dtype=torch.long, device=resolved_device)
         

    # --- 4. Generate --- 
    console(f"Generating up to {max_new_tokens} new tokens...")
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                # TODO: Add eos_token_id if available from tokenizer
                # TODO: Add repetition_penalty if desired
            )
            console("Generation complete.")
    except Exception as e:
        logger.exception("Error during model generation.")
        raise typer.Exit(code=1)

    # --- 5. Decode and Print --- 
    console("Decoding output...")
    try:
         generated_text = tokenizer.decode(output_ids[0])
         console("\n--- Generated Text ---")
         # Print prompt differently from generated part if possible
         # This depends on tokenizer/model generate behavior
         console(generated_text)
         console("----------------------")
    except Exception as e:
         logger.exception("Error during decoding.")
         # Still try to print raw IDs
         console("[bold yellow]Decoding failed. Raw output IDs:[/bold yellow]")
         console(str(output_ids[0].tolist()))
         raise typer.Exit(code=1)

    logger.info("Text generation finished successfully.") 