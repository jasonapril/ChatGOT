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
# from transformers import AutoTokenizer # Example if using HF tokenizers

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

        # TODO: Implement Tokenizer Loading!
        # How is the tokenizer defined/saved? 
        # - Is it saved in the checkpoint?
        # - Is its name/path in the config?
        # - Assuming 'tokenizer_name' or 'vocab_path' might be in data config saved in checkpoint
        data_cfg_dict = checkpoint.get("config", {}).get("data", {})
        tokenizer_name_or_path = data_cfg_dict.get("tokenizer_name", None) # Example: Might be 'gpt2'
        vocab_path = data_cfg_dict.get("vocab_path", None) # Example: Might be path to vocab.json
        
        tokenizer = None
        if tokenizer_name_or_path:
            console(f"Attempting to load tokenizer: {tokenizer_name_or_path}")
            # Placeholder: Use AutoTokenizer or load CharDataset vocab
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            logger.warning(f"Tokenizer loading for '{tokenizer_name_or_path}' not fully implemented.")
            # For now, let's assume a mock tokenizer for testing structure
            class MockTokenizer:
                def encode(self, text, return_tensors=None): return torch.randint(0, 50, (1, len(text)), device=resolved_device)
                def decode(self, ids): return "[decoded mock text]"
            tokenizer = MockTokenizer()
        elif vocab_path:
             logger.warning(f"Loading CharDataset vocab from {vocab_path} not implemented here yet.")
             tokenizer = MockTokenizer() # Fallback mock
        else:
             logger.error("Could not determine tokenizer from checkpoint configuration.")
             raise typer.Exit(code=1)
        
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