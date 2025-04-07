# src/cli/generate_commands.py
import typer
import logging
from typing import Optional
import yaml  # Import YAML
from omegaconf import OmegaConf # Import OmegaConf for instantiation

import torch
from pathlib import Path
import hydra # Import hydra for utils

# Assuming these utils might be useful
from ..utils.common import set_seed, setup_device
# Need a way to load model and tokenizer based on checkpoint/config
# --- Removed old model factory import ---
# Import tokenizer factory and specific tokenizers
from ..data.tokenizers import create_tokenizer, CharTokenizer, SentencePieceTokenizer # Add necessary imports
# Import TextGenerator
from ..training.generation import TextGenerator
# Import manual sampling as a fallback if needed, though TextGenerator is preferred
from ..training.sampling import generate_text_manual_sampling

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
    checkpoint_dir = checkpoint_path.parent # Get directory containing the checkpoint
    
    # --- 1. Setup --- 
    if seed is not None:
        set_seed(seed)
        console(f"Using random seed: {seed}")
    
    resolved_device = setup_device(device or "auto")
    console(f"Using device: {resolved_device}")

    # --- 2. Load Model and Tokenizer --- 
    try:
        # --- Load Configuration ---
        model_cfg_dict = None
        tokenizer_cfg_dict = None
        full_config_loaded = None
        
        config_yaml_path = checkpoint_dir / "config.yaml"
        
        if config_yaml_path.exists():
            logger.info(f"Found config.yaml at {config_yaml_path}. Loading configuration from YAML.")
            try:
                with open(config_yaml_path, 'r') as f:
                    full_config_loaded = yaml.safe_load(f)
                
                # Extract model and tokenizer config (adjust keys based on actual YAML structure)
                if 'experiment' in full_config_loaded:
                    if 'model' in full_config_loaded['experiment']:
                        model_cfg_dict = full_config_loaded['experiment']['model']
                        logger.info("Model config extracted from config.yaml.")
                    else:
                        logger.warning("'experiment.model' key not found in config.yaml.")
                        
                    # Attempt to get tokenizer config - might be under 'data' or 'tokenizer'
                    if 'data' in full_config_loaded['experiment'] and 'tokenizer' in full_config_loaded['experiment']['data']:
                         tokenizer_cfg_dict = full_config_loaded['experiment']['data']['tokenizer']
                         logger.info("Tokenizer config extracted from config.yaml under experiment.data.tokenizer.")
                    elif 'tokenizer' in full_config_loaded['experiment']: # Check if it's directly under experiment
                         tokenizer_cfg_dict = full_config_loaded['experiment']['tokenizer']
                         logger.info("Tokenizer config extracted from config.yaml under experiment.tokenizer.")
                    else:
                         logger.warning("Tokenizer config not found in expected locations within config.yaml.")
                         
                else:
                    logger.warning("'experiment' key not found in config.yaml.")

            except yaml.YAMLError as e:
                logger.error(f"Error parsing {config_yaml_path}: {e}. Will attempt to load config from checkpoint.")
            except Exception as e:
                 logger.error(f"Unexpected error loading {config_yaml_path}: {e}. Will attempt to load config from checkpoint.")
        else:
            logger.warning(f"{config_yaml_path} not found. Attempting to load configuration from checkpoint file.")

        # --- Fallback/Load from Checkpoint if YAML failed or model config still missing ---
        if model_cfg_dict is None:
            logger.info("Loading checkpoint file to extract configuration...")
            checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
            logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            full_config_ckpt = checkpoint.get("config")
            if full_config_ckpt:
                 model_cfg_dict = full_config_ckpt.get("model")
                 if model_cfg_dict:
                     logger.info("Model config successfully loaded from checkpoint file.")
                 else:
                     logger.error("'model' key not found within the 'config' dictionary in the checkpoint.")
                     logger.error(f"Available keys under 'config': {list(full_config_ckpt.keys())}")
            else:
                 logger.error("'config' key not found in the checkpoint dictionary.")
                 
            # Attempt to get tokenizer config from checkpoint as well (less likely to be complete)
            if tokenizer_cfg_dict is None and full_config_ckpt:
                 # Check common locations
                 if 'data' in full_config_ckpt and isinstance(full_config_ckpt['data'], dict) and 'tokenizer' in full_config_ckpt['data']:
                     tokenizer_cfg_dict = full_config_ckpt['data']['tokenizer']
                     logger.info("Tokenizer config loaded from checkpoint under config.data.tokenizer.")
                 elif 'tokenizer' in full_config_ckpt: # Direct key?
                      tokenizer_cfg_dict = full_config_ckpt['tokenizer']
                      logger.info("Tokenizer config loaded from checkpoint under config.tokenizer.")

        # Final check if model config is available
        if model_cfg_dict is None:
             logger.error("Failed to load model configuration from both config.yaml and checkpoint file.")
             raise typer.Exit(code=1)

        # --- Load Tokenizer FIRST ---
        tokenizer = None
        tokenizer_dir = checkpoint_dir / "tokenizer"
        logger.info(f"Looking for tokenizer data in: {tokenizer_dir}")

        if tokenizer_cfg_dict:
            logger.info(f"Attempting to create tokenizer from config: {tokenizer_cfg_dict.get('_target_', 'N/A')}")
            try:
                tokenizer = create_tokenizer(tokenizer_cfg_dict)
                if hasattr(tokenizer, 'load') and callable(getattr(tokenizer, 'load')):
                     try:
                         # Handle different load signatures
                         if isinstance(tokenizer, SentencePieceTokenizer):
                             model_prefix = str(tokenizer_dir / "spm") # Assuming default prefix
                             tokenizer.load(model_prefix)
                             logger.info(f"Successfully loaded SentencePieceTokenizer data from {model_prefix}")
                         elif isinstance(tokenizer, CharTokenizer):
                             tokenizer.load(str(tokenizer_dir)) # CharTokenizer expects directory
                             logger.info(f"[generate_commands] After CharTokenizer.load, vocab_size = {getattr(tokenizer, 'vocab_size', 'N/A')}")
                             logger.info(f"Successfully inferred and loaded CharTokenizer from {tokenizer_dir}")
                         else:
                             tokenizer.load(str(tokenizer_dir))
                             logger.info(f"Loaded data for {type(tokenizer).__name__} from {tokenizer_dir} (assuming directory path)")
                     except Exception as load_exc:
                          logger.error(f"Failed to load data for tokenizer {type(tokenizer).__name__} from {tokenizer_dir}: {load_exc}", exc_info=True)
                          raise typer.Exit(code=1)
                else:
                     logger.info(f"Instantiated tokenizer '{type(tokenizer).__name__}' directly from config (no load method needed/called).")
            except Exception as e:
                logger.error(f"Failed to create or load tokenizer from config and path {tokenizer_dir}: {e}", exc_info=True)
                raise typer.Exit(code=1)
        
        elif tokenizer_dir.is_dir():
             logger.warning("Tokenizer config not found, but tokenizer directory exists. Attempting to infer.")
             if (tokenizer_dir / "vocab.json").exists() and (tokenizer_dir / "tokenizer_config.json").exists():
                  logger.info("Found vocab.json/tokenizer_config.json, assuming CharTokenizer.")
                  try:
                       # Assign the result of the classmethod load
                       tokenizer = CharTokenizer.load(str(tokenizer_dir))
                       logger.info(f"[generate_commands] After CharTokenizer.load assignment, vocab_size = {getattr(tokenizer, 'vocab_size', 'N/A')}")
                       logger.info(f"Successfully inferred and loaded CharTokenizer from {tokenizer_dir}")
                  except Exception as e:
                       logger.error(f"Failed loading inferred CharTokenizer from {tokenizer_dir}: {e}", exc_info=True)
                       raise typer.Exit(code=1)
             elif (tokenizer_dir / "spm.model").exists(): # Example for SentencePiece
                  logger.info("Found spm.model, assuming SentencePieceTokenizer.")
                  try:
                      # Assign the result of the classmethod load
                      model_prefix = str(tokenizer_dir / "spm") # Assuming default prefix 'spm'
                      tokenizer = SentencePieceTokenizer.load(model_prefix) 
                      logger.info(f"[generate_commands] After SentencePieceTokenizer.load assignment, vocab_size = {getattr(tokenizer, 'vocab_size', 'N/A')}")
                      logger.info(f"Successfully inferred and loaded SentencePieceTokenizer from prefix {model_prefix}")
                  except Exception as e:
                      logger.error(f"Failed loading inferred SentencePieceTokenizer from {tokenizer_dir}: {e}", exc_info=True)
                      raise typer.Exit(code=1)
             else:
                  logger.error(f"Could not infer tokenizer type from contents of {tokenizer_dir}")
                  raise typer.Exit(code=1)
        else:
             logger.error(f"Tokenizer configuration not found AND tokenizer directory '{tokenizer_dir}' does not exist.")
             raise typer.Exit(code=1)
             
        # Ensure tokenizer is loaded
        if tokenizer is None:
             logger.error("Tokenizer could not be loaded.")
             raise typer.Exit(code=1)
        console("Tokenizer loaded successfully.")
        # --- END Load Tokenizer ---

        # --- Inject vocab_size into Model Config ---
        if model_cfg_dict is None:
            logger.error("Model configuration dictionary is None after config loading attempts.")
            raise typer.Exit(code=1)
            
        # Check if vocab_size is missing at the top level of the model config dict
        if model_cfg_dict.get('vocab_size') is None:
            if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size is not None:
                logger.info(f"Injecting vocab_size={tokenizer.vocab_size} from loaded tokenizer into model_cfg_dict['vocab_size'].")
                # Inject directly into the model config dict
                model_cfg_dict['vocab_size'] = tokenizer.vocab_size
            else:
                logger.error(f"Model config missing 'vocab_size' and could not retrieve it from tokenizer {type(tokenizer).__name__}. Cannot proceed.")
                raise typer.Exit(code=1)
        else:
             logger.info(f"Model config already contains vocab_size: {model_cfg_dict['vocab_size']}")
        # --- END Inject vocab_size ---

        # --- Create Model using Hydra instantiate (assuming _target_ is in config) ---
        console("Creating model...")
        try:
            # Convert dict to OmegaConf object for instantiate
            model_cfg_omega = OmegaConf.create(model_cfg_dict)
            model = hydra.utils.instantiate(model_cfg_omega)
            logger.info(f"Model instantiated successfully using hydra.utils.instantiate: {type(model).__name__}")
        except Exception as instantiate_err:
            logger.error(f"Failed to instantiate model using hydra.utils.instantiate: {instantiate_err}", exc_info=True)
            logger.error("Ensure the model configuration dictionary includes a valid '_target_' key.")
            raise typer.Exit(code=1)

        # --- Load Model State ---
        # Load checkpoint again only if not loaded before (if config came from YAML)
        if 'checkpoint' not in locals(): 
            logger.info("Loading checkpoint file to extract model state_dict...")
            checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
            
        if "model_state_dict" not in checkpoint:
             logger.error("Checkpoint file does not contain 'model_state_dict'.")
             raise typer.Exit(code=1)
             
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)
        model.eval()
        console("Model loaded successfully.")

    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"Failed to load model or tokenizer.") # logger.exception includes traceback
        raise typer.Exit(code=1)

    # --- 3. Prepare Input --- 
    console(f"Encoding prompt: '{prompt}'")
    try:
        if not prompt:
            console("No prompt provided, starting generation from scratch (or model's default).", style="yellow")
            # input_ids handled by TextGenerator
        else:
            console(f"Using prompt: '{prompt}'")
            # Encoding is handled internally by TextGenerator
    except Exception as e:
        logger.exception("Failed during prompt preparation/logging stage.") # Should not happen often here
        raise typer.Exit(code=1)

    # --- 4. Generate using TextGenerator ---
    console("Generating text...")
    try:
        # Ensure model is in eval mode
        model.eval()

        # Instantiate TextGenerator
        text_generator = TextGenerator(model=model, tokenizer=tokenizer)

        # Generate text using the generator
        generated_texts = text_generator.generate_text(
            prompts=[prompt] if prompt else [""], # Pass prompt as a list
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            # device=resolved_device, # Device handling might be internal to TextGenerator now
        )

        # Assuming generate_text returns a list, get the first element
        if not generated_texts:
            logger.error("Text generation returned an empty list.")
            raise typer.Exit(code=1)

        generated_text = generated_texts[0] # Get the first result
        logger.info("Generation via TextGenerator successful.")

    except Exception as e:
        logger.exception("Text generation failed.")
        raise typer.Exit(code=1)

    # --- 5. Output ---
    console("\n--- Generated Text ---", style="bold green")
    console(generated_text)
    console("--- End Generation ---")

    logger.info("Text generation finished successfully.") 