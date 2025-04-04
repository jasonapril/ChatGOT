import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import os
import sys
from pydantic import ValidationError
from typing import Optional, Any
from typing import Optional

# Add project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Delayed imports like in train.py
try:
    from craft.utils.common import setup_device
    from craft.config.schemas import GenerationConfig, AppConfig # Import relevant Pydantic schemas
    # Model/Data imports will be needed for actual generation
    from craft.models import create_model_from_config, Model
    # Import tokenizer classes and base class/type hint
    from craft.data.tokenizers import CharLevelTokenizer, SentencePieceTokenizer, SubwordTokenizer # Add others as needed
    from craft.data.tokenizers.base import BaseTokenizer
    from typing import Any # For tokenizer type hint initially
except ImportError as e:
    logging.error(f"Failed to import necessary components: {e}. Ensure 'craft' is installed.")
    sys.exit(1)

# Actual generation function
@torch.no_grad() # Ensure gradients are not computed during generation
def generate_text(model: Model, 
                  tokenizer: BaseTokenizer, 
                  prompt: str, 
                  device: str, 
                  gen_cfg: GenerationConfig) -> str:
    logger = logging.getLogger(__name__)
    logger.info("Generating text...")
    logger.info(f"  Prompt: '{prompt}'")
    logger.info(f"  Max New Tokens: {gen_cfg.max_new_tokens}")
    logger.info(f"  Temperature: {gen_cfg.temperature}")
    logger.info(f"  Top-k: {gen_cfg.top_k}")
    
    try:
        # 1. Encode prompt
        # Custom tokenizers might not support return_tensors directly
        logger.debug(f"Encoding prompt using {type(tokenizer).__name__}...")
        encoded_ids = tokenizer.encode(prompt)
        if not isinstance(encoded_ids, list):
             logger.warning(f"Tokenizer encode did not return a list, got {type(encoded_ids)}. Attempting to proceed.")
             # Try to handle tensor output directly if it happens
             if isinstance(encoded_ids, torch.Tensor):
                 input_ids_unbatched = encoded_ids
             else:
                 # Attempt conversion if possible, otherwise error
                 try:
                      input_ids_unbatched = torch.tensor(encoded_ids, dtype=torch.long)
                 except Exception as e_conv:
                      logger.error(f"Could not convert tokenizer output to tensor: {e_conv}")
                      return "[ERROR: Tokenizer output conversion failed]"
        else:
            input_ids_unbatched = torch.tensor(encoded_ids, dtype=torch.long)
            
        # Add batch dimension and move to device
        input_ids = input_ids_unbatched.unsqueeze(0).to(device)
        logger.debug(f"Prompt encoded to tensor shape: {input_ids.shape}")

        # Add attention mask if tokenizer provides it and model uses it
        # attention_mask = encoded_input.get('attention_mask', None)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(device)
        
        logger.info(f"Encoded prompt shape: {input_ids.shape}")

        # 2. Generate token IDs using the model
        # Assuming model has a generate method compatible with Hugging Face style
        # This is a HYPOTHETICAL call signature - we may need to adapt our Model base class later.
        if not hasattr(model, 'generate') or not callable(getattr(model, 'generate')):
            logger.error("The loaded model does not have a callable 'generate' method.")
            return "[ERROR: Model lacks generate method]"
            
        # Prepare generation arguments from config
        generate_kwargs = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "top_k": gen_cfg.top_k,
            # Add other common parameters if needed (top_p, do_sample, etc.)
            # "do_sample": True, # Often needed for temperature/top_k to have effect
            # "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256 # Example
        }
        # Remove None values as generate might not handle them
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
        
        logger.info(f"Calling model.generate with kwargs: {generate_kwargs}")

        # Ensure model is in eval mode (redundant if set in main, but safe)
        model.eval()

        output_ids = model.generate(input_ids, **generate_kwargs)
        logger.info(f"Generated output shape: {output_ids.shape}")

        # 3. Decode generated token IDs
        # Handle potential case where output includes prompt tokens
        # Assuming output_ids shape is [batch_size, sequence_length]
        if output_ids.shape[0] != 1:
             logger.warning("Generation output has unexpected batch size > 1. Decoding only the first sequence.")
        
        # Slice the generated tokens (excluding prompt) - assuming output includes prompt
        generated_token_ids = output_ids[0, input_ids.shape[1]:] 
        
        # --- Debug: Print Raw Token IDs --- 
        # Convert to list for consistent type before logging/decoding
        generated_ids_list = generated_token_ids.tolist()
        logger.info(f"Raw generated token IDs ({len(generated_ids_list)}): {generated_ids_list}")
        # --- End Debug ---

        # --- REMOVED Debug: Print input to decode ---
        # print(f"DEBUG: Input to tokenizer.decode (type {type(generated_ids_list)}): {generated_ids_list}")
        # --- End REMOVED Debug ---
        
        # Assuming tokenizer has a decode method
        logger.debug(f"Decoding {len(generated_ids_list)} tokens using {type(tokenizer).__name__}...")
        generated_text = tokenizer.decode(generated_ids_list)
        
        # --- REMOVED Debug: Print output from decode ---
        # print(f"DEBUG: Output from tokenizer.decode (type {type(generated_text)}): {repr(generated_text)}")
        # --- End REMOVED Debug ---
        
        # --- Debug: Print repr() of decoded text --- 
        # This logger call might fail on some consoles due to encoding
        try:
            logger.info(f"Decoded text (repr): {repr(generated_text)}")
        except UnicodeEncodeError:
             logger.warning("Could not log repr of decoded text due to console encoding issues.")
        # --- End Debug ---

        logger.info("Decoding complete.")
        return generated_text

    except AttributeError as ae:
        logger.error(f"AttributeError during generation: {ae}. Check tokenizer/model methods (encode/generate/decode).", exc_info=True)
        return f"[ERROR: AttributeError - {ae}]"
    except Exception as e:
        logger.error(f"An unexpected error occurred during text generation: {e}", exc_info=True)
        return "[ERROR: Generation failed]"

@hydra.main(config_path="../conf", config_name="generate", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main generation script execution using Hydra."""
    logger = logging.getLogger(__name__)
    logger.info("Starting generation script...")
    logger.info(f"Generation Config:\n{OmegaConf.to_yaml(cfg)}")
    current_dir = os.getcwd() # Hydra changes CWD, capture it if needed for output
    logger.info(f"Hydra CWD: {current_dir}") 

    try:
        # --- Validate Generation Config --- #
        logger.info("Validating generation parameters...")
        try:
            # Validate only the relevant part (generation + device)
            # We could create a specific GenerationScriptConfig Pydantic model if needed
            gen_params = OmegaConf.to_container(cfg.generation, resolve=True)
            validated_gen_cfg = GenerationConfig(**gen_params)
            device = setup_device(cfg.device)
            logger.info("Generation parameters validation successful.")
            logger.info(f"Using device: {device}")
        except ValidationError as e:
            logger.error(f"Generation configuration validation failed!\n{e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error processing generation config: {e}", exc_info=True)
            sys.exit(1)

        # --- Load Configuration from Training Run --- #
        if cfg.load_from_run is None:
            logger.error("Missing required parameter: `load_from_run`. Specify the path to the training run directory.")
            sys.exit(1)
        
        # Construct path relative to the *original* CWD if necessary, or assume absolute/relative to project root
        # For simplicity, assume load_from_run is relative to project root or absolute
        run_dir = os.path.abspath(os.path.join(project_root, cfg.load_from_run))
        if not os.path.isdir(run_dir):
            logger.error(f"Specified run directory does not exist: {run_dir}")
            sys.exit(1)

        training_cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.isfile(training_cfg_path):
            logger.error(f"Cannot find original training config: {training_cfg_path}")
            sys.exit(1)
        
        logger.info(f"Loading original training config from: {training_cfg_path}")
        training_cfg = OmegaConf.load(training_cfg_path)
        logger.info("Original training config loaded.")
        # Optionally, could validate this loaded config using AppConfig as well
        # try:
        #     AppConfig(**OmegaConf.to_container(training_cfg, resolve=True))
        #     logger.info("Validation of loaded training config successful.")
        # except ValidationError as e:
        #     logger.error(f"Loaded training config validation failed: {e}")
        #     # Decide whether to proceed or exit
        #     sys.exit(1)

        # --- Load Tokenizer (Revised: Load from checkpoint run directory) --- #
        logger.info("Attempting to load tokenizer...")
        tokenizer: Optional[BaseTokenizer] = None
        tokenizer_path = os.path.join(run_dir, "tokenizer") # Path within the specified run directory

        logger.info(f"Looking for tokenizer artifacts in: {tokenizer_path}")
        if os.path.isdir(tokenizer_path):
            try:
                # Try to determine tokenizer type from training config if possible
                # This makes it more general than hardcoding CharLevelTokenizer
                tokenizer_cls = None
                if training_cfg.data and hasattr(training_cfg.data, 'tokenizer') and hasattr(training_cfg.data.tokenizer, '_target_'):
                    target_cls_path = training_cfg.data.tokenizer._target_
                    logger.info(f"Attempting to load tokenizer type '{target_cls_path}' based on training config.")
                    try:
                        tokenizer_cls = hydra.utils.get_class(target_cls_path)
                    except ImportError:
                         logger.error(f"Could not import tokenizer class from training config: {target_cls_path}")
                
                # Fallback or if class found
                if tokenizer_cls and hasattr(tokenizer_cls, 'load') and callable(getattr(tokenizer_cls, 'load')):
                    tokenizer = tokenizer_cls.load(tokenizer_path)
                    logger.info(f"Successfully loaded tokenizer type {type(tokenizer).__name__} from {tokenizer_path}")
                elif not tokenizer_cls: # If config didn't specify class, try CharLevel as default for now
                     logger.warning(f"Could not determine tokenizer type from training config. Trying CharLevelTokenizer as default.")
                     try:
                         tokenizer = CharLevelTokenizer.load(tokenizer_path)
                         logger.info(f"Successfully loaded default CharLevelTokenizer from {tokenizer_path}")
                     except Exception as char_e:
                         logger.error(f"Failed to load default CharLevelTokenizer from {tokenizer_path}: {char_e}")
                else:
                     logger.error(f"Determined tokenizer class '{target_cls_path}' but it lacks a callable 'load' method.")

            except Exception as e:
                logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}", exc_info=True)
                tokenizer = None # Ensure it's None if loading fails
        else:
            logger.warning(f"Tokenizer directory not found in run directory: {tokenizer_path}")
            tokenizer = None

        if tokenizer is None:
            logger.error("Failed to load tokenizer from run directory. Cannot proceed.")
            sys.exit(1)

        # Get vocab_size from tokenizer *after* loading it
        try:
            vocab_size = tokenizer.vocab_size
            logger.info(f"Retrieved vocab_size from tokenizer: {vocab_size}")
        except AttributeError:
            logger.error("Loaded tokenizer does not have a 'vocab_size' attribute.")
            sys.exit(1)

        # --- Create Model Architecture --- #
        logger.info("Creating model architecture based on loaded training config...")
        # Use the factory function with the model section of the loaded training config
        # Pass the retrieved vocab_size to the factory function
        model = create_model_from_config(training_cfg.model, vocab_size=vocab_size)
        model.eval() # Set model to evaluation mode
        model.to(device)
        logger.info(f"Model architecture created: {model.__class__.__name__}")

        # --- Load Model Checkpoint --- #
        checkpoint_path = os.path.join(run_dir, cfg.checkpoint_name)
        if not os.path.isfile(checkpoint_path):
             # Checkpoints might be saved in a 'checkpoints' subdir
             checkpoint_path_alt = os.path.join(run_dir, "checkpoints", cfg.checkpoint_name)
             if os.path.isfile(checkpoint_path_alt):
                 checkpoint_path = checkpoint_path_alt
             else:
                 logger.error(f"Checkpoint file not found at {os.path.join(run_dir, cfg.checkpoint_name)} or {checkpoint_path_alt}")
                 sys.exit(1)
        
        logger.info(f"Loading model state dict from: {checkpoint_path}")
        try:
            # Use the model's load method (which handles state dict loading)
            # Pass strict=False initially might be safer if config slightly changed
            loaded_data = model.load(checkpoint_path, device=device, strict=True)
            logger.info(f"Model weights loaded successfully.")
            # loaded_data contains other checkpoint info if needed
        except Exception as e:
            logger.error(f"Failed to load model checkpoint: {e}", exc_info=True)
            sys.exit(1)

        # --- Generate Samples --- # 
        logger.info("Starting generation...")
        # Use the validated generation config and the current device setting
        generated_sequence = generate_text(
            model=model,
            tokenizer=tokenizer, # Pass the loaded tokenizer
            prompt=validated_gen_cfg.start_prompt, 
            device=device,
            gen_cfg=validated_gen_cfg
        )
        logger.info("Generation complete.")

        # --- Output --- #
        print("\n--- Generated Output ---")
        print(generated_sequence)
        print("------------------------")

    except Exception as e:
        logger.exception(f"An error occurred during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 