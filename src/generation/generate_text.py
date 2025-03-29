#!/usr/bin/env python
"""
Text Generation Script
======================

This script loads a trained character-level Transformer model and generates text
based on a starting prompt using the configuration managed by Hydra.
"""

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
from typing import Optional
import hydra.utils

# Assuming model, dataset, and utils are structured appropriately
from src.models.factory import create_model_from_config
from src.data.dataset import CharDataset # Needed for char<->idx mapping
from src.utils import setup_device, set_seed, log_section_header

logger = logging.getLogger(__name__)

def generate(
    model: torch.nn.Module,
    char_to_idx: dict,
    idx_to_char: dict,
    device: torch.device,
    start_prompt: str = "\n",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None, # Optional top-k sampling
    logger: logging.Logger = None
) -> str:
    """
    Generates text using the provided model and settings.

    Args:
        model: The loaded and trained PyTorch model.
        char_to_idx: Mapping from characters to token indices.
        idx_to_char: Mapping from token indices to characters.
        device: The device to run generation on (CPU or CUDA).
        start_prompt: The initial sequence of characters to seed generation.
        max_new_tokens: The maximum number of new tokens (characters) to generate.
        temperature: Softmax temperature for sampling. Lower is more greedy, higher is more random.
                     1.0 means no change. Must be positive.
        top_k: If set, only sample from the top k most likely next tokens.
        logger: The logger instance to use for logging messages.

    Returns:
        The generated text string, including the start prompt.
    """
    if logger:
        logger.info(f"Starting generation with prompt: '{start_prompt!r}'") # Use !r for repr
        logger.info(f"Max new tokens: {max_new_tokens}, Temperature: {temperature}, Top-k: {top_k}")

    # Ensure model is in evaluation mode
    model.eval()

    # Validate temperature
    if temperature <= 0:
        if logger:
            logger.warning("Temperature must be positive. Using temperature=1.0")
        temperature = 1.0

    # Encode the starting prompt
    try:
        start_indices = [char_to_idx[c] for c in start_prompt]
    except KeyError as e:
        if logger:
            logger.error(f"Character '{e}' in start_prompt not found in vocabulary.")
        return "Error: Invalid character in prompt."

    # Convert to tensor, add batch dimension (B=1)
    idx = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)

    # Generation loop
    for _ in range(max_new_tokens):
        # Crop sequence to max_seq_length (model's context window)
        # If idx becomes longer than block_size, only feed the last block_size tokens
        # Ensure model.config exists and has max_seq_length
        if hasattr(model, 'config') and hasattr(model.config, 'max_seq_length'):
            idx_cond = idx[:, -model.config.max_seq_length:]
        else:
            # Fallback or error if model config isn't structured as expected
            if logger:
                logger.warning("Model config or max_seq_length not found, using full sequence.")
            idx_cond = idx

        # Get model predictions (logits)
        logits = model(idx_cond)

        # Focus only on the logits for the very last time step
        logits = logits[:, -1, :] # Shape becomes (B, vocab_size)

        # Apply temperature scaling
        logits = logits / temperature

        # Optional top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # Set logits not in the top k to -infinity
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1) # Shape (B, vocab_size)

        # Sample the next index
        idx_next = torch.multinomial(probs, num_samples=1) # Shape (B, 1)

        # Append the sampled index to the sequence
        idx = torch.cat((idx, idx_next), dim=1) # Shape (B, T+1)

    # Decode the generated indices back to text
    # idx is shape (B, T), we take the first (and only) batch element
    final_indices = idx[0].tolist()
    generated_text = "".join([idx_to_char.get(i, '?') for i in final_indices])

    if logger:
        logger.info("Generation complete.")
    # Return the generated text
    return generated_text


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to orchestrate model loading and text generation."""
    # Pass the logger instance as the first argument
    log_section_header(logger, "GENERATION SETUP")
    logger.info("Configuration loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    # --- Basic Setup ---
    set_seed(cfg.seed)
    # Determine device based on top-level force_cpu flag
    device_setting = "cpu" if cfg.get('force_cpu', False) else "auto"
    device = setup_device(device_setting)

    # --- Load Tokenizer (Character Mappings) ---
    # We need the dataset class to build the vocab, even if we don't use the loader
    # Ensure data paths are resolved correctly relative to the original working dir
    # Use a dummy split config just to get the vocab
    try:
        # Resolve data file path relative to original CWD
        data_cfg = cfg.data # Access data config directly
        # Get file path directly from the 'train' config section
        # Assuming train/val use the same dataset file for vocab generation
        if 'train' not in data_cfg or 'file_path' not in data_cfg.train:
             raise ValueError("Missing 'data.train.file_path' in configuration.")
        relative_file_path = data_cfg.train.file_path

        # Use hydra.utils.to_absolute_path instead of resolve_config_path
        absolute_file_path = hydra.utils.to_absolute_path(relative_file_path)

        logger.info(f"Loading character map from dataset file: {absolute_file_path}")
        # Instantiate dataset only to get mappings
        # Use block_size and vocab_path from data config
        tokenizer_dataset = CharDataset(
            file_path=absolute_file_path,
            block_size=data_cfg.block_size,
            vocab_path=hydra.utils.to_absolute_path(data_cfg.vocab_path) # Add vocab_path here, resolve it too
        )
        char_to_idx = tokenizer_dataset.char_to_idx
        idx_to_char = tokenizer_dataset.idx_to_char
        vocab_size = tokenizer_dataset.vocab_size
        logger.info(f"Character map loaded. Vocab size: {vocab_size}")

    except Exception as e:
        logger.error(f"Failed to load character mappings from dataset: {e}", exc_info=True)
        return # Cannot proceed without mappings

    # --- Load Model ---
    # Pass the logger instance as the first argument
    log_section_header(logger, "LOADING MODEL")
    try:
        model = create_model_from_config(cfg.model)
        logger.info(f"Model '{cfg.model.architecture}' created.")
    except Exception as e:
        logger.error(f"Failed to create model from config: {e}", exc_info=True)
        return

    # --- Load Checkpoint ---
    # Get checkpoint path from config, with error handling
    checkpoint_path_rel = cfg.generation.get('checkpoint_path', None)
    if not checkpoint_path_rel:
        logger.error("Generation checkpoint path ('generation.checkpoint_path') not specified in config.")
        return

    # Resolve the path relative to the original working directory
    # as checkpoints might be saved outside the Hydra run directory in some workflows
    checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path_rel)

    logger.info(f"Attempting to load checkpoint from: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Filter out incompatible keys (e.g., if model architecture changed slightly)
            model_state_dict = checkpoint['model_state_dict']
            # Example filtering (might not be needed here):
            # model_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
            model.load_state_dict(model_state_dict)
            logger.info(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            # Decide whether to proceed with untrained model or exit
            logger.warning("Proceeding with untrained model weights.")
            # return # Or maybe proceed with random weights for testing?
    else:
        logger.warning("Checkpoint file not found. Using untrained model weights.")
        # return # Typically exit if checkpoint is expected

    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Generate Text ---
    # Pass the logger instance as the first argument
    log_section_header(logger, "GENERATING TEXT")
    # TODO: Get generation parameters from config or defaults
    start_prompt = cfg.generation.get('start_prompt', "Hello\n")
    max_new_tokens = cfg.generation.get('max_new_tokens', 200)
    temperature = cfg.generation.get('temperature', 0.8)
    top_k = cfg.generation.get('top_k', None)

    # Perform generation within a no_grad context
    with torch.no_grad():
        generated_output = generate(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            device=device,
            start_prompt=start_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            logger=logger
        )

    # Pass the logger instance as the first argument
    log_section_header(logger, "OUTPUT")
    print(generated_output)


if __name__ == "__main__":
    main()