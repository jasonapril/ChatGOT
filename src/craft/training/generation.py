"""
Text Generation Module
======================

This module provides classes and functions for generating text sequences using trained models.
It supports various sampling strategies like greedy, temperature, top-k, and top-p sampling.
"""

import torch
import logging
import time  # Import the time module
from typing import Dict, List, Any, Optional, Union
from unittest.mock import MagicMock
from ..data.tokenizers.base import Tokenizer
from omegaconf import DictConfig

from ..models.base import Model

class TextGenerator:
    """Generates text using a trained model."""

    def __init__(self, model: torch.nn.Module, device: torch.device, dataset: Optional[Any] = None, tokenizer: Optional[Tokenizer] = None, config: Optional[Union[Dict[str, Any], "DictConfig"]] = None):
        """
        Initializes the TextGenerator.

        Args:
            model (torch.nn.Module): The model to use for generation.
            device (torch.device): The device to run generation on.
            dataset (Optional[Any]): The dataset object, used for encoding/decoding if tokenizer not provided explicitly.
            tokenizer (Optional[Tokenizer]): Explicit tokenizer to use. If provided, this takes precedence over dataset.tokenizer.
            config (Optional[Union[Dict[str, Any], "DictConfig"]]): Configuration dictionary (optional).
        """
        self.model = model
        self.device = device
        self.dataset = dataset # Store dataset for potential fallback
        self.tokenizer = tokenizer # Prioritize explicitly passed tokenizer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Move model to the specified device
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

        # Warn if model doesn't have a generate method (unless it's mocked)
        if not hasattr(model, 'generate') and not isinstance(model, MagicMock):
            self.logger.warning(
                "The provided model does not have a 'generate' method. "
                "Text generation capabilities might be limited or unavailable."
            )
        
        self.logger.info(f"TextGenerator initialized. Using device: {self.device}")
        if self.tokenizer:
            self.logger.info(f"Using explicitly provided tokenizer: {type(self.tokenizer)}")
        elif self.dataset:
            ds_tokenizer = getattr(self.dataset, 'tokenizer', None)
            if ds_tokenizer:
                self.logger.info(f"Using tokenizer found in dataset: {type(ds_tokenizer)}")
            else:
                 self.logger.info("No explicit tokenizer provided, will rely on dataset for encoding/decoding if possible.")
        else:
             self.logger.warning("No explicit tokenizer or dataset provided.")

    def generate_text(
        self,
        start_prompt: str = "The ",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate text using the model's built-in .generate() method.

        Args:
            start_prompt: The initial text to start generation from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to consider (set to None or 0 to disable)
            top_p: Cumulative probability threshold for sampling (set to None or 1.0 to disable)
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to generate
            pad_token_id: Token ID for padding. Tries to infer from model/tokenizer if None.
            eos_token_id: Token ID for end of sequence. Tries to infer from model/tokenizer if None.
            repetition_penalty: Penalty for repeating tokens (1.0 means no penalty).
            length_penalty: Exponential penalty to the length (1.0 means no penalty).
            no_repeat_ngram_size: If set > 0, all ngrams of that size can only occur once.
            early_stopping: Whether to stop generation when EOS is produced.
            **kwargs: Additional keyword arguments passed directly to model.generate().

        Returns:
            List of generated text sequences (excluding the prompt).
        """
        if not hasattr(self.model, 'generate'):
            raise NotImplementedError(f"Model {type(self.model).__name__} does not have a .generate() method.")
            
        self.model.eval()
        
        try:
            # 1. Determine input_ids from start_prompt
            input_ids = None
            encoding_source = "None"
            prompt_to_encode = start_prompt or "" # Ensure non-None prompt
            
            # --- Select Tokenizer --- # 
            tokenizer_to_use = None
            if self.tokenizer: # Prioritize explicit tokenizer
                tokenizer_to_use = self.tokenizer
                encoding_source = "explicit self.tokenizer"
            elif self.dataset and hasattr(self.dataset, 'tokenizer'):
                tokenizer_to_use = self.dataset.tokenizer
                encoding_source = "dataset.tokenizer"
            else:
                self.logger.debug("No explicit or dataset tokenizer available.")
            
            self.logger.debug(f"Attempting encoding using: {encoding_source}")
            # --- Encoding Logic --- #
            if tokenizer_to_use and hasattr(tokenizer_to_use, 'encode') and callable(getattr(tokenizer_to_use, 'encode')):
                try:
                    encoded = tokenizer_to_use.encode(prompt_to_encode)
                    self.logger.debug(f"generate_text: {encoding_source}.encode result: {encoded}")
                    self.logger.debug(f"generate_text: {encoding_source}.encode result type: {type(encoded)}")
                    if isinstance(encoded, list):
                        if encoded and isinstance(encoded[0], list): # batch encoding?
                             input_ids = torch.tensor(encoded[0], dtype=torch.long).unsqueeze(0)
                        else: # single sequence
                             input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
                        self.logger.debug(f"Encoded using {encoding_source}. input_ids: {input_ids}")
                    else:
                        self.logger.warning(f"{encoding_source}.encode did not return a list, got: {type(encoded)}. input_ids remains None.")
                except Exception as e:
                    self.logger.warning(f"{encoding_source}.encode failed: {e}. Trying char_to_idx fallback.")
                    input_ids = None # Ensure reset
            elif self.dataset and hasattr(self.dataset, 'char_to_idx') and self.dataset.char_to_idx:
                # Fallback to char_to_idx from dataset
                self.logger.debug("Falling back to dataset.char_to_idx for encoding.")
                try:
                    input_ids = torch.tensor(
                         [self.dataset.char_to_idx.get(ch, self.dataset.char_to_idx.get('<unk>', 0)) for ch in prompt_to_encode],
                         dtype=torch.long
                     ).unsqueeze(0)
                    encoding_source = "dataset.char_to_idx"
                    self.logger.debug(f"Encoded using {encoding_source}. input_ids: {input_ids}")
                except Exception as e:
                    self.logger.error(f"Error during char_to_idx encoding: {e}")
                    input_ids = None
            else:
                # No encoding method available
                self.logger.error("Cannot encode prompt: No usable tokenizer (explicit or in dataset) or char_to_idx mapping found.")
                raise ValueError("Dataset must provide either a callable tokenizer.encode method or a char_to_idx mapping.")

            # Move input_ids to the correct device
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            else:
                # Handle case where encoding completely failed
                 self.logger.error("Input encoding failed, cannot proceed with generation.")
                 # Return error message in a list, similar to decode error handling
                 return [f"[Generation Error: Input encoding failed for prompt '{start_prompt}']"] 

            # --- Determine Special Token IDs --- #
            # Priority: Explicitly passed > Tokenizer > Model Config > Char Fallback (EOS only)

            # PAD Token
            eff_pad_token_id = pad_token_id
            if eff_pad_token_id is None:
                if hasattr(self.dataset, 'tokenizer') and hasattr(self.dataset.tokenizer, 'pad_token_id') and self.dataset.tokenizer.pad_token_id is not None:
                    eff_pad_token_id = self.dataset.tokenizer.pad_token_id
                elif hasattr(self.model, 'config') and hasattr(self.model.config, 'pad_token_id'):
                    eff_pad_token_id = self.model.config.pad_token_id
            
            # EOS Token
            eff_eos_token_id = eos_token_id
            if eff_eos_token_id is None:
                # Try tokenizer first
                tokenizer = getattr(self.dataset, 'tokenizer', None)
                if tokenizer:
                    eff_eos_token_id = getattr(tokenizer, 'eos_token_id', None)
                
                # If still None, try model config
                if eff_eos_token_id is None:
                    model_config = getattr(self.model, 'config', None)
                    if model_config:
                        eff_eos_token_id = getattr(model_config, 'eos_token_id', None)
                
                # If still None, try char_to_idx fallback
                if eff_eos_token_id is None:
                    char_map = getattr(self.dataset, 'char_to_idx', None)
                    if isinstance(char_map, dict):
                        eff_eos_token_id = char_map.get('<eos>', None)
            
            self.logger.debug(f"Effective PAD token ID: {eff_pad_token_id}")
            self.logger.debug(f"Effective EOS token ID: {eff_eos_token_id}")

            # --- Generation Configuration --- #
            # Filter out None values for top_k/top_p if not applicable
            generation_kwargs = kwargs.copy()
            generation_kwargs.update({
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "early_stopping": early_stopping,
            })
            
            # Only include sampling parameters if do_sample is True
            if do_sample:
                 generation_kwargs["temperature"] = temperature
                 if top_k is not None and top_k > 0:
                     generation_kwargs["top_k"] = top_k
                 if top_p is not None and 0.0 < top_p < 1.0:
                     generation_kwargs["top_p"] = top_p
            else: # Greedy decoding doesn't use these
                 generation_kwargs.pop("temperature", None)
                 generation_kwargs.pop("top_k", None)
                 generation_kwargs.pop("top_p", None)
            
            # Remove internal 'prompt' if it exists in kwargs before passing to model
            generation_kwargs.pop("prompt", None) 
            
            # Add pad/eos only if they are valid IDs
            if eff_pad_token_id is not None:
                 generation_kwargs["pad_token_id"] = eff_pad_token_id
            if eff_eos_token_id is not None:
                 generation_kwargs["eos_token_id"] = eff_eos_token_id
                 
            self.logger.info(f"Calling model.generate() with config: {generation_kwargs}")

            # --- Generate sequences --- #
            start_time = time.time()
            with torch.no_grad():
                # Ensure input_ids are on the correct device
                input_ids_tensor = input_ids.to(self.device)
                self.logger.debug(f"Input tensor shape for generate: {input_ids_tensor.shape}")

                # Use the generation_kwargs prepared *before* the no_grad block
                outputs = self.model.generate(
                    input_ids=input_ids_tensor,
                    **generation_kwargs 
                )
                self.logger.debug(f"Model output tensor shape: {outputs.shape}")

                # outputs includes the prompt; extract only the generated part
                # Shape: (batch_size, generated_sequence_length)
                generated_ids = outputs[:, input_ids_tensor.shape[1]:]
                self.logger.debug(f"Generated IDs tensor shape (excluding prompt): {generated_ids.shape}")

                # Move generated IDs to CPU for decoding
                generated_ids_list = generated_ids.cpu().tolist() # List[List[int]]
                self.logger.debug(f"Generated IDs (list): {generated_ids_list}")

            gen_time = time.time() - start_time
            self.logger.info(f"Model.generate() completed in {gen_time:.2f}s")

            # --- Decoding --- #
            generated_texts = []
            if not hasattr(self.dataset, 'decode'):
                 self.logger.error("Dataset does not have a required 'decode' method.")
                 return ["Error: Decoding not possible."] * num_return_sequences
                 
            for ids in generated_ids_list:
                # Decode, handling potential errors
                generated_text = "[Decoding Error]" # Default value
                try:
                    # --- Decoding Logic --- #
                    # Priority: self.tokenizer -> self.dataset.decode -> Error
                    # Try explicit tokenizer first
                    if self.tokenizer and hasattr(self.tokenizer, 'decode') and callable(getattr(self.tokenizer, 'decode')):
                        # Use explicitly provided tokenizer
                        self.logger.debug("Using self.tokenizer.decode")
                        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=True)
                        if not decoded_text: # Try without skipping specials if first attempt was empty
                            self.logger.warning(f"Decoding with skip_special_tokens=True failed or returned empty. Trying without.")
                            decoded_text = self.tokenizer.decode(ids, skip_special_tokens=False)
                    # Fallback to dataset decode method
                    elif self.dataset and hasattr(self.dataset, 'decode') and callable(getattr(self.dataset, 'decode')):
                        # Fallback to dataset decode method
                        self.logger.debug("Using self.dataset.decode")
                        decoded_text = self.dataset.decode(ids, skip_special_tokens=True)
                        # Fallback if decode fails or returns None/empty (e.g., special tokens only)
                        if not decoded_text:
                            self.logger.warning(f"Decoding with skip_special_tokens=True failed or returned empty. Trying without.")
                            decoded_text = self.dataset.decode(ids, skip_special_tokens=False)
                    else:
                        # Fallback if neither decode method exists
                        self.logger.error("Cannot decode: No usable tokenizer or dataset.decode method found.")
                        decoded_text = f"[Decoding Error: No decode method found]"

                    generated_text = decoded_text.strip() # Strip leading/trailing whitespace
                except Exception as e:
                    self.logger.error(f"Error decoding generated sequence: {e}", exc_info=True)
                    generated_text = f"Decoding error: {e}"

                generated_texts.append(generated_text)

            return generated_texts

        except Exception as e:
            self.logger.error(f"Error during text generation: {e}", exc_info=True)
            # Return error indicators instead of raising to allow partial results if num_return_sequences > 1
            return [f"[Generation Error: {e}]"] * num_return_sequences 
        finally:
            # Ensure model is back in training mode if it was before
            self.model.train() 