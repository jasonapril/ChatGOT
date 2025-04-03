"""
Text Generation Wrapper Class
===========================

This module provides the TextGenerator class, which wraps a model 
(typically one with a built-in .generate() method, like from Hugging Face)
and a dataset/tokenizer to provide a convenient interface for generation.

Note: Standalone sampling, beam search, and batch generation functions
have been moved to sampling.py, beam_search.py, and batch_generation.py respectively.
"""

import torch
import logging
import time  # Import the time module
from typing import Dict, List, Any, Optional

class TextGenerator:
    """Handles text generation by wrapping a model with a .generate() method."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Dict[str, Any], # General config if needed
        dataset: Any,  # Dataset or object with decode method and char_to_idx/tokenizer
        # vocab_path: Optional[str] = None # Likely not needed if dataset handles vocab
    ):
        self.model = model
        self.device = device
        self.config = config
        self.dataset = dataset
        # self.vocab_path = vocab_path
        self.logger = logging.getLogger(self.__class__.__name__)

        if not hasattr(model, 'generate'):
             self.logger.warning(
                 f"Model {type(model).__name__} does not have a standard `.generate()` method. "
                 "TextGenerator class may not work as expected. Consider using standalone generation functions."
             )

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
            # --- Encoding --- #
            input_ids = None
            if hasattr(self.dataset, 'tokenizer') and hasattr(self.dataset.tokenizer, 'encode'):
                # Prefer tokenizer if available
                encoded_prompt = self.dataset.tokenizer.encode(start_prompt, return_tensors="pt")
                input_ids = encoded_prompt.to(self.device)
                self.logger.debug(f"Encoded prompt using tokenizer: {input_ids.shape}")
            elif hasattr(self.dataset, 'char_to_idx') and self.dataset.char_to_idx:
                # Fallback to char-level if tokenizer unavailable
                input_ids_list = [self.dataset.char_to_idx.get(char, 0) for char in start_prompt] # Assume 0 for <unk>
                input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
                self.logger.debug(f"Encoded prompt using char_to_idx: {input_ids.shape}")
            else:
                raise ValueError("Dataset must provide either a tokenizer with an encode method or a char_to_idx mapping.")

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
            
            # Add pad/eos only if they are valid IDs
            if eff_pad_token_id is not None:
                 generation_kwargs["pad_token_id"] = eff_pad_token_id
            if eff_eos_token_id is not None:
                 generation_kwargs["eos_token_id"] = eff_eos_token_id
                 
            self.logger.info(f"Calling model.generate() with config: {generation_kwargs}")

            # --- Generate sequences --- #
            start_time = time.time()
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=input_ids,
                    **generation_kwargs
                )
            gen_time = time.time() - start_time
            self.logger.info(f"Model.generate() completed in {gen_time:.2f}s")

            # --- Decoding --- #
            generated_texts = []
            if not hasattr(self.dataset, 'decode'):
                 self.logger.error("Dataset does not have a required 'decode' method.")
                 return ["Error: Decoding not possible."] * num_return_sequences
                 
            for sequence in output_sequences:
                # Remove the prompt part to get only the generated tokens
                # Ensure sequence is on CPU before slicing/converting
                input_len = input_ids.shape[-1]
                generated_ids = sequence.cpu()[input_len:].tolist()
                
                # Decode, handling potential errors
                generated_text = "[Decoding Error]" # Default value
                try:
                    # Attempt 1: Decode with skip_special_tokens=True
                    text_attempt_1 = self.dataset.decode(generated_ids, skip_special_tokens=True)
                    generated_text = text_attempt_1 # Assign only if successful
                except TypeError:
                    self.logger.warning(
                        "Decoding failed with skip_special_tokens=True, trying again without it.",
                        exc_info=True
                    )
                    try:
                        # Attempt 2: Decode without skip_special_tokens
                        text_attempt_2 = self.dataset.decode(generated_ids, skip_special_tokens=False)
                        generated_text = text_attempt_2 # Assign only if successful
                    except Exception as decode_err_fallback:
                        self.logger.error(f"Error during fallback decoding attempt: {decode_err_fallback}", exc_info=True)
                        # Fallback failed, generated_text remains "[Decoding Error]"
                except Exception as decode_err_primary:
                    self.logger.error(f"Error decoding generated sequence: {decode_err_primary}", exc_info=True)
                    # Primary failed (not TypeError), generated_text remains "[Decoding Error]"

                generated_texts.append(generated_text.strip()) # Strip leading/trailing whitespace

            return generated_texts

        except Exception as e:
            self.logger.error(f"Error during text generation: {e}", exc_info=True)
            # Return error indicators instead of raising to allow partial results if num_return_sequences > 1
            return [f"[Generation Error: {e}]"] * num_return_sequences 
        finally:
            # Ensure model is back in training mode if it was before
            self.model.train() 