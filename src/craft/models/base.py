"""
Base model classes and abstractions for Craft.

This module provides abstract base classes for all model types in Craft,
enabling a consistent interface regardless of modality.
"""
from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.nn as nn
# Remove Pydantic imports if no longer needed directly
# from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationError
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra

# Import the config class from the new location
from craft.models.configs import BaseModelConfig # Adjust if other configs are needed directly

# Setup logger for this module
logger = logging.getLogger(__name__)

# Import from the new location
# Remove this import to break circular dependency. Checkpointing should be handled externally.
# from craft.training.checkpoint_utils import save_checkpoint, load_checkpoint


# Pydantic ModelConfig Base
# class BaseModelConfig(BaseModel):
#     """
#     Base Pydantic configuration class for all models.
#     Provides automatic validation and type hints.
#     Uses model_config for Pydantic V2 settings.
#     """
#     model_config = ConfigDict(extra='allow') # Keep allow for flexibility
#     model_type: str = Field("base", description="The type of the model (e.g., language, vision).")
#
#     # Common config fields can be added here if needed
#     # e.g., vocab_size: Optional[int] = None
#     # e.g., embedding_dim: Optional[int] = None
#
# class GenerativeModelConfig(BaseModelConfig):
#     """Config for Generative Models"""
#     model_type: str = Field("generative", description="Model type set to generative.")
#     max_seq_length: int = Field(1024, description="Maximum sequence length the model can handle")
#
# class LanguageModelConfig(GenerativeModelConfig):
#     """Config for Language Models"""
#     model_type: str = Field("language", description="Model type set to language.")
#     architecture: Optional[str] = Field(None, description="Name for the specific model architecture (e.g., transformer, rnn).")
#     vocab_size: int = Field(..., description="Size of the vocabulary (required).")
#     d_model: int = Field(768, description="Model dimension.")
#     n_head: int = Field(12, description="Number of attention heads.")
#     d_hid: Optional[int] = Field(None, description="Hidden dimension in feed-forward layers.")
#     n_layers: int = Field(12, description="Number of transformer layers.")
#     dropout: float = Field(0.1, description="Dropout probability.")
#     bias: bool = Field(True, description="Whether to use bias in linear layers.")
#     layer_norm_eps: float = Field(1e-5, description="Epsilon for layer normalization.")
#     activation: str = Field('gelu', description="Activation function.")
#     norm_first: bool = Field(True, description="Apply layer norm before attention/FFN.")
#
#     # Pydantic V2 validator for d_hid
#     @field_validator('d_hid', mode='before')
#     @classmethod
#     def set_d_hid_default(cls, v, info):
#         """Set default d_hid = d_model * 4 if not provided."""
#         # info.data should contain the raw input data being validated
#         if v is None and 'd_model' in info.data:
#             d_model = info.data.get('d_model')
#             # Use default d_model if not in data but defined in class
#             if d_model is None and 'd_model' in cls.model_fields:
#                  d_model = cls.model_fields['d_model'].default
#             
#             if isinstance(d_model, int):
#                 return d_model * 4
#         return v # Return original value if not calculated
#
# class VisionModelConfig(BaseModelConfig):
#     # ... (Additional fields specific to VisionModelConfig)
#     pass
#
# class MultiModalModelConfig(BaseModelConfig):
#     # ... (Additional fields specific to MultiModalModelConfig)
#     pass


# --- Base Model Class --- #
# *IMPORTANT*: Remove inheritance from Pydantic models. 
# Model IS an nn.Module, it HAS a config object.
class Model(nn.Module, ABC):
    """
    Abstract base class for all models.
    Inherits ONLY from nn.Module and ABC.
    Handles config assignment and basic save/load.
    """
    # Remove config related class attributes if any were added previously
    # No model_config = ConfigDict(...) here!

    def __init__(self, config: BaseModelConfig):
        """Initialize the base model with a validated Pydantic config object."""
        super().__init__() # Initialize nn.Module

        if not isinstance(config, BaseModelConfig):
             logging.error("Model received an invalid config object that is not a BaseModelConfig subclass.")
             # Raise error or use a default? Raising is safer.
             raise TypeError(f"Expected config to be a BaseModelConfig instance, got {type(config)}")
        
        self.config = config # Store the validated config object
        # Get model_type from the stored config object
        self.model_type = self.config.model_type 
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the model.
        
        Args:
            *args: Arguments for the model
            **kwargs: Keyword arguments for the model
            
        Returns:
            Model outputs
        """
        pass
    
    def _log_model_size(self):
        """Log the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"Model initialized with {n_params:,} parameters")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    def save(self, path: str, **kwargs):
        """
        Save the model's state_dict and configuration to a file using torch.save.
        
        **Purpose:** This method is intended for saving the final model artifact 
        (weights and configuration) for inference or sharing, NOT for saving the 
        full training state required for resuming training.
        
        **For resuming training, use the `CheckpointManager` class.**

        Args:
            path: Path to save the checkpoint file (.pt or .pth recommended).
            **kwargs: Additional metadata to save in the checkpoint dictionary.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.config.model_dump(), # Use Pydantic V2's model_dump
            **kwargs # Include any additional metadata provided
        }

        try:
            torch.save(checkpoint, path)
            logging.info(f"Model state and config saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save model to {path}: {e}", exc_info=True)

    def load(self, path: str, device: Optional[Union[torch.device, str]] = None,
             strict: bool = True) -> Dict[str, Any]:
        """
        Load the model's state_dict from a file saved by `Model.save`.
        
        **Purpose:** This method loads model weights and configuration for inference 
        or fine-tuning. It assumes the model architecture (instantiated via config) 
        matches the saved state.
        
        **It does NOT load optimizer, scheduler, or other training states.**
        **For resuming training, use the `CheckpointManager` class.**

        Args:
            path: Path to the checkpoint file.
            device: Device to load the model onto ('cpu', 'cuda', or torch.device object). Defaults to auto-detect.
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict().

        Returns:
            The loaded checkpoint dictionary (excluding model_state_dict if successfully loaded).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint file does not contain 'model_state_dict'.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        try:
            # Load the entire checkpoint dictionary first
            # Set weights_only=False because we save non-tensor config dict. Be cautious.
            logging.warning(f"Loading model checkpoint {path} with weights_only=False. Ensure the source is trusted.")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {path}: {e}", exc_info=True)
            raise

        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint file {path} does not contain 'model_state_dict'.")

        # Optionally verify config compatibility (basic check)
        if "model_config" in checkpoint:
            loaded_config_dict = checkpoint["model_config"]
            # Simple check for model_type mismatch
            if loaded_config_dict.get("model_type") != self.config.model_type:
                 logging.warning(
                    f"Loading checkpoint with model_type '{loaded_config_dict.get('model_type')}' "
                    f"into model with type '{self.config.model_type}'. Ensure compatibility."
                )
            # More thorough checks could be added here if needed
        else:
             logging.warning(f"Checkpoint {path} does not contain 'model_config'. Cannot verify compatibility.")

        # Load the state dict
        try:
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            if missing_keys:
                logging.warning(f"Missing keys when loading model state from {path}: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading model state from {path}: {unexpected_keys}")
        except Exception as e:
             logging.error(f"Error loading model state_dict from {path}: {e}", exc_info=True)
             raise

        # Move model to the target device AFTER loading state_dict
        self.to(device)
        logging.info(f"Model state loaded from {path} to {device}")

        # Return the rest of the checkpoint data (excluding the state dict we just loaded)
        del checkpoint["model_state_dict"]
        return checkpoint
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration using Pydantic's model_dump.
        
        Returns:
            Dictionary with the model configuration
        """
        return self.config.model_dump()


# GenerativeModel inherits from the updated Model
class GenerativeModel(Model):
    """
    Base class for generative models.
    Checks config type and provides generate method.
    """
    def __init__(self, config: BaseModelConfig): # Accept BaseModelConfig initially
        super().__init__(config)
        
        # It's safe to assume self.config is GenerativeModelConfig or subclass here due to factory
        self.max_seq_length = getattr(self.config, 'max_seq_length', 1024)

    # Remove @abstractmethod decorator and provide implementation
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None, # Changed default to None
        top_p: Optional[float] = None, # Changed default to None
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None, # Added EOS token handling
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Generate sequences auto-regressively from the model.

        Handles common sampling techniques like temperature, top-k, top-p, 
        and repetition penalty.
        
        Args:
            input_ids: Input token ids tensor of shape [batch_size, seq_len].
            max_new_tokens: Maximum number of new tokens to generate per sequence.
            temperature: Sampling temperature. Higher values (e.g., > 1) make 
                         output more random, lower values (e.g., < 1) make it more 
                         deterministic. Set to 0 for greedy decoding.
            top_k: If set, only the top_k highest probability tokens are considered 
                   for sampling at each step.
            top_p: If set, only the smallest set of tokens whose cumulative probability 
                   exceeds top_p are considered (nucleus sampling).
            repetition_penalty: Penalty applied to logits of previously generated 
                                tokens (excluding padding). 1.0 means no penalty.
            eos_token_id: If specified, generation stops when this token is generated.
            verbose: Whether to log generation progress.
            
        Returns:
            Generated token ids tensor of shape [batch_size, seq_len + generated_len].
        """
        self.eval() # Set model to evaluation mode
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # --- Start: Input Truncation --- #
        max_input_len = self.max_seq_length - max_new_tokens
        if max_input_len < 0:
            # This case is problematic - cannot generate anything if max_new_tokens > max_seq_length
            # Raise an error or issue a warning?
            logger.warning(
                f"max_new_tokens ({max_new_tokens}) is greater than max_seq_length "
                f"({self.max_seq_length}). Cannot generate any tokens. "
                f"Returning original input."
            )
            # Or maybe allow generating up to max_seq_length if prompt is empty?
            # For now, let's just return input or maybe truncated input to max_seq_length.
            # Returning input seems safest.
            return input_ids 
            # Alternative: max_input_len = 0 # Allow empty prompt only

        if input_ids.size(1) > max_input_len:
            logger.warning(
                f"Input prompt length ({input_ids.size(1)}) exceeds max allowed input length "
                f"({max_input_len} = max_seq_length {self.max_seq_length} - max_new_tokens {max_new_tokens}). "
                f"Truncating prompt to last {max_input_len} tokens."
            )
            input_ids = input_ids[:, -max_input_len:]
        # --- End: Input Truncation --- #

        generated_tokens = input_ids.clone()
        stop_generation = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for i in range(max_new_tokens):
                # 1. Prepare inputs: Context window management within the loop
                # The input to the model should be the current sequence, 
                # potentially truncated if it exceeds max_seq_length.
                current_seq_len = generated_tokens.size(1)
                if current_seq_len > self.max_seq_length:
                    # If sequence grows beyond max length, take the last `max_seq_length` tokens
                    # Note: This means the effective context might shrink over time
                    # if max_new_tokens is large. A sliding window approach might be better.
                    model_input = generated_tokens[:, -self.max_seq_length:]
                else:
                    model_input = generated_tokens

                # 2. Forward pass to get logits for the next token
                # Assumes the model's forward pass returns logits of shape 
                # [batch_size, sequence_length, vocab_size]
                logits = self.forward(model_input) 
                next_token_logits = logits[:, -1, :] # Get logits for the last position

                # 3. Apply repetition penalty (skip padding if applicable)
                if repetition_penalty != 1.0:
                    # Iterate through batch items that haven't stopped
                    for b in range(batch_size):
                        if not stop_generation[b]:
                            # Penalize tokens present in the current generated sequence for this item
                            # Avoid penalizing special tokens like padding if necessary (requires knowing padding_id)
                            for token_id in generated_tokens[b]:
                                # Check bounds before penalizing
                                if 0 <= token_id < next_token_logits.size(-1):
                                     if next_token_logits[b, token_id] > 0: # Penalize positive logits more
                                         next_token_logits[b, token_id] /= repetition_penalty
                                     else: # Penalize negative logits less (bring towards zero)
                                         next_token_logits[b, token_id] *= repetition_penalty 

                # 4. Apply temperature scaling
                if temperature == 0: # Greedy decoding
                     next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    # 5. Apply Top-K filtering
                    if top_k is not None and top_k > 0:
                        # Remove tokens with probability less than the top_k token
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

                    # 6. Apply Top-p (nucleus) filtering
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Always keep at least one token
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        
                        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                        next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

                    # 7. Sample the next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # 8. Append generated token and check stopping criteria
                # Only append and check EOS for sequences that haven't stopped yet
                not_stopped_mask = ~stop_generation.unsqueeze(-1)
                
                # Use a placeholder (like padding) for stopped sequences to keep tensor shape consistent
                # Assuming padding_id is 0, otherwise use a known safe value. Let's use 0 for now.
                placeholder_token = torch.zeros_like(next_token) 
                token_to_append = torch.where(not_stopped_mask, next_token, placeholder_token)
                generated_tokens = torch.cat([generated_tokens, token_to_append], dim=1)

                # Update stop generation flags
                if eos_token_id is not None:
                    just_stopped = (next_token.squeeze(-1) == eos_token_id) & (~stop_generation)
                    stop_generation |= just_stopped

                # If all sequences have stopped, break early
                if stop_generation.all():
                    break
                
                if verbose and (i + 1) % 10 == 0:
                    logging.info(f"Generated {i + 1}/{max_new_tokens} tokens...")
        
        # Return only the generated part for sequences that stopped, or full if not stopped?
        # Current implementation returns the full sequence including prompt + generated.
        return generated_tokens


# LanguageModel inherits from the updated GenerativeModel
class LanguageModel(GenerativeModel):
    """
    Base class for language models.
    Checks config type.
    """
    def __init__(self, config: BaseModelConfig): # Expects BaseModelConfig specifically
        super().__init__(config)
    
    def calculate_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate perplexity from logits and targets.
        
        Args:
            logits: Model output logits
            targets: Target token indices
            
        Returns:
            Perplexity
        """
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1)
        )
        return torch.exp(loss)


class VisionModel(Model):
    """
    Abstract base class for vision models in Craft.
    """
    
    def __init__(self, config: BaseModelConfig):
        if not isinstance(config, BaseModelConfig):
            raise ValueError("VisionModel requires a BaseModelConfig instance.")
        super().__init__(config)


class MultiModalModel(Model):
    """
    Abstract base class for multi-modal models in Craft.
    """
    
    def __init__(self, config: BaseModelConfig):
        if not isinstance(config, BaseModelConfig):
            raise ValueError("MultiModalModel requires a BaseModelConfig instance.")
        super().__init__(config)


# --- Remove Factory Function --- 
# The create_model_from_config function has been moved to src/models/factory.py
# def create_model_from_config(config_dict: Dict[str, Any]) -> Model:
#    ... 