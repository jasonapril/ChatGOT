from typing import Dict, Any, Union
from .base import Tokenizer
from .char import CharTokenizer
from .subword import SubwordTokenizer
from .sentencepiece import SentencePieceTokenizer
import logging

logger = logging.getLogger(__name__) # Ensure logger is initialized

# Define public interface
__all__ = [
    "Tokenizer",
    "CharTokenizer",
    "SubwordTokenizer",
    "SentencePieceTokenizer",
    "create_tokenizer"
]

def create_tokenizer(config: Union[Dict[str, Any], str]) -> Tokenizer:
    """Creates a tokenizer instance based on a config dict or path string."""

    if isinstance(config, str):
        # Assume it's a path for SentencePiece model directory
        logger.info(f"Attempting to load SentencePieceTokenizer from path: {config}")
        try:
            # Use the renamed classmethod
            return SentencePieceTokenizer.load_from_prefix(config)
        except FileNotFoundError as e:
             logger.error(f"Failed to load SentencePiece tokenizer from path {config}: File not found - {e}")
             raise ValueError(f"Cannot load SentencePiece tokenizer from path: {config}") from e
        except Exception as e:
             logger.error(f"Unexpected error loading SentencePiece tokenizer from path {config}: {e}")
             raise ValueError(f"Cannot load SentencePiece tokenizer from path: {config}") from e

    elif isinstance(config, dict):
        config_copy = config.copy() # Work with a copy
        tokenizer_type = config_copy.pop('type', None)
        if not tokenizer_type:
            raise ValueError("Tokenizer config dict must include a 'type' key.")

        logger.info(f"Creating tokenizer of type '{tokenizer_type}' from dict config: {config_copy}")

        if tokenizer_type == 'char':
            # CharTokenizer(**kwargs: Any)
            return CharTokenizer(**config_copy)

        elif tokenizer_type == 'subword':
            # SubwordTokenizer(vocab_size: int, **kwargs: Any)
            vocab_size = config_copy.get("vocab_size")
            if vocab_size is None:
                raise ValueError("SubwordTokenizer config requires 'vocab_size'.")
            # Pass vocab_size positionally, rest as kwargs
            return SubwordTokenizer(vocab_size=int(vocab_size), **config_copy)

        elif tokenizer_type == "sentencepiece":
            # SentencePieceTokenizer(model_path: str, **kwargs: Any)
            model_path = config_copy.get("model_path")
            if not model_path or not isinstance(model_path, str):
                raise ValueError("SentencePieceTokenizer config dict requires 'model_path' (string).")
            # Pass model_path positionally, rest as kwargs
            return SentencePieceTokenizer(model_path=model_path, **config_copy)
        else:
            raise ValueError(f"Unknown tokenizer type in config dict: {tokenizer_type}")
    else:
        raise TypeError(f"Invalid config type for create_tokenizer: {type(config)}") 