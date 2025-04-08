from typing import Dict, Any, List, Optional, Union, cast
import os
# Alias the tokenizer library's Tokenizer to avoid name clash with our base class
from tokenizers import Tokenizer as HFTokenizer # type: ignore[import-untyped]
from tokenizers import models, pre_tokenizers, trainers # type: ignore[import-untyped]
from .base import Tokenizer # Import our base class
import logging

logger = logging.getLogger(__name__)

class SubwordTokenizer(Tokenizer): # Inherit from our base Tokenizer
    """Tokenizer using Hugging Face tokenizers library (e.g., BPE)."""
    def __init__(self, 
                 vocab_size: int, # Required for training
                 # Accept special tokens/IDs matching the base class signature
                 pad_token: str = "<pad>", 
                 unk_token: str = "]", 
                 bos_token: str = "[", # Default specific to this type? OK.
                 eos_token: str = "]", # Default specific to this type? OK.
                 pad_id: int = -1, 
                 unk_id: int = 0, 
                 bos_id: int = 1, 
                 eos_id: int = 2,
                 # Allow passing other config params if needed
                 config: Optional[Dict[str, Any]] = None 
                 ):
        # Call base __init__ with the special tokens/IDs first
        super().__init__(pad_token, unk_token, bos_token, eos_token, pad_id, unk_id, bos_id, eos_id)
        
        logger.info(f"Initializing SubwordTokenizer with vocab_size={vocab_size}")
        # Store vocab_size passed during init (primarily needed for training)
        self.vocab_size = vocab_size 
        # Placeholder for the actual HF tokenizer instance
        self.tokenizer: Optional[HFTokenizer] = None 
        # Store any additional config if provided
        self.config = config or {}
        
        logger.info(f"SubwordTokenizer initialized.")
        
    def train(self, text_file: str, output_dir: str) -> None:
        """Train the tokenizer on input text file(s). NOTE: Base expects single file."""
        # Convert single file path to a list for consistency within this method
        files = [text_file]
        logger.info(f"Training SubwordTokenizer on files: {files}")
        if not files:
            raise ValueError("Input file list cannot be empty for training.")
        
        # Initialize the Hugging Face Tokenizer instance using BPE model
        # Pass unk_token from base class attribute
        self.tokenizer = HFTokenizer(models.BPE(unk_token=self.unk_token))
        
        # Configure pre-tokenization (ByteLevel is common for BPE)
        # `add_prefix_space=False` is often recommended for BPE
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Configure the trainer, using special tokens from base class attributes
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token
            ]
        )
        
        # Train the tokenizer
        self.tokenizer.train(files=files, trainer=trainer)
        
        # Update instance vocab size from trained tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()
        logger.info(f"Tokenizer training complete. Effective vocab size: {self.vocab_size}")

        # Save the tokenizer
        self.save(output_dir) # Use the save method
        
    def load(self, model_path: str) -> None:
        """Load a pre-trained tokenizer from a directory containing tokenizer.json."""
        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        logger.info(f"Loading SubwordTokenizer from {tokenizer_file}")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
        try:
            self.tokenizer = HFTokenizer.from_file(tokenizer_file)
            # Update instance vocab size from loaded tokenizer
            self.vocab_size = self.tokenizer.get_vocab_size()
            logger.info(f"SubwordTokenizer loaded successfully. Vocab size: {self.vocab_size}")
        except Exception as e:
            logger.exception(f"Failed to load tokenizer from {tokenizer_file}")
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_file}") from e
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        try:
            # Cast the return value to satisfy mypy
            return cast(List[int], self.tokenizer.encode(text).ids)
        except Exception as e:
            logger.exception(f"Error during encoding text: '{text[:50]}...'")
            raise RuntimeError("Encoding failed") from e
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to remove special tokens (like padding, bos, eos) 
                                 and clean up whitespace artifacts during decoding.
                                 Defaults to True (common use case).
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        try:
            # Cast the return value to satisfy mypy
            return cast(str, self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))
        except Exception as e:
            logger.exception(f"Error during decoding IDs: '{token_ids[:10]}...'")
            raise RuntimeError("Decoding failed") from e
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.tokenizer:
            # Cast the return value to satisfy mypy
            return cast(int, self.tokenizer.get_vocab_size())
        # Fallback to the size stored during init if tokenizer not loaded/trained yet
        logger.warning("Tokenizer not loaded, returning vocab_size from init.")
        return self.vocab_size 
    
    def save(self, output_dir: str) -> None:
        """Save the tokenizer configuration (tokenizer.json)."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Cannot save.")
        os.makedirs(output_dir, exist_ok=True)
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        try:
            self.tokenizer.save(tokenizer_path)
            logger.info(f"SubwordTokenizer saved to {tokenizer_path}")
            # Optionally save a separate config.json with our base class args?
            # For now, assumes HF tokenizer.json is sufficient.
        except Exception as e:
            logger.exception(f"Failed to save tokenizer to {tokenizer_path}")
            raise RuntimeError("Failed to save tokenizer") from e 