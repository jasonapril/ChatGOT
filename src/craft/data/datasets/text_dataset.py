import torch
import logging
import os
from typing import List, Union, Optional, Tuple
import inspect
import re
from pathlib import Path

from torch.utils.data import Dataset

# Import the new BaseDataset directly
from ..base import BaseDataset
# Import BaseTokenizer for type hinting
from ..tokenizers.base import Tokenizer
# Import hydra utils for path resolution
import hydra.utils

logger = logging.getLogger(__name__)

class TextDataset(BaseDataset):
    """Dataset that loads raw text files and tokenizes on the fly."""
    
    def __init__(self, file_paths: List[str], block_size: int, tokenizer: Tokenizer) -> None:
        """
        Initializes the Dataset from raw text files.

        Args:
            file_paths (List[str]): List of paths to text files.
            block_size (int): Maximum sequence length for blocks.
            tokenizer (Tokenizer): An initialized tokenizer instance.
        """
        super().__init__() # Initialize BaseDataset
        self.file_paths = file_paths
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.token_ids: torch.Tensor = torch.tensor([], dtype=torch.long)
        self.vocab_size: int = 0
        
        if not self.tokenizer:
             raise ValueError("A valid tokenizer must be provided to TextDataset.")

        # --- Load Tokenizer Model First (if applicable) --- 
        # Handle tokenizers that might need loading (like SubwordTokenizer)
        # This needs to happen BEFORE encoding
        if hasattr(self.tokenizer, 'load') and hasattr(self.tokenizer, 'config') and isinstance(self.tokenizer.config, dict) and 'model_path' in self.tokenizer.config:
            model_path_rel = self.tokenizer.config['model_path'] # Path relative to project root
            if model_path_rel:
                 model_path_abs = hydra.utils.to_absolute_path(model_path_rel)
                 try:
                     logger.info(f"Attempting to load tokenizer model from: {model_path_abs}")
                     self.tokenizer.load(model_path_abs) # Assuming load method takes path
                     logger.info(f"Tokenizer model loaded successfully.")
                 except Exception as e:
                     logger.error(f"Failed to load tokenizer model from {model_path_abs}: {e}", exc_info=True)
                     raise ValueError(f"Failed to load required tokenizer model: {e}") from e
            else:
                 # Allow tokenizers that don't need loading (e.g., char level)
                 logger.warning("Tokenizer has 'load' method and 'model_path' in config, but path is empty. Proceeding without explicit load.")
        elif hasattr(self.tokenizer, 'load'):
            # Log if load method exists but no path configured
            logger.info("Tokenizer has a 'load' method, but no model path was configured. Assuming pre-initialized or no loading needed.")
        # -------------------------------------------------- #
        
        # --- Read and Tokenize Text Data --- #
        all_text = ""
        logger.info(f"Loading and concatenating text data from: {self.file_paths}")
        for file_path in self.file_paths:
            try:
                 # Resolve path using Hydra utils if it might be relative
                 abs_file_path = hydra.utils.to_absolute_path(file_path)
                 if not os.path.exists(abs_file_path):
                      logger.error(f"Text file not found: {abs_file_path} (resolved from {file_path})")
                      continue # Skip missing files
                      
                 with open(abs_file_path, 'r', encoding='utf-8') as f:
                     all_text += f.read()
                 logger.debug(f"Read content from {abs_file_path}")
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}", exc_info=True)
                # Decide if this should be a fatal error
        
        if not all_text:
            logger.error("No text data could be loaded from provided file paths.")
            # Set token_ids to empty tensor to avoid downstream errors with None
            self.token_ids = torch.tensor([], dtype=torch.long)
            self.vocab_size = self.tokenizer.get_vocab_size() if self.tokenizer else 0
            return # Exit __init__ early

        logger.info(f"Loaded total {len(all_text)} characters. Tokenizing...")
        try:
            # Encode the entire concatenated text and convert directly to tensor
            encoded_list: List[int] = self.tokenizer.encode(all_text)
            self.token_ids = torch.tensor(encoded_list, dtype=torch.long)
            self.vocab_size = self.tokenizer.get_vocab_size()
            logger.info(f"Tokenization complete. Total tokens: {self.token_ids.numel()}, Vocab size: {self.vocab_size}")
            
            # Validation
            if self.token_ids.numel() <= self.block_size:
                logger.warning(
                    f"Total tokens ({self.token_ids.numel()}) is less than or equal to block_size ({self.block_size}). "
                    f"Dataset will be empty or have limited samples."
                )
        except Exception as e:
            logger.error(f"Failed during tokenization: {e}", exc_info=True)
            raise ValueError("Text encoding failed.") from e
        # ----------------------------------- #

    def __len__(self) -> int:
        if not hasattr(self, 'token_ids') or not isinstance(self.token_ids, torch.Tensor):
             return 0
        if self.token_ids.numel() <= self.block_size:
            return 0
        return self.token_ids.numel() - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
            
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size determined during initialization."""
        return self.vocab_size

    def decode(self, tokens: Union[torch.Tensor, List[int]], skip_special_tokens: bool = False) -> str:
        """Decodes a sequence of token IDs using the dataset's tokenizer."""
        if not self.tokenizer:
            logger.warning("Cannot decode: Tokenizer not available.")
            return "[Decode unavailable: No tokenizer]"
        
        # Convert tensor to list if necessary
        token_list: List[int]
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = tokens

        try:
            # Assuming base Tokenizer.decode expects List[int] and might not support skip_special_tokens
            # TODO: Check Tokenizer.decode signature and adjust if skip_special_tokens is valid
            return self.tokenizer.decode(token_list)
        except Exception as e:
            logger.error(f"Error during tokenizer decoding: {e}", exc_info=True)
            return f"[Decode error: {e}]" 