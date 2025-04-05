"""
Dataset implementations for character-level models.
"""
import torch
import logging
import os
import json # Import json
import pickle
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple

from torch.utils.data import Dataset

# Import the new BaseDataset directly
from .base import BaseDataset
# Import BaseTokenizer for type hinting
from .tokenizers.base import BaseTokenizer
# Import hydra utils for path resolution
import hydra.utils

logger = logging.getLogger(__name__)


# Removed load_data function. Use factory functions like create_dataset_from_config. 


class PickledDataset(BaseDataset):
    """PyTorch Dataset for loading data pre-tokenized and saved in a .pkl file."""

    def __init__(self, file_path: str, block_size: int, vocab_path: str = None):
        """
        Initializes the Dataset from a .pkl file containing a list or numpy array of token IDs.
        Metadata (like vocab size, mappings) must be provided via the external `vocab_path` JSON file.

        Args:
            file_path (str): Path to the .pkl file (containing list/array of token IDs).
            block_size (int): Maximum sequence length for blocks.
            vocab_path (str, optional): Path to a JSON file containing vocabulary metadata.
        """
        super().__init__() # Initialize BaseDataset
        self.file_path = file_path
        self.block_size = block_size
        self._vocab_path = vocab_path # Store path to potential external vocab
        self._metadata = None # Cache for metadata

        logger.info(f"Loading pre-tokenized token IDs from: {self.file_path}")
        if not os.path.exists(self.file_path):
            logger.error(f"Pickled data file not found: {self.file_path}")
            raise FileNotFoundError(f"Pickled data file not found: {self.file_path}")

        try:
            with open(self.file_path, 'rb') as f:
                loaded_data = pickle.load(f)

            # Expect a list or numpy array directly
            if isinstance(loaded_data, np.ndarray):
                self.token_ids = torch.from_numpy(loaded_data.astype(np.int64))
            elif isinstance(loaded_data, list):
                self.token_ids = torch.tensor(loaded_data, dtype=torch.long)
            else:
                raise TypeError(f"Pickled file content must be a list or numpy array of token IDs, found {type(loaded_data)}")

            # Basic validation
            if len(self.token_ids) == 0:
                 logger.warning(f"Dataset loaded from {self.file_path} is empty (0 tokens).")
            elif len(self.token_ids) < self.block_size + 1:
                logger.warning(
                    f"Dataset length ({len(self.token_ids)}) is less than block_size+1 ({self.block_size + 1}). "
                    f"This might lead to issues during training if drop_last=False."
                )
            logger.info(f"Successfully loaded {len(self.token_ids)} tokens from {self.file_path}.")

        except FileNotFoundError:
            raise # Re-raise specific error
        except (pickle.UnpicklingError, TypeError, Exception) as e:
            logger.error(f"Failed to load or parse pickle file {self.file_path}: {e}", exc_info=True)
            raise IOError(f"Failed to load pickle file {self.file_path}: {e}") from e

    def __len__(self) -> int:
        """Returns the number of possible starting positions for sequences."""
        # Adjust length to account for block_size + 1 tokens needed for each item (x and y)
        # If length is less than block_size + 1, len is 0.
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a block of data and the target block offset by one.

        Args:
            idx (int): Starting index for the block.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence (x) and target sequence (y).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        
        # Get sequence block (x) and target block (y)
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y

    def get_metadata(self) -> Dict[str, Any]:
        """
        Loads and returns vocabulary metadata from the specified vocab_path.
        Caches the metadata after the first load.

        Returns:
            Dict[str, Any]: Dictionary containing vocabulary metadata (e.g., vocab_size). 
                          Returns an empty dict if no vocab_path was provided or loading fails.
        """
        if self._metadata is not None:
            return self._metadata

        if not self._vocab_path:
            logger.warning("No external vocab_path provided. Cannot load metadata.")
            self._metadata = {}
            return self._metadata

        logger.info(f"Loading vocabulary metadata from: {self._vocab_path}")
        if not os.path.exists(self._vocab_path):
            logger.error(f"Vocabulary metadata file not found: {self._vocab_path}")
            self._metadata = {}
            return self._metadata

        try:
            with open(self._vocab_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                 logger.error(f"Vocabulary file {self._vocab_path} does not contain a valid JSON dictionary.")
                 metadata = {}
            self._metadata = metadata
            logger.info(f"Successfully loaded vocabulary metadata. Keys: {list(self._metadata.keys())}")
            return self._metadata
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from vocabulary file {self._vocab_path}: {e}")
            self._metadata = {}
            return self._metadata
        except Exception as e:
            logger.error(f"Failed to load vocabulary file {self._vocab_path}: {e}", exc_info=True)
            self._metadata = {}
            return self._metadata

    @property
    def vocab_size(self) -> Optional[int]:
        """Attempts to get vocab_size from loaded metadata."""
        metadata = self.get_metadata()
        vs = metadata.get('vocab_size')
        if vs is None:
             logger.warning("vocab_size not found in metadata.")
        return vs

    def decode(self, ids: Union[torch.Tensor, List[int]]) -> str:
        """
        Decodes a sequence of token IDs back into text using idx_to_char from metadata.
        Returns a placeholder if metadata or idx_to_char is unavailable.
        """
        metadata = self.get_metadata()
        idx_to_char = metadata.get('idx_to_char')

        if not idx_to_char:
            logger.warning("Cannot decode: idx_to_char mapping not found in metadata.")
            return "[Decode unavailable: Missing metadata]"
        
        # Convert tensor to list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        try:
            # Handle potential string keys from JSON
            if all(isinstance(k, str) for k in idx_to_char.keys()):
                 idx_map = {int(k): v for k, v in idx_to_char.items()} 
            else:
                 idx_map = idx_to_char # Assume keys are already integers
            return ''.join([idx_map.get(i, '?') for i in ids]) # Use '?' for unknown IDs
        except Exception as e:
            logger.error(f"Error during decoding: {e}", exc_info=True)
            return f"[Decode error: {e}]"


# --- TextDataset --- #
class TextDataset(BaseDataset):
    """Dataset that loads raw text files and tokenizes on the fly."""
    
    def __init__(self, file_paths: List[str], block_size: int, tokenizer: BaseTokenizer):
        """
        Initializes the Dataset from raw text files.

        Args:
            file_paths (List[str]): List of paths to text files.
            block_size (int): Maximum sequence length for blocks.
            tokenizer (BaseTokenizer): An initialized tokenizer instance.
        """
        super().__init__() # Initialize BaseDataset
        self.file_paths = file_paths
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.token_ids = [] # Initialize token_ids
        self.vocab_size = 0 # Initialize vocab_size
        
        if not self.tokenizer:
             raise ValueError("A valid tokenizer must be provided to TextDataset.")

        # --- Load Tokenizer Model First (if applicable) --- 
        # Handle tokenizers that might need loading (like SubwordTokenizer)
        # This needs to happen BEFORE encoding
        if hasattr(self.tokenizer, 'load') and hasattr(self.tokenizer, 'config') and isinstance(self.tokenizer.config, dict) and 'model_path' in self.tokenizer.config:
            model_path_rel = self.tokenizer.config['model_path'] # Path relative to project root
            if model_path_rel:
                 model_path_abs = hydra.utils.to_absolute_path(model_path_rel)
                 logger.info(f"Attempting to load tokenizer model from {model_path_abs} (resolved from {model_path_rel})...")
                 try:
                     self.tokenizer.load(model_path_abs) # Load using absolute path
                     logger.info(f"Tokenizer model loaded successfully from {model_path_abs}")
                 except FileNotFoundError:
                     logger.error(f"Tokenizer model file not found at {model_path_abs}. Check the path in the config.")
                     raise RuntimeError(f"Failed to load required tokenizer model from {model_path_abs}") from FileNotFoundError
                 except Exception as load_err:
                     logger.error(f"Failed to load tokenizer model from {model_path_abs}: {load_err}", exc_info=True)
                     raise RuntimeError(f"Failed to load required tokenizer model from {model_path_abs}") from load_err
            else:
                 logger.warning("Tokenizer has a load method but model_path is empty or not set in its config.")
        # -----------------------------------------------

        # Load and concatenate text data
        full_text = ""
        logger.info(f"Loading and concatenating text from {len(file_paths)} files...")
        for file_path_rel in file_paths:
            file_path_abs = hydra.utils.to_absolute_path(file_path_rel)
            if not os.path.exists(file_path_abs):
                logger.warning(f"File not found: {file_path_abs} (resolved from {file_path_rel}). Skipping.")
                continue
            try:
                with open(file_path_abs, 'r', encoding='utf-8') as f:
                    full_text += f.read()
                logger.debug(f"Read {len(full_text)} characters so far from {file_path_abs}")
            except Exception as e:
                logger.error(f"Failed to read file {file_path_abs}: {e}", exc_info=True)
                # Decide whether to raise or just continue
                raise
        
        logger.info(f"Total characters loaded: {len(full_text)}")
        
        if not full_text:
            logger.warning("No text loaded from provided file paths. Dataset will be empty.")
            return # Exit init early if no text
        
        # Tokenize the entire text
        logger.info("Tokenizing the loaded text...")
        try:
            # Check if tokenizer is loaded (relevant for tokenizers needing explicit load)
            # This basic check assumes tokenizer.encode works if initialized.
            # More robust check might be needed depending on tokenizer implementation.
            if not hasattr(self.tokenizer, 'encode') or not callable(self.tokenizer.encode):
                 raise ValueError("Provided tokenizer does not have a callable 'encode' method.")
                 
            self.token_ids = self.tokenizer.encode(full_text)
            self.vocab_size = self.tokenizer.get_vocab_size()
            logger.info(f"Text tokenized into {len(self.token_ids)} tokens with vocab size {self.vocab_size}.")
        except ValueError as ve:
            logger.error(f"Tokenizer error during encoding: {ve}", exc_info=True)
            raise
        except RuntimeError as rte:
             logger.error(f"Runtime error during tokenization (possibly loading failed): {rte}", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"Failed to encode text with tokenizer {type(self.tokenizer).__name__}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Return the number of blocks in the dataset."""
        # Subtract block_size because the last block needs a target
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        """Get a block of token IDs (input and target)."""
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} out of range for dataset length {len(self)}.")
            
        # Extract input sequence (x) and target sequence (y)
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        
        # Convert lists to tensors
        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size used by the tokenizer."""
        return self.vocab_size

    def decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        """Decode a sequence of token IDs back to text using the tokenizer."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens) 