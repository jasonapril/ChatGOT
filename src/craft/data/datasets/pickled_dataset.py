import torch
import logging
import os
import json # Import json
import pickle
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path

from torch.utils.data import Dataset

# Import the new BaseDataset directly
from ..base import BaseDataset

logger = logging.getLogger(__name__)

class PickledDataset(BaseDataset):
    """PyTorch Dataset for loading data pre-tokenized and saved in a .pkl file."""

    def __init__(self, file_path: str, block_size: int, vocab_path: Optional[str] = None, **kwargs):
        """
        Initializes the Dataset from a .pkl file containing a list or numpy array of token IDs.
        Metadata (like vocab size, mappings) must be provided via the external `vocab_path` JSON file.

        Args:
            file_path (str): Path to the .pkl file (containing list/array of token IDs).
            block_size (int): Maximum sequence length for blocks.
            vocab_path (Optional[str]): Path to the external vocabulary JSON file.
            **kwargs: Additional keyword arguments.
        """
        super().__init__() # Initialize BaseDataset
        self.file_path = Path(file_path)
        self.block_size = block_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing PickledDataset with file: {self.file_path}")
        # Ignore any extra kwargs passed (e.g., by Hydra instantiation)
        if kwargs:
            self.logger.debug(f"Ignoring unexpected arguments: {kwargs.keys()}")
        
        self._vocab_path = vocab_path # Store path to potential external vocab
        self._metadata = None # Cache for metadata

        if not self.file_path.exists():
            raise FileNotFoundError(f"Pickled dataset file not found: {self.file_path}")

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
                 self.logger.warning(f"Dataset loaded from {self.file_path} is empty (0 tokens).")
            elif len(self.token_ids) < self.block_size + 1:
                self.logger.warning(
                    f"Dataset length ({len(self.token_ids)}) is less than block_size+1 ({self.block_size + 1}). "
                    f"This might lead to issues during training if drop_last=False."
                )
            self.logger.info(f"Successfully loaded {len(self.token_ids)} tokens from {self.file_path}.")

        except FileNotFoundError:
            raise # Re-raise specific error
        except (pickle.UnpicklingError, TypeError, Exception) as e:
            self.logger.error(f"Failed to load or parse pickle file {self.file_path}: {e}", exc_info=True)
            raise IOError(f"Failed to load pickle file {self.file_path}: {e}") from e

    def __len__(self) -> int:
        """Returns the number of sequences of length block_size that can be obtained."""
        # Check if token_ids is initialized and is a tensor
        if not hasattr(self, 'token_ids') or not isinstance(self.token_ids, torch.Tensor):
             # Should not happen if __init__ ran correctly, but good safeguard
             return 0 
             
        # Correct check for empty or too short tensor
        if self.token_ids.numel() == 0 or self.token_ids.numel() <= self.block_size:
             return 0
             
        # Correct calculation: number of full blocks of size block_size.
        # Example: 10 tokens, block_size 3 -> (10-1)//3 = 9//3 = 3 blocks ([0:3], [3:6], [6:9])
        # The number of items is the total number of tokens minus 1 (for the target offset)
        # divided by the block size (sequence length). Integer division gives full blocks.
        return (self.token_ids.numel() - 1) // self.block_size

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
            self.logger.warning("No external vocab_path provided. Cannot load metadata.")
            self._metadata = {}
            return self._metadata

        self.logger.info(f"Loading vocabulary metadata from: {self._vocab_path}")
        if not os.path.exists(self._vocab_path):
            self.logger.error(f"Vocabulary metadata file not found: {self._vocab_path}")
            self._metadata = {}
            return self._metadata

        try:
            with open(self._vocab_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                 self.logger.error(f"Vocabulary file {self._vocab_path} does not contain a valid JSON dictionary.")
                 metadata = {}
            self._metadata = metadata
            self.logger.info(f"Successfully loaded vocabulary metadata. Keys: {list(self._metadata.keys())}")
            return self._metadata
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from vocabulary file {self._vocab_path}: {e}")
            self._metadata = {}
            return self._metadata
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary file {self._vocab_path}: {e}", exc_info=True)
            self._metadata = {}
            return self._metadata

    @property
    def vocab_size(self) -> Optional[int]:
        """Attempts to get vocab_size from loaded metadata."""
        metadata = self.get_metadata()
        vs = metadata.get('vocab_size')
        if vs is None:
             self.logger.warning("vocab_size not found in metadata.")
        return vs

    def decode(self, ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of token IDs back into text using idx_to_char from metadata.
        Returns a placeholder if metadata or idx_to_char is unavailable.
        
        Args:
            ids: The token IDs to decode (Tensor or list).
            skip_special_tokens: Whether to skip PAD and EOS tokens during decoding.

        Returns:
            The decoded string.
        """
        metadata = self.get_metadata()
        idx_to_char = metadata.get('idx_to_char')
        pad_token_id = metadata.get('pad_token_id') # Get pad_token_id from metadata
        eos_token_id = metadata.get('eos_token_id') # Get eos_token_id from metadata

        if not idx_to_char:
            self.logger.warning("Cannot decode: idx_to_char mapping not found in metadata.")
            return "[Decode unavailable: Missing metadata]"
        
        # Convert tensor to list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        try:
            # Handle potential string keys from JSON
            if not hasattr(self, '_idx_map_cache'): # Cache the integer map
                if all(isinstance(k, str) for k in idx_to_char.keys()):
                     self._idx_map_cache = {int(k): v for k, v in idx_to_char.items()} 
                else:
                     self._idx_map_cache = idx_to_char # Assume keys are already integers
            idx_map = self._idx_map_cache

            decoded_chars = []
            for i in ids:
                if skip_special_tokens:
                    if pad_token_id is not None and i == pad_token_id:
                        continue
                    if eos_token_id is not None and i == eos_token_id:
                        continue
                decoded_chars.append(idx_map.get(i, '?')) # Use '?' for unknown IDs

            return ''.join(decoded_chars)
        except Exception as e:
            self.logger.error(f"Error during decoding: {e}", exc_info=True)
            return f"[Decode error: {e}]" 