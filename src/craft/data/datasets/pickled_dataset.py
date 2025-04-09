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
# Import SentencePieceTokenizer for dynamic loading during decode
from ..tokenizers.sentencepiece import SentencePieceTokenizer

logger = logging.getLogger(__name__)

class PickledDataset(BaseDataset):
    """PyTorch Dataset for loading data pre-tokenized and saved in a .pkl file."""

    # Cache for loaded tokenizer instances (keyed by model path)
    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(self, file_path: str, block_size: int, **kwargs: Any):
        """
        Initializes the Dataset from a .pkl file containing a list or numpy array of token IDs.
        Metadata (vocab size, tokenizer info) is expected in a 'metadata.json'
        file located in the same directory as the .pkl file.

        Args:
            file_path (str): Path to the .pkl file (e.g., 'data/processed/my_data/train.pkl').
            block_size (int): Maximum sequence length for blocks.
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__() # Initialize BaseDataset
        self.file_path = Path(file_path)
        self.block_size = block_size
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing PickledDataset with file: {self.file_path}")

        # Metadata file path is inferred from the data file path
        self.metadata_path = self.file_path.parent / "metadata.json"

        # Ignore any extra kwargs passed
        if kwargs:
            self.logger.debug(f"Ignoring unexpected arguments: {list(kwargs.keys())}")

        self._metadata: Optional[Dict[str, Any]] = None # Cache for metadata
        self._idx_map_cache: Optional[Dict[int, str]] = None # Cache for char idx map

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
            elif isinstance(loaded_data, torch.Tensor):
                if loaded_data.dtype != torch.long:
                     self.logger.warning(f"Loaded tensor has dtype {loaded_data.dtype}, converting to long.")
                     self.token_ids = loaded_data.long()
                else:
                     self.token_ids = loaded_data
            else:
                raise TypeError(f"Pickled file content must be a list, numpy array, or torch.Tensor of token IDs, found {type(loaded_data)}")

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
        Loads and returns metadata from 'metadata.json' located in the same
        directory as the data file.
        Caches the metadata after the first load.

        Returns:
            Dict[str, Any]: Dictionary containing metadata.
                          Returns an empty dict if loading fails.
        """
        if self._metadata is not None:
            return self._metadata

        self.logger.info(f"Loading metadata from: {self.metadata_path}")
        if not self.metadata_path.exists():
            self.logger.error(f"Metadata file not found: {self.metadata_path}")
            self._metadata = {}
            return self._metadata

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                 self.logger.error(f"Metadata file {self.metadata_path} does not contain a valid JSON dictionary.")
                 metadata = {}
            self._metadata = metadata
            self.logger.info(f"Successfully loaded metadata. Keys: {list(self._metadata.keys())}")
            return self._metadata
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from metadata file {self.metadata_path}: {e}")
            self._metadata = {}
            return self._metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata file {self.metadata_path}: {e}", exc_info=True)
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

    def _get_tokenizer(self) -> Optional[Any]:
        """Helper to load and cache tokenizer based on metadata."""
        metadata = self.get_metadata()
        tokenizer_type = metadata.get('tokenizer_type')
        tokenizer_model_path = metadata.get('tokenizer_model_path')

        if tokenizer_type == 'SentencePiece':
            if not tokenizer_model_path:
                self.logger.error("Cannot load SentencePiece tokenizer: 'tokenizer_model_path' missing from metadata.")
                return None
            # Use cache
            if tokenizer_model_path not in PickledDataset._tokenizer_cache:
                try:
                    self.logger.info(f"Loading SentencePiece tokenizer from {tokenizer_model_path}...")
                    tokenizer_instance = SentencePieceTokenizer.load_from_prefix(tokenizer_model_path)
                    PickledDataset._tokenizer_cache[tokenizer_model_path] = tokenizer_instance
                except Exception as e:
                    self.logger.error(f"Failed to load SentencePiece tokenizer from {tokenizer_model_path}: {e}", exc_info=True)
                    # Cache failure marker to avoid retrying repeatedly
                    PickledDataset._tokenizer_cache[tokenizer_model_path] = None
                    return None
            return PickledDataset._tokenizer_cache[tokenizer_model_path]
        elif tokenizer_type == 'CharTokenizer': # Or based on metadata key presence
            # CharTokenizer is simpler, decode uses idx_to_char directly
            return None # No separate tokenizer object needed for decode
        else:
            # self.logger.warning(f"Unsupported tokenizer type '{tokenizer_type}' for dynamic loading in decode.")
            return None

    def decode(self, ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of token IDs back into text using information from metadata.json.
        Handles 'char' type via idx_to_char map and 'SentencePiece' type by loading the tokenizer.

        Args:
            ids: The token IDs to decode (Tensor or list).
            skip_special_tokens: Whether to skip PAD and EOS tokens during decoding (Currently only supports char).

        Returns:
            The decoded string.
        """
        metadata = self.get_metadata()
        tokenizer_type = metadata.get('tokenizer_type')
        # Convert tensor to list if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if tokenizer_type == 'CharTokenizer' or metadata.get('idx_to_char'):
            # --- Character Decoding Logic --- #
            idx_to_char = metadata.get('idx_to_char')
            if not idx_to_char:
                self.logger.warning("Cannot decode: idx_to_char mapping not found in metadata for char type.")
                return "[Decode unavailable: Missing idx_to_char]"
            # Handle special tokens if needed (example)
            pad_token_id = metadata.get('pad_token_id')
            eos_token_id = metadata.get('eos_token_id')

            try:
                # Cache integer map if keys are strings
                if self._idx_map_cache is None:
                    if idx_to_char and all(isinstance(k, str) for k in idx_to_char.keys()):
                         self._idx_map_cache = {int(k): v for k, v in idx_to_char.items()}
                    else:
                         self._idx_map_cache = idx_to_char # Assume keys are already integers or map is missing

                idx_map = self._idx_map_cache or {}

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
                self.logger.error(f"Error during character decoding: {e}", exc_info=True)
                return f"[Char Decode error: {e}]"
            # --- End Character Decoding --- #

        elif tokenizer_type == 'SentencePiece':
            # --- SentencePiece Decoding Logic --- #
            tokenizer = self._get_tokenizer()
            if tokenizer:
                try:
                    # Note: SentencePiece decode handles skip_special_tokens internally based on its model
                    # We might need to pass flags if the tokenizer interface supports it
                    return tokenizer.decode(ids)
                except Exception as e:
                    self.logger.error(f"Error during SentencePiece decoding: {e}", exc_info=True)
                    return f"[SP Decode error: {e}]"
            else:
                return "[Decode unavailable: Failed to load SP tokenizer]"
            # --- End SentencePiece Decoding --- #

        else:
            self.logger.warning(f"Decode not implemented for tokenizer type: {tokenizer_type}")
            return "[Decode unavailable: Unknown tokenizer type]" 