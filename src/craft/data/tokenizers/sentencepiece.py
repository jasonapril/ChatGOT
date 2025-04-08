import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import shutil

import sentencepiece as spm # type: ignore[import-untyped]

from .base import Tokenizer
from .sentencepiece_trainer import train_sentencepiece_model # Import trainer function

logger = logging.getLogger(__name__)

METADATA_FILENAME = "tokenizer_metadata.json" # Standardized name

class SentencePieceTokenizer(Tokenizer):
    """
    A tokenizer using Google's SentencePiece library.
    Loads a pre-trained model and provides the standard Tokenizer interface.
    Training is handled separately by `sentencepiece_trainer.train_sentencepiece_model`.
    """
    MODEL_FILENAME = "sentencepiece.model" # Standardize model filename

    def __init__(
        self,
        model_path: str, # Path to the DIRECTORY containing the model and metadata
        add_bos_token: Optional[bool] = None,
        add_eos_token: Optional[bool] = None,
        # Removed training-specific args: vocab_size, model_prefix, model_type, character_coverage
        # special_tokens handled via metadata loading
        **kwargs: Any, # Pass remaining kwargs to base Tokenizer
    ):
        """
        Initializes the SentencePieceTokenizer by loading a pre-trained model.

        Args:
            model_path: Path to the *directory* containing the trained
                        'sentencepiece.model' and 'tokenizer_metadata.json' files.
            add_bos_token: Explicitly control BOS token addition (overrides metadata if set).
            add_eos_token: Explicitly control EOS token addition (overrides metadata if set).
            **kwargs: Additional arguments passed to the base Tokenizer.
        """
        self.model_dir = Path(model_path)
        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self.metadata: Dict[str, Any] = {}
        self.model_file = self.model_dir / self.MODEL_FILENAME
        self.metadata_file = self.model_dir / METADATA_FILENAME

        loaded_special_tokens = {}
        loaded_config = {}

        # --- Load Model and Metadata --- #
        if not self.model_dir.is_dir():
             raise FileNotFoundError(f"SentencePiece model directory not found: {self.model_dir}")
        if not self.model_file.is_file():
             raise FileNotFoundError(f"SentencePiece model file not found: {self.model_file}")
        if not self.metadata_file.is_file():
             raise FileNotFoundError(f"SentencePiece metadata file not found: {self.metadata_file}")

        try:
            # Load metadata first
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                logger.info(f"Loaded SentencePiece metadata from: {self.metadata_file}")
                # Extract info needed for base class and self
                loaded_config['vocab_size'] = self.metadata.get('vocab_size')
                loaded_special_tokens = self.metadata.get('special_tokens_map', {}) # Use the saved map
                # Determine add_bos/eos primarily from metadata, allow override
                # Look for explicit flags saved during training (if added to metadata later)
                meta_add_bos = self.metadata.get("add_bos_as_control", False)
                meta_add_eos = self.metadata.get("add_eos_as_control", False)
                self._add_bos_token = add_bos_token if add_bos_token is not None else meta_add_bos
                self._add_eos_token = add_eos_token if add_eos_token is not None else meta_add_eos

            # Load SentencePiece model
            self.sp_model = spm.SentencePieceProcessor(str(self.model_file))
            logger.info(f"Loaded SentencePiece model from: {self.model_file}")

        except Exception as e:
            logger.error(f"Failed to load SentencePiece model or metadata from {self.model_dir}: {e}")
            raise RuntimeError(f"Failed to load SentencePiece tokenizer from {self.model_dir}") from e
        # ---------------------------- #

        # Initialize base class - Use loaded vocab size and special tokens map
        # Pass add_bos/eos determined above
        super().__init__(
            vocab_size=loaded_config.get('vocab_size'),
            special_tokens=loaded_special_tokens,
            add_bos_token=self._add_bos_token,
            add_eos_token=self._add_eos_token,
            **kwargs,
        )

        # Sync vocab/ids from loaded SentencePiece model
        # This ensures internal state matches the loaded sp_model and metadata
        self._sync_vocab_from_sp_model()


    def _sync_vocab_from_sp_model(self) -> None:
        """Updates internal vocab mappings from the loaded sp_model and metadata."""
        if not self.sp_model:
             raise RuntimeError("SentencePiece model (sp_model) is not loaded.")

        # Cast the result of get_piece_size
        self.vocab_size = cast(int, self.sp_model.get_piece_size())
        
        # Get the special tokens map either from loaded metadata or the base class defaults
        current_special_map: Dict[str, str] = self.special_tokens_map # Map initialized by base class
        if 'special_tokens_map' in self.metadata:
            # Ensure the base class map is also updated if metadata overrides it
            # Cast the loaded metadata to the expected type
            self.special_tokens_map = cast(Dict[str, str], self.metadata['special_tokens_map'])
            current_special_map = self.special_tokens_map

        # Update specific token attributes and IDs from the map
        self.bos_token = current_special_map.get('bos')
        self.eos_token = current_special_map.get('eos')
        self.unk_token = current_special_map.get('unk')
        self.pad_token = current_special_map.get('pad')
        # Add others if defined (e.g., mask, cls, sep)
        # ...

        # Get IDs from the loaded SP model
        self.unk_token_id = self.sp_model.unk_id() if self.sp_model.unk_id() != -1 else None
        self.bos_token_id = self.sp_model.bos_id() if self.sp_model.bos_id() != -1 else None
        self.eos_token_id = self.sp_model.eos_id() if self.sp_model.eos_id() != -1 else None
        self.pad_token_id = self.sp_model.pad_id() if self.sp_model.pad_id() != -1 else None

        # Rebuild vocab mappings using SP model directly
        self.idx_to_token = {i: self.sp_model.id_to_piece(i) for i in range(self.vocab_size)}
        self.token_to_idx = {v: k for k, v in self.idx_to_token.items()}

        # Verify/log special token IDs match expectations from the map
        for role, token in current_special_map.items():
             if token:
                 try:
                     sp_id = self.sp_model.piece_to_id(token)
                     assigned_id = getattr(self, f"{role}_token_id", None)
                     # Log if the role ID doesn't match SP's ID for that token string
                     # This shouldn't happen if training args were set correctly, but good check.
                     if assigned_id != sp_id:
                          logger.warning(f"Special token mismatch for role '{role}' ('{token}'): Assigned ID {assigned_id}, SP model ID {sp_id}. Using SP model ID.")
                          # Correct the assigned ID based on the loaded model
                          setattr(self, f"{role}_token_id", sp_id if sp_id != -1 else None)
                          
                     # Ensure our token string is in the vocab map
                     if sp_id >= 0 and sp_id < self.vocab_size:
                         self.idx_to_token[sp_id] = token
                         self.token_to_idx[token] = sp_id
                     else:
                          logger.warning(f"Special token '{token}' (role: {role}) has invalid ID {sp_id} in SP model.")
                 except ValueError:
                      logger.warning(f"Special token '{token}' (role: {role}) defined in map but not found in loaded SentencePiece model vocab.")
                      # Ensure the ID attribute is None if token not found
                      setattr(self, f"{role}_token_id", None)

        logger.debug(f"Synced vocab from SentencePiece model. Size: {self.vocab_size}")
        # Log final assigned IDs
        logger.debug(f"  UNK ID: {self.unk_token_id} ('{self.unk_token}')")
        logger.debug(f"  BOS ID: {self.bos_token_id} ('{self.bos_token}')")
        logger.debug(f"  EOS ID: {self.eos_token_id} ('{self.eos_token}')")
        logger.debug(f"  PAD ID: {self.pad_token_id} ('{self.pad_token}')")

    @classmethod
    def load_from_prefix(cls, model_path: Union[str, Path], **kwargs: Any) -> "SentencePieceTokenizer":
        """Loads a tokenizer from a directory containing model and metadata files."""
        # The main loading logic is now in __init__
        return cls(model_path=str(model_path), **kwargs)

    # --- REMOVE train method --- #
    # def train(...): ...

    # --- Standard Tokenizer Methods --- #

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text into token IDs."""
        if not self.sp_model:
            raise RuntimeError("SentencePiece model not loaded.")

        # Let SentencePiece handle bos/eos based on its loaded config or options.
        # Our add_bos/eos flags were mainly for base class consistency.
        # Check sp_model options if needed, but usually rely on its defaults.
        # Cast the result of the underlying SentencePiece encode
        encoded_ids: List[int] = cast(List[int], self.sp_model.encode(text))

        # Manually add BOS/EOS based on our flags if SP didn't do it?
        # This might be complex. Simpler to configure SP correctly during training.
        # For now, assume SP handles it based on how it was trained/loaded.
        # However, the base class expects *us* to handle it via add_special_tokens flag.

        # Adhere to base class `add_special_tokens` contract:
        bos_id = self.bos_token_id
        eos_id = self.eos_token_id

        if add_special_tokens:
            final_ids = []
            if self._add_bos_token and bos_id is not None:
                 # Check if SP already added it (usually first token is bos_id)
                 if not encoded_ids or encoded_ids[0] != bos_id:
                     final_ids.append(bos_id)
            final_ids.extend(encoded_ids)
            if self._add_eos_token and eos_id is not None:
                 # Check if SP already added it (usually last token is eos_id)
                 if not encoded_ids or encoded_ids[-1] != eos_id:
                      final_ids.append(eos_id)
            return final_ids
        else:
             # If add_special_tokens is False, try to strip BOS/EOS if SP added them
             stripped_ids = encoded_ids
             if self._add_bos_token and bos_id is not None and stripped_ids and stripped_ids[0] == bos_id:
                 stripped_ids = stripped_ids[1:]
             if self._add_eos_token and eos_id is not None and stripped_ids and stripped_ids[-1] == eos_id:
                  stripped_ids = stripped_ids[:-1]
             return stripped_ids

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encodes a batch of texts into token IDs."""
        # SentencePiece processor has encode method that takes list
        if not self.sp_model:
            raise RuntimeError("SentencePiece model not loaded.")
        
        encoded_batch = self.sp_model.encode(texts)
        
        # Apply add_special_tokens logic to each sequence in the batch
        processed_batch = []
        for encoded_ids in encoded_batch:
             bos_id = self.bos_token_id
             eos_id = self.eos_token_id
             if add_special_tokens:
                 final_ids = []
                 if self._add_bos_token and bos_id is not None:
                     if not encoded_ids or encoded_ids[0] != bos_id:
                         final_ids.append(bos_id)
                 final_ids.extend(encoded_ids)
                 if self._add_eos_token and eos_id is not None:
                     if not encoded_ids or encoded_ids[-1] != eos_id:
                          final_ids.append(eos_id)
                 processed_batch.append(final_ids)
             else:
                 stripped_ids = encoded_ids
                 if self._add_bos_token and bos_id is not None and stripped_ids and stripped_ids[0] == bos_id:
                     stripped_ids = stripped_ids[1:]
                 if self._add_eos_token and eos_id is not None and stripped_ids and stripped_ids[-1] == eos_id:
                     stripped_ids = stripped_ids[:-1]
                 processed_batch.append(stripped_ids)
                 
        return processed_batch


    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back into text."""
        if not self.sp_model:
            raise RuntimeError("SentencePiece model not loaded.")
        # SP expects List[int]
        # Cast the result
        return cast(str, self.sp_model.decode(token_ids))

    def batch_decode(self, list_of_token_ids: List[List[int]]) -> List[str]:
         """Decodes a batch of token ID lists back into text."""
         if not self.sp_model:
             raise RuntimeError("SentencePiece model not loaded.")
         # Cast the result
         return cast(List[str], self.sp_model.decode(list_of_token_ids))

    # --- Vocab Methods --- #
    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary mapping tokens to their IDs."""
        # Ensure token_to_idx is populated correctly by __init__ / _sync_vocab_from_sp_model
        if not self.token_to_idx:
            raise RuntimeError("Vocab mappings not initialized")
        return self.token_to_idx.copy()

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.vocab_size

    def token_to_id(self, token: str) -> Optional[int]:
        """Converts a token string to its ID."""
        # Use the internal map which is synced with SP model and handles special tokens
        return self.token_to_idx.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Converts a token ID back to its string representation."""
        # Ensure idx_to_token is populated correctly
        return self.idx_to_token.get(token_id)

    # --- Save Method --- #
    def save(self, output_dir: Union[str, Path], **kwargs: Any) -> None:
         """
         Saves the SentencePiece model and tokenizer metadata to the specified directory.

         This method primarily copies the existing model file and saves a new
         metadata file containing configuration used by this Tokenizer class instance.

         Args:
             output_dir: Directory to save the tokenizer files.
             **kwargs: Optional arguments (currently unused).
         """
         output_dir = Path(output_dir)
         output_dir.mkdir(parents=True, exist_ok=True)

         target_model_file = output_dir / self.MODEL_FILENAME
         target_metadata_file = output_dir / METADATA_FILENAME

         if not self.model_file.is_file():
             raise RuntimeError("Cannot save tokenizer: Original model file not found or not loaded.")

         try:
             # Copy the existing model file
             shutil.copyfile(self.model_file, target_model_file)
             logger.info(f"Copied SentencePiece model file to: {target_model_file}")
         except Exception as e:
             logger.error(f"Failed to copy SentencePiece model file to {target_model_file}: {e}", exc_info=True)
             raise RuntimeError("Failed to save SentencePiece model file") from e

         # Prepare metadata to save (use current instance state)
         save_metadata = {
             "model_prefix": self.model_dir.name, # Save the original directory name as prefix?
             "vocab_size": self.vocab_size,
             "model_type": self.metadata.get('model_type', 'unknown'), # Get from loaded meta if possible
             "character_coverage": self.metadata.get('character_coverage', 0.9995),
             "special_tokens_map": self.special_tokens_map, # Save the current map
             "add_bos_as_control": self._add_bos_token,
             "add_eos_as_control": self._add_eos_token,
             # Add any other relevant metadata from self.metadata if desired
         }

         try:
             with open(target_metadata_file, "w", encoding="utf-8") as f:
                 json.dump(save_metadata, f, indent=4)
             logger.info(f"Saved tokenizer metadata to: {target_metadata_file}")
         except Exception as e:
             logger.error(f"Failed to save tokenizer metadata to {target_metadata_file}: {e}", exc_info=True)
             # Should saving fail if metadata fails? Maybe not critical.
             logger.warning("Tokenizer metadata saving failed, but model file was copied.")

    # Config property (optional but can be useful)
    @property
    def config(self) -> Dict[str, Any]:
        """Returns a dictionary representing the tokenizer's configuration."""
        # Return relevant loaded/configured state
        return {
            "model_path": str(self.model_dir),
            "vocab_size": self.vocab_size,
            "special_tokens_map": self.special_tokens_map,
            "add_bos_token": self._add_bos_token,
            "add_eos_token": self._add_eos_token,
            # Add other relevant fields from self.metadata if needed
            "model_type": self.metadata.get("model_type"),
        }

    # --- Abstract Method Implementations (Not Used) --- #

    def train(self, text_file: str, output_dir: str) -> None:
        """Training is handled externally by sentencepiece_trainer.py."""
        raise NotImplementedError("SentencePieceTokenizer training is handled by a separate script.")

    def load(self, model_path: str) -> None:
        """Loading is handled during __init__ or via load_from_prefix classmethod."""
        raise NotImplementedError("Use SentencePieceTokenizer(model_path=...) or SentencePieceTokenizer.load_from_prefix(...) to load.")