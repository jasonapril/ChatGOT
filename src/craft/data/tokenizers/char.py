from typing import Dict, Any, List, Optional
import os
import json
from collections import defaultdict
from .base import Tokenizer
import pickle
from pathlib import Path
import logging

class CharTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        # **Call super().__init__ FIRST** to initialize base attributes
        # including special tokens and their IDs based on kwargs.
        super().__init__(**kwargs)

        # Initialize CharTokenizer specific attributes
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        # unk_token_id is used directly in encode/decode, so ensure it reflects base state
        self.unk_token_id: Optional[int] = self.unk_id # Get ID from base class

        # Store the rest of the config if provided
        # Filter out base class args that are already handled
        base_args = ['pad_token', 'unk_token', 'bos_token', 'eos_token', 'pad_id', 'unk_id', 'bos_id', 'eos_id']
        self.config = {k: v for k, v in kwargs.items() if k not in base_args}
        self.config.setdefault('model_type', 'char') # Ensure model_type is set

        # No need to call _update_unk_from_base here, as base class handles it.
        logger = logging.getLogger(__name__)
        logger.debug(f"CharTokenizer initialized. UNK token: '{self.unk_token}', ID: {self.unk_id}")
        logger.debug(f"Internal unk_token_id set to: {self.unk_token_id}")

    def train(self, text_file: str, output_dir: str) -> None:
        """Train the tokenizer by building character vocabulary."""
        char_freq = defaultdict(int)
        chars = set()
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                for char in line:
                    chars.add(char)
                    char_freq[char] += 1

        logger = logging.getLogger(__name__)
        
        # Get special tokens directly from base class attributes
        special_tokens_to_add = {
            self.pad_token, 
            self.unk_token, 
            self.bos_token, 
            self.eos_token
        }
        
        # Add defined special tokens if they are not already in the text characters
        for token_str in special_tokens_to_add:
             if token_str is not None and token_str not in chars:
                  chars.add(token_str)
                  logger.debug(f"Added special token '{token_str}' from base class to vocabulary set.")
             elif token_str is not None: # Token is defined but already exists
                  logger.debug(f"Special token '{token_str}' already found in text, not adding separately.")
             # If token_str is None, do nothing

        # Create vocabulary
        sorted_chars = sorted(list(chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Update config dict with final vocab size
        self.config['vocab_size'] = self.vocab_size 

        # **Update base class IDs and self.unk_token_id based on the final vocabulary**
        self._sync_special_ids_with_vocab()

        # Save config and vocabulary separately
        os.makedirs(output_dir, exist_ok=True)
        self.save(output_dir) # save method now uses base class attributes

    def _sync_special_ids_with_vocab(self):
         """Update base class special token IDs and self.unk_token_id based on the final vocabulary."""
         logger = logging.getLogger(__name__)
         # Iterate through the special tokens defined in the base class
         for token_name, id_name in [("pad_token", "pad_id"), 
                                     ("unk_token", "unk_id"), 
                                     ("bos_token", "bos_id"), 
                                     ("eos_token", "eos_id")]:
             token_str = getattr(self, token_name, None)
             if token_str is not None:
                 token_id = self.char_to_idx.get(token_str)
                 if token_id is not None:
                     # Update the ID attribute on the instance (which updates base)
                     setattr(self, id_name, token_id)
                     logger.debug(f"Synced ID for '{token_str}' ({id_name}) to {token_id}")
                 else:
                     logger.warning(f"Special token '{token_str}' defined but not found in final vocabulary. Setting its ID to None.")
                     setattr(self, id_name, None)
             else:
                 # Ensure ID is None if token string is None
                 setattr(self, id_name, None)
         # Specifically update self.unk_token_id after syncing base self.unk_id
         self.unk_token_id = self.unk_id
         logger.debug(f"Post-sync: internal unk_token_id set to: {self.unk_token_id}")

    @classmethod
    def load(cls, load_dir: str):
        """Load the tokenizer config and vocabulary from files in a directory."""
        config_path = os.path.join(load_dir, 'tokenizer_config.json')
        vocab_path = os.path.join(load_dir, 'vocab.json')
        legacy_pickle_path = os.path.join(load_dir, 'char_tokenizer.pkl')

        loaded_config = None
        loaded_char_map = None
        loaded_idx_map = None
        special_tokens_from_config = {}

        logger = logging.getLogger(__name__)
        # Try loading from JSON files first
        if os.path.exists(config_path) and os.path.exists(vocab_path):
            logger.info(f"Loading CharTokenizer from JSON: {config_path}, {vocab_path}")
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    loaded_char_map = json.load(f)
                # Reconstruct idx_to_char from char_to_idx
                loaded_idx_map = {v: k for k, v in loaded_char_map.items()}
                # Extract special tokens from config for instantiation
                # Base Tokenizer __init__ expects keys like unk_token, unk_id etc directly
                special_tokens_from_config = loaded_config.pop('special_tokens', {}) # Remove from main config

            except Exception as e:
                logger.error(f"Error loading tokenizer from JSON files in {load_dir}: {e}", exc_info=True)
                raise IOError(f"Error loading tokenizer from JSON files in {load_dir}") from e

        # Fallback to legacy pickle format if JSONs not found or failed
        elif os.path.exists(legacy_pickle_path):
            logger.warning(f"Loading CharTokenizer from legacy pickle format: {legacy_pickle_path}")
            with open(legacy_pickle_path, 'rb') as f:
                legacy_data = pickle.load(f)
            if not all(k in legacy_data for k in ['config', 'char_to_idx', 'idx_to_char']):
                 raise ValueError(f"Legacy pickle file {legacy_pickle_path} is missing required keys.")
            loaded_config = legacy_data['config']
            loaded_char_map = legacy_data['char_to_idx']
            loaded_idx_map = legacy_data['idx_to_char']
            # Extract special tokens from legacy config
            special_tokens_from_config = loaded_config.pop('special_tokens', {})

        else:
            raise FileNotFoundError(
                f"Could not find tokenizer files (tokenizer_config.json/vocab.json or char_tokenizer.pkl) "
                f"in {load_dir}"
            )

        if loaded_config is None or loaded_char_map is None or loaded_idx_map is None:
             raise RuntimeError("Failed to load tokenizer data correctly.")

        vocab_size = len(loaded_char_map)
        loaded_config['vocab_size'] = vocab_size # Ensure vocab_size is in config

        # === Instantiate Approach ===
        # 1. Prepare init args: combine loaded config and extracted special tokens
        init_args = loaded_config.copy()
        init_args.update(special_tokens_from_config)

        # 2. Instantiate the class
        # Base __init__ will handle setting special tokens/IDs from init_args
        tokenizer = cls(**init_args)

        # 3. Directly assign loaded vocabulary mappings
        tokenizer.char_to_idx = loaded_char_map
        tokenizer.idx_to_char = loaded_idx_map
        tokenizer.vocab_size = vocab_size # Assign the calculated vocab size

        # 4. **Sync base IDs with the loaded vocabulary**
        # This ensures self.pad_id, self.unk_id etc. match the loaded vocab
        tokenizer._sync_special_ids_with_vocab()

        logger.info(f"[CharTokenizer.load] Loaded instance. Vocab size: {tokenizer.vocab_size}. UNK token: '{tokenizer.unk_token}', ID: {tokenizer.unk_id}")
        logger.debug(f"[CharTokenizer.load] Internal unk_token_id: {tokenizer.unk_token_id}")

        return tokenizer

    def encode(self, text: str) -> List[int]:
        # Use self.unk_token_id which is synced with self.unk_id
        unk = self.unk_token_id
        encoded = []
        for char in text:
            idx = self.char_to_idx.get(char)
            if idx is not None:
                encoded.append(idx)
            elif unk is not None:
                encoded.append(unk)
            # If idx is None and unk is None, skip the character
        return encoded

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text, handling unknown IDs."""
        # Use self.unk_token (from base class) for unknown characters
        unknown_char = self.unk_token if self.unk_token is not None else ''
        # Use the instance's idx_to_char map
        return ''.join(self.idx_to_char.get(idx, unknown_char) for idx in ids)

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return self.vocab_size

    def save(self, output_dir: str) -> None:
        """Save the tokenizer config and vocabulary to separate files."""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, 'tokenizer_config.json')
        vocab_path = os.path.join(output_dir, 'vocab.json')

        # Prepare config to save - start with the instance's config dict
        config_to_save = self.config.copy()
        config_to_save['vocab_size'] = self.get_vocab_size()

        # Explicitly add special tokens and their IDs *from base class attributes*
        # This ensures the saved config reflects the actual state
        special_tokens_dict = {
             "pad_token": self.pad_token,
             "unk_token": self.unk_token,
             "bos_token": self.bos_token,
             "eos_token": self.eos_token,
             "pad_id": self.pad_id,
             "unk_id": self.unk_id,
             "bos_id": self.bos_id,
             "eos_id": self.eos_id,
        }
        # Clean up None values and add to config
        config_to_save['special_tokens'] = {k: v for k, v in special_tokens_dict.items() if v is not None}

        # Save configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Saving CharTokenizer config to: {config_path}")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2)

        # Save vocabulary (char_to_idx mapping)
        logger.info(f"Saving CharTokenizer vocabulary to: {vocab_path}")
        if not self.char_to_idx:
             logger.warning("Vocabulary (char_to_idx) is empty, saving empty vocab file.")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_idx, f, indent=2) 