from typing import Dict, Any, List
import os
import json
from collections import defaultdict
from .base import BaseTokenizer
import pickle
from pathlib import Path

class CharLevelTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any] = None):
        # Ensure config is a dict, use a copy, and set default model_type
        processed_config = config.copy() if config is not None else {} # Use .copy()
        processed_config.setdefault('model_type', 'char_level')
        super().__init__(processed_config)
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.unk_token: str | None = None
        self.unk_token_id: int | None = None
        self._update_unk_from_config()
        
    def _update_unk_from_config(self):
        """Sets unk_token and unk_token_id based on current config and vocab."""
        self.unk_token = self.config.get('special_tokens', {}).get('unk')
        if self.unk_token:
            self.unk_token_id = self.char_to_idx.get(self.unk_token)
        else:
            self.unk_token_id = None

    def train(self, text_file: str, output_dir: str) -> None:
        """Train the tokenizer by building character vocabulary."""
        char_freq = defaultdict(int)
        chars = set()
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                for char in line:
                    chars.add(char)
                    char_freq[char] += 1 # Keep freq counting? Maybe not needed now.

        # Add UNK token to char set if specified in config
        self._update_unk_from_config() # Ensure unk_token is set based on initial config
        if self.unk_token and self.unk_token not in chars:
            chars.add(self.unk_token)

        # Create vocabulary
        sorted_chars = sorted(list(chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.config['vocab_size'] = self.vocab_size # Update config with final vocab size

        # Update unk_token_id based on the final vocabulary
        self._update_unk_from_config()

        # Save config and vocabulary separately
        os.makedirs(output_dir, exist_ok=True)
        self.save(output_dir)
        
    @classmethod
    def load(cls, model_path: str) -> 'CharLevelTokenizer':
        """Load a pre-trained tokenizer from config and vocab files or legacy pickle."""
        model_path_obj = Path(model_path)
        config_path = model_path_obj / 'tokenizer_config.json'
        vocab_path = model_path_obj / 'vocab.json'
        legacy_pickle_path = model_path_obj / 'char_tokenizer.pkl'

        loaded_config = None
        loaded_char_map = None
        loaded_idx_map = None

        if config_path.exists() and vocab_path.exists():
            # Load from modern JSON format
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            with open(vocab_path, 'r') as f:
                loaded_char_map = json.load(f)
            # Rebuild idx_to_char
            loaded_idx_map = {int(idx): char for char, idx in loaded_char_map.items()}

        elif legacy_pickle_path.exists():
            # Load from legacy pickle format
            with open(legacy_pickle_path, 'rb') as f:
                legacy_data = pickle.load(f)
            if not all(k in legacy_data for k in ['config', 'char_to_idx', 'idx_to_char']):
                 raise ValueError(f"Legacy pickle file {legacy_pickle_path} is missing required keys.")
            loaded_config = legacy_data['config']
            loaded_char_map = legacy_data['char_to_idx']
            loaded_idx_map = legacy_data['idx_to_char']

        else:
            raise FileNotFoundError(
                f"Could not find tokenizer files (tokenizer_config.json/vocab.json or char_tokenizer.pkl) "
                f"in {model_path}"
            )

        # Ensure data was loaded
        if loaded_config is None or loaded_char_map is None or loaded_idx_map is None:
             raise RuntimeError("Failed to load tokenizer data correctly.")

        # Calculate vocab size
        vocab_size = len(loaded_char_map)
        # Ensure vocab_size is reflected in the loaded config dict
        loaded_config['vocab_size'] = vocab_size

        # === Refactored Approach ===
        # 1. Initialize an empty instance
        tokenizer = cls() 

        # 2. Directly assign loaded attributes
        tokenizer.config = loaded_config        # Assign the complete config dict
        tokenizer.char_to_idx = loaded_char_map
        tokenizer.idx_to_char = loaded_idx_map
        tokenizer.vocab_size = vocab_size      # Assign the calculated vocab size

        # 3. Update UNK token info based on assigned attributes
        tokenizer._update_unk_from_config()
        # ==========================

        return tokenizer

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs, handling unknown characters."""
        token_ids = []
        if self.unk_token_id is not None:
            for char in text:
                token_ids.append(self.char_to_idx.get(char, self.unk_token_id))
        else:
            # If no unk token, skip unknown characters
            for char in text:
                token_id = self.char_to_idx.get(char)
                if token_id is not None:
                    token_ids.append(token_id)
        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text, handling unknown IDs."""
        # Determine character to use for unknown IDs
        unknown_char = self.unk_token if self.unk_token else '' # Or maybe '?'?
        return ''.join(self.idx_to_char.get(idx, unknown_char) for idx in ids)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        # Ensure vocab_size attribute is consistent
        return len(self.char_to_idx) if self.char_to_idx else 0
    
    def save(self, output_dir: str) -> None:
        """Save the tokenizer config and vocabulary to separate files."""
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, 'tokenizer_config.json')
        vocab_path = os.path.join(output_dir, 'vocab.json')

        # Save configuration
        with open(config_path, 'w') as f:
            # Ensure config exists before saving
            config_to_save = self.config if self.config is not None else {}
            json.dump(config_to_save, f, indent=2)

        # Save vocabulary (char_to_idx mapping)
        with open(vocab_path, 'w') as f:
            json.dump(self.char_to_idx, f, indent=2) 