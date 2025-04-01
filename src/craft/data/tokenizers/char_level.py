from typing import Dict, Any, List
import os
import json
from collections import defaultdict
from .base import BaseTokenizer
import pickle
from pathlib import Path

class CharLevelTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any] = None):
        # Ensure config is a dict and set default model_type
        processed_config = config if config is not None else {}
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
        
    def load(self, model_path: str) -> None:
        """Load a pre-trained tokenizer from config and vocab files or legacy pickle."""
        config_path = Path(model_path) / 'tokenizer_config.json'
        vocab_path = Path(model_path) / 'vocab.json'
        legacy_pickle_path = Path(model_path) / 'char_tokenizer.pkl'

        if config_path.exists() and vocab_path.exists():
            # Load from JSON config and vocab
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            with open(vocab_path, 'r') as f:
                self.char_to_idx = json.load(f)
            # Rebuild dependent attributes
            self.idx_to_char = {int(idx): char for char, idx in self.char_to_idx.items()} 
            # Or maybe vocab stores idx as int? json loads strings. Assume string keys from json
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)
            if 'vocab_size' not in self.config or self.config.get('vocab_size') != self.vocab_size:
                 print(f"Warning: vocab_size in config ({self.config.get('vocab_size')}) doesn't match loaded vocab ({self.vocab_size}). Using loaded vocab size.")
                 self.config['vocab_size'] = self.vocab_size
            self._update_unk_from_config()

        elif legacy_pickle_path.exists():
            # Load from legacy pickle format
            print(f"Loading legacy pickle format from {legacy_pickle_path}")
            with open(legacy_pickle_path, 'rb') as f:
                legacy_data = pickle.load(f)
            self.config = legacy_data.get('config', {})
            self.char_to_idx = legacy_data.get('char_to_idx', {})
            self.idx_to_char = legacy_data.get('idx_to_char', {})
            # Ensure idx_to_char keys are integers if loaded from older pickles
            self.idx_to_char = {int(k): v for k, v in self.idx_to_char.items()}
            self.vocab_size = len(self.char_to_idx)
            if 'vocab_size' not in self.config:
                self.config['vocab_size'] = self.vocab_size
            self._update_unk_from_config()
            print("Legacy load complete. Consider resaving in the new format (config.json + vocab.json).")

        else:
            raise FileNotFoundError(f"Could not find tokenizer files (tokenizer_config.json and vocab.json) or legacy char_tokenizer.pkl in {model_path}")
        
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