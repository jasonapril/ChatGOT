from typing import Dict, Any, List
import os
import json
from collections import defaultdict
from .base import BaseTokenizer

class CharLevelTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def train(self, text_file: str, output_dir: str) -> None:
        """Train the tokenizer by building character vocabulary."""
        # Count character frequencies
        char_freq = defaultdict(int)
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                for char in line:
                    char_freq[char] += 1
        
        # Create vocabulary
        sorted_chars = sorted(char_freq.keys())
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Save vocabulary
        os.makedirs(output_dir, exist_ok=True)
        self.save(output_dir)
        
    def load(self, model_path: str) -> None:
        """Load a pre-trained tokenizer."""
        config_path = os.path.join(model_path, 'tokenizer_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.char_to_idx = config['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in config['idx_to_char'].items()}
        self.vocab_size = len(self.char_to_idx)
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx[char] for char in text]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.idx_to_char[idx] for idx in ids)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size
    
    def save(self, output_dir: str) -> None:
        """Save the tokenizer configuration."""
        config = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }
        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
            json.dump(config, f, indent=2) 