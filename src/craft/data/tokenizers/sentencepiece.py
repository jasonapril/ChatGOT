import sentencepiece as spm
import os
from typing import List, Optional, Dict, Any
from .base import BaseTokenizer

class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vocab_size = config['vocab_size']
        self.model_prefix = config['model_prefix']
        self.character_coverage = config.get('character_coverage', 1.0)
        
        # Special tokens
        self.pad_token = config.get('pad_token', '<pad>')
        self.unk_token = config.get('unk_token', '<unk>')
        self.bos_token = config.get('bos_token', '<s>')
        self.eos_token = config.get('eos_token', '</s>')
        
        # Special token IDs
        self.pad_id = config.get('pad_id', 0)
        self.unk_id = config.get('unk_id', 1)
        self.bos_id = config.get('bos_id', 2)
        self.eos_id = config.get('eos_id', 3)
        
        self.sp = None
        self.model_path = None
        
    def train(self, text_file: str, output_dir: str) -> None:
        """Train the SentencePiece model on the input text file."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up model path
        self.model_path = os.path.join(output_dir, self.model_prefix)
        
        # Train the model
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=self.model_path,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type='bpe',
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_piece=self.pad_token,
            unk_piece=self.unk_token,
            bos_piece=self.bos_token,
            eos_piece=self.eos_token,
            train_extremely_large_corpus=True
        )
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{self.model_path}.model")
        
    def load(self, model_path: str) -> None:
        """Load a pre-trained SentencePiece model."""
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_path}.model")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        return self.sp.decode_ids(ids)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        return self.sp.get_piece_size()
    
    def save(self, output_dir: str) -> None:
        """Save the tokenizer configuration."""
        os.makedirs(output_dir, exist_ok=True)
        if self.model_path is None:
            raise RuntimeError("No model to save. Call train() first.")
        # The model files (.model and .vocab) are already saved during training
        # We just need to save the configuration
        config = {
            'vocab_size': self.vocab_size,
            'model_prefix': self.model_prefix,
            'character_coverage': self.character_coverage,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_id': self.pad_id,
            'unk_id': self.unk_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id
        }
        import json
        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
            json.dump(config, f, indent=2) 