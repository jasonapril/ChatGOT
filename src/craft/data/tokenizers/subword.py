from typing import Dict, Any, List
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from .base import BaseTokenizer
import logging

class SubwordTokenizer(BaseTokenizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        logging.info(f"Initializing SubwordTokenizer with config: {config}")
        self.vocab_size = config['vocab_size']
        self.tokenizer = None
        
        # Special tokens
        self.pad_token = config.get('pad_token', '<pad>')
        self.unk_token = config.get('unk_token', '<unk>')
        self.bos_token = config.get('bos_token', '<s>')
        self.eos_token = config.get('eos_token', '</s>')
        
        logging.info(f"SubwordTokenizer initialized with vocab_size={self.vocab_size}")
        
    def train(self, files: List[str], output_dir: str) -> None:
        """Train the tokenizer on a list of input text files."""
        logging.info(f"Training SubwordTokenizer on {len(files)} files.")
        if not files:
            raise ValueError("Input file list cannot be empty for training.")
        logging.debug(f"Training files: {files}")
        
        # Initialize a BPE tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Configure the trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token
            ]
        )
        
        # Train the tokenizer
        self.tokenizer.train(files=files, trainer=trainer)
        
        # Save the tokenizer
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        logging.info(f"SubwordTokenizer trained and saved to {output_dir}")
        
    def load(self, model_path: str) -> None:
        """Load a pre-trained tokenizer."""
        logging.info(f"Loading SubwordTokenizer from {model_path}")
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        logging.info("SubwordTokenizer loaded successfully")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call train() or load() first.")
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call train() or load() first.")
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        if self.tokenizer is None:
            return self.vocab_size
        return self.tokenizer.get_vocab_size()
    
    def save(self, output_dir: str) -> None:
        """Save the tokenizer configuration."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json")) 