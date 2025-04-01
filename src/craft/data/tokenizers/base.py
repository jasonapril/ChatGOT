from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseTokenizer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def train(self, text_file: str, output_dir: str) -> None:
        """Train the tokenizer on the input text file."""
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load a pre-trained tokenizer."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        pass
    
    @abstractmethod
    def save(self, output_dir: str) -> None:
        """Save the tokenizer configuration."""
        pass 