from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

class Tokenizer(ABC):
    def __init__(
        self, 
        pad_token: Optional[str] = "<pad>", 
        unk_token: Optional[str] = "<unk>", 
        bos_token: Optional[str] = "<bos>", 
        eos_token: Optional[str] = "<eos>",
        pad_id: Optional[int] = -1, 
        unk_id: Optional[int] = 0, 
        bos_id: Optional[int] = 1, 
        eos_id: Optional[int] = 2,
        **kwargs
    ):
        """Initializes the base tokenizer with special token definitions."""
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self._init_kwargs = kwargs
        if kwargs:
            logger = logging.getLogger(__name__)
            logger.debug(f"Tokenizer base __init__ ignoring unused kwargs: {kwargs}")

        self._base_special_tokens = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
            self.bos_token: self.bos_id,
            self.eos_token: self.eos_id
        }
    
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