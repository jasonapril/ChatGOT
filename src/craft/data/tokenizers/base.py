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
        **kwargs: Any
    ):
        """Initializes the base tokenizer with special token definitions."""
        self.pad_token: Optional[str] = pad_token
        self.unk_token: Optional[str] = unk_token
        self.bos_token: Optional[str] = bos_token
        self.eos_token: Optional[str] = eos_token
        self.pad_id: Optional[int] = pad_id
        self.unk_id: Optional[int] = unk_id
        self.bos_id: Optional[int] = bos_id
        self.eos_id: Optional[int] = eos_id
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

        # Initialize the special tokens map
        self.special_tokens_map: Dict[str, str] = {}
        if self.pad_token is not None: self.special_tokens_map['pad'] = self.pad_token
        if self.unk_token is not None: self.special_tokens_map['unk'] = self.unk_token
        if self.bos_token is not None: self.special_tokens_map['bos'] = self.bos_token
        if self.eos_token is not None: self.special_tokens_map['eos'] = self.eos_token
        # Add other potential roles if needed
    
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