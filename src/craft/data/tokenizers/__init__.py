from typing import Dict, Any
from .base import Tokenizer
from .char import CharTokenizer
from .subword import SubwordTokenizer
from .sentencepiece import SentencePieceTokenizer

# Define public interface
__all__ = [
    "Tokenizer",
    "CharTokenizer",
    "SubwordTokenizer",
    "SentencePieceTokenizer",
    "create_tokenizer"
]

def create_tokenizer(config: Dict[str, Any]) -> Tokenizer:
    """Create a tokenizer based on the configuration."""
    tokenizer_type = config.get('type', 'char')
    
    if tokenizer_type == 'char':
        return CharTokenizer(config)
    elif tokenizer_type == 'subword':
        return SubwordTokenizer(config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}") 