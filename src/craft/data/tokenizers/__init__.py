from typing import Dict, Any
from .base import BaseTokenizer
from .char_level import CharLevelTokenizer
from .subword import SubwordTokenizer

def create_tokenizer(config: Dict[str, Any]) -> BaseTokenizer:
    """Create a tokenizer based on the configuration."""
    tokenizer_type = config.get('type', 'char_level')
    
    if tokenizer_type == 'char_level':
        return CharLevelTokenizer(config)
    elif tokenizer_type == 'subword':
        return SubwordTokenizer(config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}") 