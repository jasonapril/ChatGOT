import os
from src.data.tokenizers import create_tokenizer
import logging

def train_tokenizer():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Define paths
    train_path = "data/raw/got/game_of_thrones.txt"
    output_dir = "data/processed/got/tokenizer"
    
    # Create tokenizer config
    tokenizer_config = {
        "type": "subword",
        "vocab_size": 32000,
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>"
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tokenizer
    tokenizer = create_tokenizer(tokenizer_config)
    
    # Train tokenizer
    logger.info("Training subword tokenizer...")
    tokenizer.train(train_path, output_dir)
    
    # Save tokenizer config
    tokenizer.save(output_dir)
    
    # Log vocabulary size
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Tokenizer trained successfully. Vocabulary size: {vocab_size}")
    
    # Test tokenizer
    test_text = "The night is dark and full of"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    logger.info(f"Test encoding/decoding:")
    logger.info(f"Original: {test_text}")
    logger.info(f"Encoded: {encoded}")
    logger.info(f"Decoded: {decoded}")
    logger.info(f"Roundtrip successful: {test_text == decoded}")

if __name__ == "__main__":
    train_tokenizer() 