#!/usr/bin/env python
"""
Script to examine processed data structure
"""
import os
import pickle
import json
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to examine data."""
    logger.info("Starting data examination")
    
    # Load the processed data
    try:
        data_path = os.path.join("processed_data", "got_char_data.pkl")
        vocab_path = os.path.join("processed_data", "vocab.json")
        
        logger.info(f"Loading processed data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Print the keys in the data dictionary
        logger.info(f"Data keys: {data.keys()}")
        
        # Print the type and length of each item in data
        for key, value in data.items():
            logger.info(f"Key: {key}, Type: {type(value)}, Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            
            # If it's a list, print the type and shape of the first item
            if isinstance(value, list) and len(value) > 0:
                first_item = value[0]
                logger.info(f"  First item type: {type(first_item)}")
                logger.info(f"  First item length: {len(first_item) if hasattr(first_item, '__len__') else 'N/A'}")
                if hasattr(first_item, '__getitem__') and len(first_item) > 0:
                    if isinstance(first_item, tuple) and len(first_item) >= 2:
                        logger.info(f"  First tuple item type: {type(first_item[0])}")
                        logger.info(f"  First tuple item shape: {len(first_item[0]) if hasattr(first_item[0], '__len__') else 'N/A'}")
                        logger.info(f"  Second tuple item type: {type(first_item[1])}")
                        logger.info(f"  Second tuple item shape: {len(first_item[1]) if hasattr(first_item[1], '__len__') else 'N/A'}")
                        logger.info(f"  First few elements of first tuple item: {first_item[0][:10] if hasattr(first_item[0], '__getitem__') else first_item[0]}")
                    else:
                        logger.info(f"  First few elements: {first_item[:10]}")
                        
        # Print additional information about train_sequences
        if 'train_sequences' in data:
            ts = data['train_sequences']
            if ts and len(ts) > 0:
                item = ts[0]
                logger.info(f"Example train sequence (first item): {item}")
                if isinstance(item, tuple) and len(item) >= 2:
                    logger.info(f"  Item[0]: {item[0][:20] if hasattr(item[0], '__getitem__') else item[0]} (type: {type(item[0])})")
                    logger.info(f"  Item[1]: {item[1][:20] if hasattr(item[1], '__getitem__') else item[1]} (type: {type(item[1])})")
        
        # Print sequence_length and metadata if available
        if 'sequence_length' in data:
            logger.info(f"Sequence length: {data['sequence_length']}")
            
        if 'metadata' in data:
            logger.info(f"Metadata: {data['metadata']}")
        
        logger.info(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Print vocabulary structure
        logger.info(f"Vocabulary keys: {vocab.keys()}")
        
        # If char2idx exists, print some examples
        if 'char2idx' in vocab:
            char2idx = vocab['char2idx']
            logger.info(f"char2idx type: {type(char2idx)}, length: {len(char2idx)}")
            logger.info(f"Some character mappings:")
            for char, idx in list(char2idx.items())[:10]:
                logger.info(f"  '{char}' -> {idx}")
        
        # If idx2char exists, print some examples
        if 'idx2char' in vocab:
            idx2char = vocab['idx2char']
            logger.info(f"idx2char type: {type(idx2char)}, length: {len(idx2char)}")
            logger.info(f"Some index mappings:")
            items = list(idx2char.items())
            if items:
                for idx, char in items[:10]:
                    logger.info(f"  {idx} -> '{char}'")
        
    except Exception as e:
        logger.error(f"Error examining data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
