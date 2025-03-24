"""
Simple data processor module for ChatGoT.
This is a streamlined implementation for character-level tokenization.
"""

import os
import logging
import pickle
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def simple_process_data(cfg: DictConfig) -> int:
    """
    Process data based on configuration, using a simplified approach.
    
    Args:
        cfg: Configuration object
        
    Returns:
        0 for success, non-zero for failure
    """
    try:
        logger.info("Starting simple data processor")
        
        # Extract config parameters
        input_file = cfg.paths.data_file
        output_path = cfg.paths.processed_data
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Log configuration
        logger.info(f"Processing text from {input_file}")
        logger.info(f"Output will be saved to {output_path}")
        
        # Read the text
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        
        logger.info(f"Text length: {len(text)} characters")
        
        # Get processing options from config
        lowercase = cfg.data.processing.lowercase
        if lowercase:
            logger.info("Converting text to lowercase")
            text = text.lower()
        
        # Create character to index mapping
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        logger.info(f"Vocabulary size: {len(chars)} unique characters")
        
        # Convert text to indices
        indices = [char_to_idx[ch] for ch in text]
        
        # Create sequences for training
        seq_length = cfg.training.sequence_length
        stride = seq_length // 2  # 50% overlap
        sequences = []
        
        for i in range(0, len(indices) - seq_length, stride):
            # Convert to tensors for PyTorch
            input_seq = torch.tensor(indices[i:i + seq_length], dtype=torch.long)
            target_seq = torch.tensor(indices[i + 1:i + seq_length + 1], dtype=torch.long)
            sequences.append((input_seq, target_seq))
        
        logger.info(f"Created {len(sequences)} sequences of length {seq_length}")
        
        # Split into training and validation
        val_split = 1.0 - cfg.data.dataset.split_ratio
        val_size = int(len(sequences) * val_split)
        train_sequences = sequences[:-val_size] if val_size > 0 else sequences
        val_sequences = sequences[-val_size:] if val_size > 0 else []
        
        logger.info(f"Training sequences: {len(train_sequences)}")
        logger.info(f"Validation sequences: {len(val_sequences)}")
        
        # Package data
        data = {
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'sequence_length': seq_length,
            'metadata': {
                'original_file': input_file,
                'text_length': len(text),
                'vocab_size': len(char_to_idx),
                'settings': {
                    'lowercase': lowercase,
                    'sequence_length': seq_length,
                    'val_split': val_split
                }
            }
        }
        
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        # Save vocabulary 
        vocab_path = os.path.join(os.path.dirname(output_path), "vocab.json")
        logger.info(f"Saving vocabulary to {vocab_path}")
        
        # Saving as text since JSON is more human-readable
        with open(vocab_path, "w", encoding="utf-8") as f:
            import json
            vocab_data = {
                "char_to_idx": {str(k): v for k, v in char_to_idx.items()},
                "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
                "vocab_size": len(char_to_idx)
            }
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Data processing completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1 