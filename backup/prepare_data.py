import os
import glob
import json
import numpy as np
from collections import Counter
import pickle

def load_got_data(data_dir="data"):
    """
    Load all Game of Thrones transcript files and concatenate them.
    """
    all_text = ""
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    print(f"Found {len(files)} files in {data_dir}")
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            all_text += text + "\n\n"
    
    print(f"Total text size: {len(all_text)} characters")
    return all_text

def create_char_vocab(text, min_freq=5):
    """
    Create a character-level vocabulary.
    """
    chars = Counter(text)
    print(f"Total unique characters: {len(chars)}")
    
    # Filter by frequency and sort
    vocab = {char: i+1 for i, (char, count) in enumerate(
        sorted(chars.items(), key=lambda x: -x[1])) 
        if count >= min_freq}
    
    # Add special tokens
    vocab = {"<pad>": 0, **vocab, "<unk>": len(vocab)+1}
    
    # Create inverse mapping
    idx_to_char = {i: char for char, i in vocab.items()}
    
    print(f"Vocabulary size after filtering: {len(vocab)}")
    return vocab, idx_to_char

def encode_text(text, char_to_idx):
    """
    Encode text using character vocabulary.
    """
    encoded = [char_to_idx.get(char, char_to_idx["<unk>"]) for char in text]
    return encoded

def prepare_training_data(text, char_to_idx, seq_length=256):
    """
    Prepare training data with overlapping sequences.
    """
    encoded_text = encode_text(text, char_to_idx)
    
    # Create sequences with sliding window
    sequences = []
    for i in range(0, len(encoded_text) - seq_length, seq_length // 2):
        seq = encoded_text[i:i+seq_length+1]  # +1 to include the target
        if len(seq) == seq_length + 1:
            sequences.append(seq)
    
    # Convert to numpy array
    data = np.array(sequences)
    print(f"Created {len(sequences)} sequences of length {seq_length+1}")
    
    # Split into inputs and targets
    inputs = data[:, :-1]
    targets = data[:, 1:]
    
    return inputs, targets

def main():
    # Load and process data
    text = load_got_data()
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_char_vocab(text)
    
    # Prepare training data
    inputs, targets = prepare_training_data(text, char_to_idx)
    
    # Calculate split indices (90% train, 10% val)
    split_idx = int(0.9 * len(inputs))
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    print(f"Train set: {len(train_inputs)} sequences")
    print(f"Validation set: {len(val_inputs)} sequences")
    
    # Save processed data
    os.makedirs("processed_data", exist_ok=True)
    
    data_dict = {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "val_inputs": val_inputs,
        "val_targets": val_targets,
    }
    
    with open("processed_data/got_char_data.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    
    # Also save vocabulary as JSON for easier inspection
    with open("processed_data/vocab.json", "w") as f:
        json.dump(char_to_idx, f, indent=2)
    
    print("Data processing complete. Files saved to 'processed_data/'")

if __name__ == "__main__":
    main() 