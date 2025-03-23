import pickle
import numpy as np
import json
import os

# Load the processed data
print("Loading data...")
data = pickle.load(open('processed_data/got_char_data.pkl', 'rb'))

# Open a file to write the results
with open('data_inspection_results.txt', 'w', encoding='utf-8') as f:
    # Basic statistics
    f.write(f"Vocabulary size: {len(data['char_to_idx'])}\n")
    f.write(f"Training samples: {len(data['train_inputs'])}\n")
    f.write(f"Validation samples: {len(data['val_inputs'])}\n")
    f.write(f"Sequence length: {data['train_inputs'].shape[1]}\n\n")
    
    # Print vocabulary info
    f.write("Vocabulary characters:\n")
    for char, idx in sorted(data['char_to_idx'].items(), key=lambda x: x[1]):
        if char in ['\n', '\t', ' ']:
            char_repr = repr(char)
        else:
            char_repr = char
        f.write(f"{idx}: {char_repr}\n")
    
    # Print some sample sequences
    f.write("\nSample sequences from training data:\n")
    for i in range(min(5, len(data['train_inputs']))):
        sample = data['train_inputs'][i]
        target = data['train_targets'][i]
        
        # Convert indices to characters
        sample_text = ''.join([data['idx_to_char'][idx] for idx in sample])
        target_text = ''.join([data['idx_to_char'][idx] for idx in target])
        
        f.write(f"\nSample {i+1}:\n")
        f.write(f"Input: {repr(sample_text[:100])}...\n")
        f.write(f"Target: {repr(target_text[:100])}...\n")
    
    # Check for character frequency distribution
    f.write("\nCharacter frequency in training data:\n")
    char_counts = {}
    for sample in data['train_inputs']:
        for idx in sample:
            char = data['idx_to_char'][idx]
            char_counts[char] = char_counts.get(char, 0) + 1
    
    # Sort by frequency and print top 20
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        if char in ['\n', '\t', ' ']:
            char_repr = repr(char)
        else:
            char_repr = char
        f.write(f"{char_repr}: {count}\n")

print("Inspection complete. Results saved to data_inspection_results.txt") 