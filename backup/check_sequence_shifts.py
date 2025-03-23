import pickle
import numpy as np

# Load the processed data
print("Loading data...")
data = pickle.load(open('processed_data/got_char_data.pkl', 'rb'))

# Get the first few samples
num_samples = 3
max_length = 50  # Only show this many characters per sample

for i in range(num_samples):
    # Get the input and target sequences
    input_seq = data['train_inputs'][i]
    target_seq = data['train_targets'][i]
    
    # Convert indices to characters
    input_text = ''.join([data['idx_to_char'][idx] for idx in input_seq[:max_length]])
    target_text = ''.join([data['idx_to_char'][idx] for idx in target_seq[:max_length]])
    
    # Print the results
    print(f"\nSample {i+1}:")
    print(f"Input:  {repr(input_text)}")
    print(f"Target: {repr(target_text)}")
    
    # Check if target is shifted by 1 (next character prediction)
    is_shifted = True
    for j in range(len(input_seq) - 1):
        if input_seq[j+1] != target_seq[j]:
            is_shifted = False
            break
    
    if is_shifted:
        print("✓ Target is a shift of input (next character prediction)")
    else:
        print("✗ Target is NOT a simple shift of input")
        
    # Show the character mappings
    print("Character mapping (first 10 positions):")
    for j in range(10):
        input_char = data['idx_to_char'][input_seq[j]]
        target_char = data['idx_to_char'][target_seq[j]]
        print(f"  Position {j}: Input '{input_char}' → Target '{target_char}'") 