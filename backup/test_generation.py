import torch
import pickle
import os
import numpy as np
from model import CharTransformer

# Load the data to get character mappings
print("Loading data...")
data_path = 'processed_data/got_char_data.pkl'
data = pickle.load(open(data_path, 'rb'))

char_to_idx = data['char_to_idx']
idx_to_char = data['idx_to_char']
vocab_size = len(char_to_idx)

print(f"Vocabulary size: {vocab_size}")

# Model parameters (match the ones used for training)
d_model = 384
n_head = 6
d_hid = 1536
nlayers = 8
dropout = 0.2

# Create the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CharTransformer(
    vocab_size=vocab_size, 
    d_model=d_model, 
    n_head=n_head, 
    num_layers=nlayers,
    dim_feedforward=d_hid, 
    dropout=dropout
).to(device)

# Load the best model
checkpoint_path = 'checkpoints/model_best.pt'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded, best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")
    exit(1)

# Set model to evaluation mode
model.eval()

# Function to generate text
def generate_text(model, seed_text, max_length=300, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
    print(f"\nGenerating text with seed: '{seed_text}'")
    print(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}")
    
    # Convert seed text to indices
    input_indices = []
    for char in seed_text:
        if char in char_to_idx:
            input_indices.append(char_to_idx[char])
        else:
            input_indices.append(char_to_idx['<unk>'])
    
    # Convert to tensor
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = seed_text
    
    # Generate characters
    with torch.no_grad():
        for i in range(max_length):
            # Get the model's output logits
            output = model(input_tensor)
            
            # Take the last time step
            next_token_logits = output[0, -1, :].cpu()
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for id in set(input_indices):
                    next_token_logits[id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=(len(next_token_logits) - top_k))[1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Convert to probabilities
            probabilities = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the probability distribution
            next_token_id = torch.multinomial(probabilities, 1).item()
            
            # Debug: Print the top 5 tokens and their probabilities
            top_probs, top_indices = torch.topk(probabilities, k=5)
            print(f"Step {i+1}, top tokens:")
            for j in range(5):
                char_rep = idx_to_char[top_indices[j].item()]
                if char_rep in ['\n', '\t', ' ']:
                    char_rep = repr(char_rep)
                print(f"  {char_rep}: {top_probs[j].item():.4f}")
            
            # Get the character for the selected token
            next_char = idx_to_char[next_token_id]
            
            # Append to the generated text
            generated_text += next_char
            print(f"Selected: '{next_char if next_char not in ['\n', '\t', ' '] else repr(next_char)}', Current text: '{generated_text[-20:]}'")
            
            # Update input for next iteration
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
            if input_tensor.size(1) > 256:  # truncate if too long
                input_tensor = input_tensor[:, -256:]
    
    return generated_text

# Test the model with different seeds
seed_texts = ["TYRION: ", "CERSEI: ", "ARYA: ", "JON: ", "DAENERYS: "]

for seed_text in seed_texts:
    print("\n" + "="*80)
    generated_text = generate_text(
        model, 
        seed_text, 
        max_length=100,  # Shorter for testing
        temperature=1.0,  # Try different temperatures
        top_k=10,        # Limit to top 10 tokens
        repetition_penalty=1.2  # Slight penalty for repetition
    )
    print("\nGenerated text:")
    print(generated_text)
    print("="*80) 