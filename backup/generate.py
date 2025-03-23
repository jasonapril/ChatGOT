import os
import argparse
import torch
import pickle
from torch.amp import autocast
from model import create_char_transformer, CharTransformer

def load_checkpoint(checkpoint_path, device):
    """Load a trained model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocabulary from checkpoint
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    vocab_size = len(char_to_idx)
    
    # Create model with the same configuration
    device_memory_gb = 4  # Default
    if torch.cuda.is_available():
        device_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
    model = create_char_transformer(
        vocab_size=vocab_size, 
        device_memory_gb=device_memory_gb
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, char_to_idx, idx_to_char

def generate_text(model, char_to_idx, idx_to_char, seed_text, max_length=1000, temperature=0.8, top_k=40, device="cuda"):
    """Generate text from the model using sampling with temperature."""
    print(f"Generating text with seed: '{seed_text}'")
    
    # Convert seed text to tensor
    chars = list(seed_text)
    input_ids = [char_to_idx.get(c, char_to_idx["<unk>"]) for c in chars]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    generated = list(seed_text)
    
    # Use the model's built-in generation if available
    if hasattr(model, 'generate'):
        with torch.no_grad():
            generated_ids = model.generate(
                input_tensor, 
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                verbose=(max_length <= 100)  # Only show verbose output for short generations
            )
        
        # Convert IDs back to characters
        generated_text = ''.join([idx_to_char[idx.item()] for idx in generated_ids[0][len(input_ids):]])
        
        return seed_text + generated_text
    
    # Fallback to manual generation for backward compatibility
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass with mixed precision
            with autocast('cuda', enabled=device=="cuda"):
                logits = model(input_tensor)
            
            # Get last time step logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Convert to character and append to result
            next_char = idx_to_char[next_token]
            generated.append(next_char)
            
            # Print progress
            if i > 0 and i % 100 == 0:
                print(f"Generated {i} characters...")
            
            # Update input tensor for next iteration
            input_tensor = torch.cat([
                input_tensor, 
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ], dim=1)
            
            # Truncate to prevent excessive memory usage
            if input_tensor.size(1) > 100:
                input_tensor = input_tensor[:, -100:]
    
    return ''.join(generated)

def interactive_generation(model, char_to_idx, idx_to_char, device="cuda"):
    """Run an interactive generation session."""
    print("\n" + "=" * 80)
    print(" Interactive Text Generation Mode ".center(80, "="))
    print("=" * 80)
    print("Type a prompt to start generation")
    print("Type 'exit' to quit")
    print("=" * 80 + "\n")
    
    while True:
        prompt = input("Enter prompt: ")
        if prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode.")
            break
            
        if not prompt:
            continue
            
        # Get generation parameters
        try:
            max_length = int(input("Max tokens to generate [200]: ") or "200")
            temperature = float(input("Temperature (0.5-1.5) [0.8]: ") or "0.8")
            top_k = int(input("Top-k (0 for no filtering) [40]: ") or "40")
        except ValueError:
            print("Invalid input. Using default values.")
            max_length = 200
            temperature = 0.8
            top_k = 40
            
        print("\nGenerating...\n")
        
        # Generate text
        generated_text = generate_text(
            model, 
            char_to_idx, 
            idx_to_char, 
            prompt, 
            max_length=max_length, 
            temperature=temperature,
            top_k=top_k,
            device=device
        )
        
        # Print result
        print("\nGenerated Text:")
        print("-" * 80)
        print(generated_text)
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained character-level language model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--seed_text", type=str, default="TYRION: ", 
                        help="Seed text to start generation")
    parser.add_argument("--max_length", type=int, default=1000, 
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Output file to save generated text (optional)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    args = parser.parse_args()
    
    # Configure device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model, char_to_idx, idx_to_char = load_checkpoint(args.checkpoint, device)
    
    # Interactive mode
    if args.interactive:
        interactive_generation(model, char_to_idx, idx_to_char, device)
        return
    
    # Generate text
    generated_text = generate_text(
        model,
        char_to_idx,
        idx_to_char,
        args.seed_text,
        args.max_length,
        args.temperature,
        args.top_k,
        device
    )
    
    # Print generated text
    print("\nGenerated Text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"Generated text saved to {args.output_file}")

if __name__ == "__main__":
    main() 