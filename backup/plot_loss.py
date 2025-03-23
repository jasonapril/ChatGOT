import os
import pickle
import argparse
import matplotlib.pyplot as plt

def plot_training_loss(train_losses, val_losses, filename='training_loss.png'):
    """Plot and save training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss from saved data")
    parser.add_argument("--loss_file", type=str, default="checkpoints/loss_history.pkl", 
                        help="Path to loss history file")
    parser.add_argument("--output", type=str, default="training_loss.png", 
                        help="Output filename for the plot")
    args = parser.parse_args()
    
    # Load loss data
    print(f"Loading loss data from {args.loss_file}")
    with open(args.loss_file, 'rb') as f:
        loss_data = pickle.load(f)
    
    train_losses = loss_data['train_losses']
    val_losses = loss_data['val_losses']
    
    print(f"Loaded data for {len(train_losses)} epochs")
    
    # Create and save the plot
    plot_training_loss(train_losses, val_losses, args.output)
    
    # Print some statistics
    print("\nLoss statistics:")
    print(f"Initial training loss: {train_losses[0]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Initial validation loss: {val_losses[0]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    # Check for improvement
    if val_losses[-1] < val_losses[0]:
        improvement = (1 - val_losses[-1] / val_losses[0]) * 100
        print(f"\nModel improved by {improvement:.2f}% on validation set")
    else:
        print("\nWarning: Validation loss did not improve")

if __name__ == "__main__":
    main() 