import json
import matplotlib.pyplot as plt
import os
import argparse
import re # Import regex module

# --- Function to parse log and create plot ---
def plot_log_loss(log_file_path):
    """Parses a training log file and generates/saves a detailed loss plot."""

    # Determine plot save path based on log file path
    plot_save_path = os.path.join(os.path.dirname(log_file_path), "step_loss_plot.png")

    # Regex patterns to extract data
    step_loss_pattern = re.compile(r"Step: (\d+), Batch: \d+/\d+, Loss: (\d+\.\d+)")
    # Simplified epoch summary pattern focusing on Val Loss
    epoch_summary_pattern = re.compile(r"Epoch \d+/\d+ finished.*?Val Loss: (\d+\.\d+)")

    steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    current_step = 0 # Keep track of the step number for validation loss association

    try:
        # Read the log file
        with open(log_file_path, 'r') as f:
            for line in f:
                # Extract step-level training loss
                step_match = step_loss_pattern.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                    loss = float(step_match.group(2))
                    steps.append(current_step)
                    train_losses.append(loss)
                    continue # Move to next line once matched

                # Extract validation loss from epoch summary lines
                epoch_match = epoch_summary_pattern.search(line)
                if epoch_match:
                    val_loss = float(epoch_match.group(1))
                    # Associate validation loss with the last recorded step
                    if current_step > 0:
                         val_steps.append(current_step)
                         val_losses.append(val_loss)

        if not steps:
            print(f"Error: No step-level loss data found in {log_file_path}. Check log format.")
            return

        # Create the plot
        plt.figure(figsize=(12, 7))

        # Plot training loss
        plt.plot(steps, train_losses, linestyle='-', alpha=0.8, label='Training Loss (per step)')

        # Plot validation loss markers
        if val_steps:
            plt.scatter(val_steps, val_losses, marker='o', s=100, c='red', label='Validation Loss (end of epoch)', zorder=5) # zorder to keep markers on top

        # Add labels and title
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Step-Level Training and Epoch Validation Loss\n(Source: {os.path.basename(log_file_path)})")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log') # Often helpful for loss plots
        plt.tight_layout() # Adjust layout

        # Save the plot
        plt.savefig(plot_save_path)
        print(f"Plot saved successfully to: {plot_save_path}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- Main execution block ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Plot step-level training loss from a log file.")
    parser.add_argument("log_file", help="Path to the training log file (e.g., train_runner.log).")

    # Parse arguments
    args = parser.parse_args()

    # Call the plotting function
    plot_log_loss(args.log_file) 