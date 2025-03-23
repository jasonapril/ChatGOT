#!/usr/bin/env python
# Run the complete character-level language model pipeline
# This script runs data preparation, training, and text generation in sequence

import os
import time
import argparse
import subprocess
import shutil
import sys
import datetime

def run_command(cmd, description):
    """Run a command with error handling and timing."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running command: {cmd}")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"\nCommand completed successfully in {elapsed:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete character-level language model pipeline")
    
    # Main arguments
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing the Game of Thrones transcript files")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the final model and results")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip data preparation (use existing processed data)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--skip_generate", action="store_true",
                        help="Skip text generation")
    parser.add_argument("--auto_tune", action="store_true",
                        help="Auto-tune batch size based on GPU memory")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"GAME OF THRONES CHARACTER-LEVEL LANGUAGE MODEL PIPELINE")
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    pipeline_success = True
    
    # Step 1: Data Preparation
    if not args.skip_prepare:
        prepare_cmd = f"python prepare_data.py --input_dir {args.data_dir} --output_file processed_data/got_char_data.pkl"
        pipeline_success = run_command(prepare_cmd, "Data Preparation") and pipeline_success
    else:
        print("\nSkipping data preparation as requested")
    
    # Step 2: Training
    if not args.skip_train and pipeline_success:
        train_cmd = f"python train.py --data_path processed_data/got_char_data.pkl --epochs {args.epochs} --checkpoint_dir checkpoints --save_loss_plot"
        
        if args.auto_tune:
            train_cmd += " --auto_tune"
            
        pipeline_success = run_command(train_cmd, "Model Training") and pipeline_success
    else:
        print("\nSkipping training as requested or due to previous failure")
    
    # Step 3: Text Generation
    if not args.skip_generate and pipeline_success:
        generate_cmd = "python generate.py --checkpoint checkpoints/model_best.pt --seed_text \"TYRION: \" --temperature 0.8 --max_length 500"
        pipeline_success = run_command(generate_cmd, "Text Generation (Tyrion)") and pipeline_success
        
        generate_cmd = "python generate.py --checkpoint checkpoints/model_best.pt --seed_text \"DAENERYS: \" --temperature 1.0 --max_length 500"
        pipeline_success = run_command(generate_cmd, "Text Generation (Daenerys)") and pipeline_success
    else:
        print("\nSkipping text generation as requested or due to previous failure")
    
    # Copy results to output directory
    print(f"\n{'='*80}")
    print(f"Copying results to {args.output_dir}")
    
    if os.path.exists("checkpoints/model_best.pt"):
        shutil.copy("checkpoints/model_best.pt", os.path.join(args.output_dir, "best_model.pt"))
        print(f"Copied best model to {args.output_dir}/best_model.pt")
        
    if os.path.exists("checkpoints/training_loss.png"):
        shutil.copy("checkpoints/training_loss.png", os.path.join(args.output_dir, "training_loss.png"))
        print(f"Copied loss plot to {args.output_dir}/training_loss.png")
        
    if os.path.exists("processed_data/vocab.json"):
        shutil.copy("processed_data/vocab.json", os.path.join(args.output_dir, "vocab.json"))
        print(f"Copied vocabulary to {args.output_dir}/vocab.json")
    
    # Final message
    if pipeline_success:
        print(f"\n{'='*80}")
        print(f"Pipeline completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to {args.output_dir}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"Pipeline completed with errors at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Check the logs for more information")
        print(f"{'='*80}")
    
    return 0 if pipeline_success else 1

if __name__ == "__main__":
    sys.exit(main()) 