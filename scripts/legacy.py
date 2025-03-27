#!/usr/bin/env python
"""
Training Pipeline

This script provides a complete pipeline, handling:
1. Data processing from raw text files
2. Optimization for the current environment (batch size, CUDA settings)
3. Model training with checkpoints
4. Text generation/sampling

The pipeline tracks its progress, allowing you to resume from any point.

Usage:
    python pipeline.py --input_file data/game_of_thrones_dataset.txt
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import torch
from typing import Dict, Any, Optional

# Local imports
from src.logger import setup_logger, log_section_header, force_flush_logs
from src.utils import setup_device, set_seed, create_output_dir
from src.data_processor import process_raw_data  # We'll need to ensure this exists

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Complete pipeline for processing data, optimizing, and training the model."
    )
    
    # Pipeline control arguments
    parser.add_argument("--pipeline_dir", type=str, default="pipeline",
                        help="Directory to store pipeline state and outputs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume pipeline from last completed stage.")
    parser.add_argument("--force_restart", action="store_true", 
                        help="Force restart the pipeline from the beginning.")
    parser.add_argument("--stage", type=str, choices=["process", "optimize", "train", "generate"],
                        help="Start pipeline from a specific stage.")
    
    # Data processing arguments
    parser.add_argument("--input_file", type=str, 
                        help="Path to the raw input text file.")
    parser.add_argument("--processed_data_path", type=str, default="processed_data/got_char_data.pkl",
                        help="Path to save/load processed data.")
    parser.add_argument("--sequence_length", type=int, default=1024,
                        help="Maximum sequence length for training data.")
    parser.add_argument("--skip_process", action="store_true",
                        help="Skip the data processing stage and use existing processed data.")
    
    # Optimization arguments
    parser.add_argument("--skip_optimization", action="store_true",
                        help="Skip the optimization stage and use default or provided values.")
    parser.add_argument("--test_batches", type=int, default=5,
                        help="Number of batches to test during optimization.")
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Minimum batch size to try during optimization.")
    parser.add_argument("--max_batch_size", type=int, default=256,
                        help="Maximum batch size to try during optimization.")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (default: determined by optimization)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of update steps to accumulate gradients for")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision for faster training")
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Use torch.compile to optimize model (requires PyTorch 2.0+)")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="Compilation mode for torch.compile")
    parser.add_argument("--force_aggressive_memory", action="store_true",
                        help="Use aggressive memory optimization techniques")
    
    # Generation arguments
    parser.add_argument("--generate_length", type=int, default=500,
                        help="Length of generated text.")
    parser.add_argument("--generate_seed", type=str, default="TYRION: ",
                        help="Seed text for generation.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for text generation.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of text samples to generate.")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if CUDA is available.")
    
    # Model configuration arguments (passed to training stage)
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model embedding dimension.")
    parser.add_argument("--n_head", type=int, default=12,
                        help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12,
                        help="Number of transformer layers.")
    parser.add_argument("--d_hid", type=int, default=3072,
                        help="Hidden dimension of feedforward layers.")
    
    return parser.parse_args()

class Pipeline:
    """Pipeline manager for the complete training process."""
    
    STAGES = ["process", "optimize", "train", "generate"]
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the pipeline with command line arguments.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.pipeline_dir = args.pipeline_dir
        os.makedirs(self.pipeline_dir, exist_ok=True)
        
        # Set up logging
        self.log_file = os.path.join(self.pipeline_dir, "pipeline.log")
        setup_logger(self.log_file, args.log_level)
        
        # Set random seed
        set_seed(args.seed)
        
        # Set up device
        self.device, self.is_cuda, _ = setup_device(args.force_cpu)
        
        # Initialize pipeline state
        self.state_file = os.path.join(self.pipeline_dir, "state.json")
        self.state = self._load_state()
        
        # Create stage-specific directories
        for stage in self.STAGES:
            os.makedirs(os.path.join(self.pipeline_dir, stage), exist_ok=True)
    
    def _load_state(self) -> Dict[str, Any]:
        """
        Load pipeline state from state file, or initialize if not exists.
        
        Returns:
            Pipeline state dictionary
        """
        if os.path.exists(self.state_file) and not self.args.force_restart:
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logging.info(f"Loaded pipeline state: {state}")
                return state
            except Exception as e:
                logging.warning(f"Failed to load state file: {e}")
        
        # Initialize fresh state
        state = {
            "last_completed_stage": None,
            "start_time": time.time(),
            "stages": {
                "process": {"completed": False, "timestamp": None, "output_path": None},
                "optimize": {"completed": False, "timestamp": None, "settings": {}},
                "train": {"completed": False, "timestamp": None, "checkpoint_path": None, 
                         "best_validation_loss": float('inf')},
                "generate": {"completed": False, "timestamp": None, "output_file": None}
            }
        }
        
        self._save_state(state)
        return state
    
    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """
        Save pipeline state to disk.
        
        Args:
            state: State to save, uses self.state if None
        """
        if state is not None:
            self.state = state
            
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _get_start_stage(self) -> str:
        """
        Determine which stage to start from based on arguments and state.
        
        Returns:
            Stage name to start from
        """
        if self.args.force_restart:
            return self.STAGES[0]
            
        if self.args.stage:
            return self.args.stage
            
        if self.args.resume and self.state["last_completed_stage"] is not None:
            # Find the next stage after the last completed one
            last_idx = self.STAGES.index(self.state["last_completed_stage"])
            if last_idx < len(self.STAGES) - 1:
                return self.STAGES[last_idx + 1]
            else:
                logging.info("Pipeline already completed. Running final stage again.")
                return self.STAGES[-1]
        
        return self.STAGES[0]
    
    def _mark_stage_completed(self, stage: str, **kwargs) -> None:
        """
        Mark a stage as completed in the pipeline state.
        
        Args:
            stage: Stage name
            kwargs: Additional data to store with the stage
        """
        self.state["stages"][stage]["completed"] = True
        self.state["stages"][stage]["timestamp"] = time.time()
        self.state["last_completed_stage"] = stage
        
        # Store any additional data
        for key, value in kwargs.items():
            self.state["stages"][stage][key] = value
            
        self._save_state()
    
    def run(self) -> None:
        """Run the pipeline from the appropriate starting point."""
        start_stage = self._get_start_stage()
        logging.info(f"Starting pipeline from stage: {start_stage}")
        
        # Determine which stages to run
        start_idx = self.STAGES.index(start_stage)
        stages_to_run = self.STAGES[start_idx:]
        
        for stage in stages_to_run:
            stage_method = getattr(self, f"run_{stage}")
            log_section_header(f"RUNNING STAGE: {stage.upper()}")
            
            try:
                stage_method()
                self._mark_stage_completed(stage)
                logging.info(f"Completed stage: {stage}")
            except Exception as e:
                logging.exception(f"Error in stage {stage}: {e}")
                sys.exit(1)
        
        log_section_header("PIPELINE COMPLETED")
        logging.info("All pipeline stages completed successfully!")
    
    def run_process(self) -> None:
        """
        Run the data processing stage.
        Converts raw text data into tokenized and formatted data for training.
        """
        if self.args.skip_process:
            if self.args.processed_data_path and os.path.exists(self.args.processed_data_path):
                logging.info("Skipping data processing stage as requested.")
                self.state["stages"]["process"]["output_path"] = self.args.processed_data_path
                return
            else:
                logging.warning("--skip_process flag set but processed data not found. Will process data.")
            
        if self.state["stages"]["process"]["completed"] and not self.args.force_restart:
            if self.args.processed_data_path and os.path.exists(self.args.processed_data_path):
                logging.info("Data already processed. Skipping processing stage.")
                return
            
        if not self.args.input_file:
            if self.args.processed_data_path and os.path.exists(self.args.processed_data_path):
                logging.info("No input file specified, but processed data exists. Skipping processing stage.")
                return
            else:
                raise ValueError("Input file required for data processing stage.")
        
        logging.info(f"Processing raw data from: {self.args.input_file}")
        output_path = self.args.processed_data_path
        
        # Import and run data processor
        from src.data_processor import process_raw_data
        process_raw_data(
            input_file=self.args.input_file,
            output_path=output_path,
            sequence_length=self.args.sequence_length
        )
        
        logging.info(f"Data processing completed. Processed data saved to: {output_path}")
        self.state["stages"]["process"]["output_path"] = output_path
        
    def run_optimize(self) -> None:
        """
        Run the optimization stage.
        Find optimal batch size and CUDA settings for the current environment.
        """
        if self.args.skip_optimization:
            logging.info("Optimization stage skipped as requested.")
            
            # Set default values if not provided
            settings = {
                "batch_size": self.args.batch_size or 32,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps or 1,
                "use_amp": self.args.use_amp,
                "cuda_optimized": False
            }
            
            self.state["stages"]["optimize"]["settings"] = settings
            return
            
        if self.state["stages"]["optimize"]["completed"] and not self.args.force_restart:
            logging.info("Optimization already completed. Using stored settings.")
            return
        
        # Ensure we have processed data
        processed_data_path = self.args.processed_data_path
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at: {processed_data_path}")
        
        logging.info("Finding optimal training settings...")
        
        # Apply CUDA optimizations first
        if self.is_cuda:
            from src.cuda_optimizations import apply_all_cuda_optimizations
            cuda_results = apply_all_cuda_optimizations()
            logging.info(f"CUDA optimization results: {cuda_results}")
        
        # Find optimal batch size
        if self.is_cuda and self.args.batch_size is None:
            from src.batch_size_finder import find_optimal_batch_size
            
            # Create args for batch size finder
            bsf_args = argparse.Namespace()
            bsf_args.data_path = processed_data_path
            bsf_args.sequence_length = self.args.sequence_length
            bsf_args.min_batch = self.args.min_batch_size
            bsf_args.max_batch = self.args.max_batch_size
            bsf_args.test_batches = self.args.test_batches
            bsf_args.force_cpu = self.args.force_cpu
            
            optimal_batch_size = find_optimal_batch_size(bsf_args)
            logging.info(f"Found optimal batch size: {optimal_batch_size}")
        else:
            optimal_batch_size = self.args.batch_size or 32
            logging.info(f"Using specified batch size: {optimal_batch_size}")
        
        # Determine gradient accumulation steps based on GPU
        from src.memory_management import get_memory_optimized_settings
        if self.is_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            optimized_settings = get_memory_optimized_settings(
                gpu_name, 
                self.args.force_aggressive_memory
            )
            
            grad_accum_steps = self.args.gradient_accumulation_steps or optimized_settings.get('gradient_accumulation_steps', 1)
        else:
            grad_accum_steps = self.args.gradient_accumulation_steps or 1
        
        # Store optimized settings
        settings = {
            "batch_size": optimal_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "use_amp": self.args.use_amp,
            "cuda_optimized": self.is_cuda
        }
        
        self.state["stages"]["optimize"]["settings"] = settings
        logging.info(f"Optimization completed with settings: {settings}")
    
    def run_train(self) -> None:
        """
        Run the training stage.
        Train the model using optimized parameters and save checkpoints.
        """
        # Ensure we have processed data
        processed_data_path = self.args.processed_data_path
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at: {processed_data_path}")
        
        # Get optimized settings
        opt_settings = self.state["stages"]["optimize"]["settings"]
        
        # Set up training arguments
        train_args = [
            "--data_path", processed_data_path,
            "--batch_size", str(opt_settings["batch_size"]),
            "--epochs", str(self.args.epochs),
            "--learning_rate", str(self.args.learning_rate),
            "--gradient_accumulation_steps", str(opt_settings["gradient_accumulation_steps"]),
            "--save_every", str(self.args.save_every),
            "--seed", str(self.args.seed),
            "--log_level", self.args.log_level
        ]
        
        # Add optional flags
        if opt_settings["use_amp"] or self.args.use_amp:
            train_args.append("--use_amp")
            
        if self.args.use_torch_compile:
            train_args.append("--use_torch_compile")
            train_args.append("--compile_mode")
            train_args.append(self.args.compile_mode)
            
        if self.args.force_aggressive_memory:
            train_args.append("--force_aggressive_memory")
            
        if self.args.force_cpu:
            train_args.append("--force_cpu")
        
        # Resume from checkpoint if available
        checkpoint_path = self.state["stages"]["train"].get("checkpoint_path")
        if checkpoint_path and os.path.exists(checkpoint_path):
            train_args.extend(["--resume_from", checkpoint_path])
            
        # Convert arguments to command string
        train_cmd = " ".join(train_args)
        logging.info(f"Running training with command: {train_cmd}")
        
        # Run training
        import subprocess
        process = subprocess.run(train_cmd, shell=True, check=True)
        
        # Find the best checkpoint
        checkpoint_dir = os.path.join(self.pipeline_dir, "train", "checkpoints")
        best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
        
        if os.path.exists(best_checkpoint):
            logging.info(f"Training completed. Best checkpoint: {best_checkpoint}")
            self.state["stages"]["train"]["checkpoint_path"] = best_checkpoint
        else:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
                logging.info(f"Training completed. Latest checkpoint: {latest_checkpoint}")
                self.state["stages"]["train"]["checkpoint_path"] = latest_checkpoint
    
    def run_generate(self) -> None:
        """
        Run the text generation stage.
        Generate text samples using the trained model.
        """
        # Ensure we have a trained model
        checkpoint_path = self.state["stages"]["train"].get("checkpoint_path")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No trained model checkpoint found.")
        
        # Import generation function
        from src.trainer import generate_text
        
        # Load model and tokenizer data
        import torch
        from src.model import create_transformer_model
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        char_to_idx = checkpoint.get('char_to_idx', {})
        idx_to_char = checkpoint.get('idx_to_char', {})
        
        # Recreate model with same architecture
        model_args = checkpoint.get('args', {})
        model = create_transformer_model(
            vocab_size=len(char_to_idx),
            max_seq_length=self.args.sequence_length,
            d_model=model_args.get('d_model', self.args.d_model),
            n_head=model_args.get('n_head', self.args.n_head),
            d_hid=model_args.get('d_hid', self.args.d_hid),
            n_layers=model_args.get('n_layers', self.args.n_layers),
            dropout=model_args.get('dropout', 0.2),
            memory_efficient=True
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Generate samples
        samples_dir = os.path.join(self.pipeline_dir, "generate", "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        # Generate multiple samples
        for i in range(self.args.num_samples):
            logging.info(f"Generating sample {i+1}/{self.args.num_samples}...")
            
            generated_text = generate_text(
                model=model,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                seed_text=self.args.generate_seed,
                max_length=self.args.generate_length,
                temperature=self.args.temperature,
                device=self.device
            )
            
            # Save sample
            sample_file = os.path.join(samples_dir, f"sample_{i+1}.txt")
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            
            logging.info(f"Sample saved to: {sample_file}")
        
        # Create consolidated samples file
        all_samples_file = os.path.join(self.pipeline_dir, "generate", "all_samples.txt")
        with open(all_samples_file, 'w', encoding='utf-8') as f:
            f.write(f"GENERATED SAMPLES (Temperature: {self.args.temperature})\n")
            f.write(f"Seed text: '{self.args.generate_seed}'\n\n")
            
            for i in range(self.args.num_samples):
                sample_file = os.path.join(samples_dir, f"sample_{i+1}.txt")
                with open(sample_file, 'r', encoding='utf-8') as sf:
                    sample_text = sf.read()
                
                f.write(f"=== SAMPLE {i+1} ===\n")
                f.write(sample_text)
                f.write("\n\n")
        
        logging.info(f"All samples consolidated to: {all_samples_file}")
        self.state["stages"]["generate"]["output_file"] = all_samples_file

def main():
    """Main function to run the pipeline."""
    args = parse_args()
    pipeline = Pipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main() 