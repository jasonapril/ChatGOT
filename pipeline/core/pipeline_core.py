#!/usr/bin/env python
"""
Pipeline Core Module
==================

This module provides the core pipeline functionality, including:

1. Pipeline state management
2. Stage execution orchestration 
3. Error handling and recovery
4. Progress tracking
5. Configuration management

The Pipeline class coordinates the entire text generation workflow.
"""

import argparse
import json
import logging
import os
import time
import sys
from typing import Dict, Any, Optional, List, Callable

from src.logger import setup_logger, log_section_header, force_flush_logs
from src.utils.device import setup_device
from src.utils.model import set_seed

class Pipeline:
    """Pipeline manager for the complete ChatGoT training process."""
    
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
        
        # Skip stages if requested
        if self.args.skip_process and "process" in stages_to_run:
            logging.info("Skipping process stage as requested")
            stages_to_run.remove("process")
            
        if self.args.skip_optimization and "optimize" in stages_to_run:
            logging.info("Skipping optimization stage as requested")
            stages_to_run.remove("optimize")
        
        if not stages_to_run:
            logging.warning("No stages to run!")
            return
            
        # Import stage functions dynamically to avoid circular imports
        from pipeline.stages.pipeline_stages import (
            run_process_stage, 
            run_optimize_stage,
            run_train_stage,
            run_generate_stage
        )
        
        # Map stages to functions
        stage_funcs = {
            "process": run_process_stage,
            "optimize": run_optimize_stage,
            "train": run_train_stage,
            "generate": run_generate_stage
        }
        
        # Run each stage
        try:
            for stage in stages_to_run:
                log_section_header(f"STAGE: {stage.upper()}")
                
                stage_func = stage_funcs.get(stage)
                if not stage_func:
                    logging.error(f"No implementation found for stage: {stage}")
                    continue
                
                # Run the stage
                stage_start = time.time()
                stage_data = stage_func(self)
                stage_time = time.time() - stage_start
                
                # Mark stage as completed and store data
                self._mark_stage_completed(stage, **stage_data)
                
                logging.info(f"Completed stage '{stage}' in {stage_time:.2f} seconds")
                force_flush_logs()
                
        except KeyboardInterrupt:
            logging.warning("Pipeline interrupted by user!")
        except Exception as e:
            logging.error(f"Pipeline error: {e}", exc_info=True)
            raise
        
        # Calculate total time
        total_time = time.time() - self.state["start_time"]
        logging.info(f"Pipeline completed in {total_time:.2f} seconds")
    
    def get_stage_output(self, stage: str, key: str) -> Any:
        """
        Get output data from a previous stage.
        
        Args:
            stage: Stage name
            key: Output data key
            
        Returns:
            Stage output data or None if not found
        """
        stage_data = self.state["stages"].get(stage, {})
        return stage_data.get(key)
    
    def save_artifact(self, filename: str, data: Any, stage: str = None) -> str:
        """
        Save an artifact file for the current stage.
        
        Args:
            filename: Artifact filename
            data: Data to save
            stage: Stage name (uses current stage if None)
            
        Returns:
            Full path to the saved artifact
        """
        if stage is None:
            stage = self.state["last_completed_stage"] or "process"
            
        stage_dir = os.path.join(self.pipeline_dir, stage)
        filepath = os.path.join(stage_dir, filename)
        
        # Determine file type and save accordingly
        if isinstance(data, dict) or isinstance(data, list):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, str):
            with open(filepath, 'w') as f:
                f.write(data)
        else:
            # Use pickle for other data types
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        logging.info(f"Saved artifact to {filepath}")
        return filepath 