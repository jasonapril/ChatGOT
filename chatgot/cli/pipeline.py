"""CLI module for running the ChatGoT pipeline."""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from chatgot.core.config import get_config_path, load_config
from chatgot.pipeline.runner import run_pipeline
from chatgot.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the ChatGoT pipeline")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="default",
        help="Configuration name (default: 'default')"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Configuration directory path (default: project configs directory)"
    )
    
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=None,
        choices=["process", "optimize", "train", "generate", "all"],
        help="Pipeline stages to run (default: use config)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart of pipeline (ignore checkpoints)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args(args)


def run_cli(args: Optional[List[str]] = None) -> int:
    """
    Run the ChatGoT pipeline from the command line.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parse_args(args)
    
    # Set up logging
    setup_logging(level=parsed_args.log_level)
    
    try:
        # Load configuration
        config_name = parsed_args.config
        config_dir = parsed_args.config_dir
        
        logger.info(f"Loading configuration: {config_name}")
        
        # Prepare overrides
        overrides = []
        
        if parsed_args.stages:
            if "all" in parsed_args.stages:
                # Enable all stages
                stages = ["process", "optimize", "train", "generate"]
                overrides.append(f"pipeline.stages={stages}")
            else:
                # Enable only specified stages
                overrides.append(f"pipeline.stages={parsed_args.stages}")
        
        if parsed_args.resume:
            overrides.append("pipeline.resume=true")
        
        if parsed_args.force_restart:
            overrides.append("pipeline.force_restart=true")
        
        if parsed_args.seed is not None:
            overrides.append(f"system.seed={parsed_args.seed}")
        
        # Load configuration with overrides
        cfg = load_config(config_name, overrides)
        
        # Print effective configuration
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Effective configuration:\n{OmegaConf.to_yaml(cfg)}")
        
        # Run pipeline
        return run_pipeline(cfg)
    
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(run_cli()) 