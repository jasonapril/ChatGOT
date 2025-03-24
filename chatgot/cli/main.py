#!/usr/bin/env python
"""
ChatGoT CLI main entry point.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import os

# Completely removing coloredlogs
# try:
#     import coloredlogs
#     COLOREDLOGS_AVAILABLE = True
# except ImportError:
#     COLOREDLOGS_AVAILABLE = False

# Check hydra imports with more detailed error reporting
try:
    import hydra
    HYDRA_AVAILABLE = True
    print("Successfully imported hydra from:", hydra.__file__)
except ImportError as e:
    HYDRA_AVAILABLE = False
    print(f"Error importing hydra: {e}")

try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
    print("Successfully imported omegaconf")
except ImportError as e:
    OMEGACONF_AVAILABLE = False
    print(f"Error importing omegaconf: {e}")

# Check if both are available
if not (HYDRA_AVAILABLE and OMEGACONF_AVAILABLE):
    print("Error: hydra-core and omegaconf are required. Please install them using:")
    print("pip install hydra-core omegaconf")
    sys.exit(1)

from chatgot import __version__
from chatgot.cli import pipeline
# Import get_logger but not setup_logging to avoid conflict
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)

def local_setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def print_version() -> None:
    """Print version information."""
    print(f"ChatGoT version {__version__}")

def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="ChatGoT - Character-level GPT for Game of Thrones Text Generation"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to run",
        required=True
    )
    
    # Add version argument
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Print version information and exit"
    )
    
    # Add global arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    # Add train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model"
    )
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        default="default",
        help="Configuration name (default: 'default')"
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    # Add generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text using a trained model"
    )
    generate_parser.add_argument(
        "--config", "-c",
        type=str,
        default="default",
        help="Configuration name (default: 'default')"
    )
    generate_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    generate_parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt to start generation with"
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate"
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature"
    )
    generate_parser.add_argument(
        "--output",
        type=str,
        help="File to write generated text to"
    )
    
    # Add process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process text data"
    )
    process_parser.add_argument(
        "--config", "-c",
        type=str,
        default="default",
        help="Configuration name (default: 'default')"
    )
    
    # Add pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the full pipeline"
    )
    pipeline_parser.add_argument(
        "--config", "-c",
        type=str,
        default="default",
        help="Configuration name (default: 'default')"
    )
    pipeline_parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=["process", "optimize", "train", "generate", "all"],
        help="Pipeline stages to run (default: use config)"
    )
    pipeline_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    pipeline_parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart of pipeline (ignore checkpoints)"
    )
    
    return parser.parse_args(args)

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the ChatGoT CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parse_args(args)
    
    # Handle version flag
    if parsed_args.version:
        print_version()
        return 0
    
    # Set up logging - use local function
    local_setup_logging(log_level=parsed_args.log_level)
    
    try:
        # Dispatch to appropriate command
        if parsed_args.command == "pipeline":
            # Convert Namespace to list of args for the pipeline command
            pipeline_args = ["--config", parsed_args.config]
            
            if parsed_args.stages:
                pipeline_args.extend(["--stages"] + parsed_args.stages)
            
            if parsed_args.resume:
                pipeline_args.append("--resume")
            
            if parsed_args.force_restart:
                pipeline_args.append("--force-restart")
            
            if parsed_args.log_level:
                pipeline_args.extend(["--log-level", parsed_args.log_level])
            
            return pipeline.run_cli(pipeline_args)
        
        elif parsed_args.command == "train":
            # Create configuration overrides for the train command
            overrides = []
            
            if parsed_args.resume:
                overrides.append("pipeline.resume=true")
            
            # Set pipeline stages to just training
            overrides.append("pipeline.stages=[train]")
            
            # Load configuration
            with hydra.initialize_config_module(config_module="configs"):
                cfg = hydra.compose(config_name=parsed_args.config, overrides=overrides)
            
            # Run only the training stage
            from chatgot.training.trainer import train_model
            return train_model(cfg, resume=parsed_args.resume)
        
        elif parsed_args.command == "generate":
            # Create configuration overrides for the generate command
            overrides = []
            
            # Set pipeline stages to just generation
            overrides.append("pipeline.stages=[generate]")
            
            if parsed_args.checkpoint:
                overrides.append(f"pipeline.generate.checkpoint_path={parsed_args.checkpoint}")
            
            if parsed_args.prompt:
                overrides.append(f"pipeline.generate.prompt={parsed_args.prompt}")
            
            if parsed_args.max_tokens:
                overrides.append(f"pipeline.generate.max_new_tokens={parsed_args.max_tokens}")
            
            if parsed_args.temperature:
                overrides.append(f"pipeline.generate.temperature={parsed_args.temperature}")
            
            if parsed_args.output:
                overrides.append(f"pipeline.generate.output_file={parsed_args.output}")
            
            # Load configuration
            with hydra.initialize_config_module(config_module="configs"):
                cfg = hydra.compose(config_name=parsed_args.config, overrides=overrides)
            
            # Run just the generate stage
            from chatgot.models.generate import generate_text
            generate_text(cfg)
            return 0
        
        elif parsed_args.command == "process":
            # Create configuration overrides for the process command
            overrides = []
            
            # Set pipeline stages to just processing
            overrides.append("+pipeline.stages=[process]")
            
            # Load configuration
            with hydra.initialize_config_module(config_module="configs"):
                cfg = hydra.compose(config_name=parsed_args.config, overrides=overrides)
            
            # Debug - print out config values
            logger.info("Config loaded successfully. Values:")
            logger.info(f"Config has keys: {list(cfg.keys())}")
            if hasattr(cfg, 'paths'):
                logger.info(f"Paths config: {cfg.paths}")
            else:
                logger.info("Config does not have 'paths' section")
            
            # Run just the process stage
            try:
                import traceback
                
                # Enable debug logging
                import logging
                logging.getLogger().setLevel(logging.DEBUG)
                
                # Check if the Game of Thrones dataset exists
                data_path = cfg.paths.data_file
                full_data_path = os.path.join(os.getcwd(), data_path)
                logger.debug(f"Looking for data file at: {full_data_path}")
                
                if not os.path.exists(full_data_path):
                    logger.error(f"Game of Thrones dataset not found at {full_data_path}")
                    logger.error("Please download the dataset and place it in the specified location")
                    return 1
                
                logger.debug(f"Data file found at: {full_data_path}")
                
                # Create output directories if they don't exist
                processed_dir = os.path.dirname(cfg.paths.processed_data)
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir, exist_ok=True)
                    logger.debug(f"Created processed data directory: {processed_dir}")
                
                analysis_dir = cfg.paths.analysis_dir
                if not os.path.exists(analysis_dir):
                    os.makedirs(analysis_dir, exist_ok=True)
                    logger.debug(f"Created analysis directory: {analysis_dir}")
                
                logger.debug("Starting data processing")
                
                # Use the simple processor instead
                from chatgot.data.simple_processor import simple_process_data
                logger.info("Using simplified data processor")
                return simple_process_data(cfg)
                
            except Exception as e:
                logger.error(f"Error in process_data: {str(e)}")
                logger.error(traceback.format_exc())
                return 1
        
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

def entry_point() -> None:
    """Entry point for the ChatGoT CLI when installed as a package."""
    # Process version flag before Hydra takes over
    if "--version" in sys.argv:
        print_version()
        sys.exit(0)
        
    # Let Hydra handle the rest
    sys.exit(main())

if __name__ == "__main__":
    entry_point() 