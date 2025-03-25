"""
Pipeline Stages Module
=====================

This module implements the individual pipeline stages:

1. Process: Data preprocessing
2. Optimize: Model optimization
3. Train: Model training
4. Generate: Text generation

Each stage function takes a Pipeline instance and returns a dictionary 
of stage results to be stored in the pipeline state.
"""

import logging
import os
import time
from typing import Dict, Any

def run_process_stage(pipeline) -> Dict[str, Any]:
    """
    Run the data processing stage.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        Dictionary of stage results
    """
    logging.info("Starting data processing stage")
    
    # Import here to avoid circular imports
    from src.data_handler import DataHandler
    
    args = pipeline.args
    
    # Create a data handler
    data_handler = DataHandler(
        input_file=args.input_file,
        output_dir=os.path.join(pipeline.pipeline_dir, "process"),
        vocab_size=args.vocab_size,
        seq_length=args.seq_length
    )
    
    # Process the data
    train_file, val_file, test_file, vocab_file = data_handler.process()
    
    # Store the processed data paths
    output_data = {
        "output_path": os.path.join(pipeline.pipeline_dir, "process"),
        "train_file": train_file,
        "val_file": val_file,
        "test_file": test_file,
        "vocab_file": vocab_file,
        "vocab_size": data_handler.get_vocab_size()
    }
    
    return output_data

def run_optimize_stage(pipeline) -> Dict[str, Any]:
    """
    Run the optimization stage.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        Dictionary of stage results
    """
    logging.info("Starting optimization stage")
    
    # Import optimizer
    from src.optimizer import ModelOptimizer
    
    args = pipeline.args
    
    # Get data paths from process stage
    process_data = pipeline.state["stages"]["process"]
    if not process_data.get("completed", False):
        raise ValueError("Processing stage must be completed before optimization")
    
    train_file = process_data["train_file"]
    val_file = process_data["val_file"]
    vocab_size = process_data["vocab_size"]
    
    # Create optimizer
    optimizer = ModelOptimizer(
        train_file=train_file,
        val_file=val_file,
        vocab_size=vocab_size,
        device=pipeline.device
    )
    
    # Determine optimization parameters
    if args.auto_optimize:
        logging.info("Running auto-optimization")
        # Run auto optimization
        batch_size, learning_rate, hidden_size, num_layers = optimizer.auto_optimize(
            time_budget=args.optimization_time_budget
        )
    else:
        # Use provided parameters
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        
        logging.info(f"Using provided optimization parameters: batch_size={batch_size}, "
                    f"learning_rate={learning_rate}, hidden_size={hidden_size}, "
                    f"num_layers={num_layers}")
    
    # Store the optimization results
    settings = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "embedding_dim": args.embedding_dim,  # Not optimized
        "dropout": args.dropout              # Not optimized
    }
    
    # Save settings to a file
    settings_file = pipeline.save_artifact("model_settings.json", settings, "optimize")
    
    return {
        "settings": settings,
        "settings_file": settings_file
    }

def run_train_stage(pipeline) -> Dict[str, Any]:
    """
    Run the model training stage.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        Dictionary of stage results
    """
    logging.info("Starting model training stage")
    
    # Import trainer
    from src.trainer import Trainer
    from src.model import Model
    
    args = pipeline.args
    
    # Get data paths from process stage
    process_data = pipeline.state["stages"]["process"]
    if not process_data.get("completed", False):
        raise ValueError("Processing stage must be completed before training")
    
    # Get model settings from optimize stage
    optimize_data = pipeline.state["stages"]["optimize"]
    if not optimize_data.get("completed", False) and not args.skip_optimization:
        raise ValueError("Optimization stage must be completed before training (or use --skip-optimization)")
    
    # Get files from process stage
    train_file = process_data["train_file"]
    val_file = process_data["val_file"]
    vocab_file = process_data["vocab_file"]
    vocab_size = process_data["vocab_size"]
    
    # Get settings from optimize stage or use defaults
    if args.skip_optimization:
        settings = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout
        }
    else:
        settings = optimize_data["settings"]
    
    # Create model with optimized parameters
    model = Model(
        vocab_size=vocab_size,
        embedding_dim=settings["embedding_dim"],
        hidden_size=settings["hidden_size"],
        num_layers=settings["num_layers"],
        dropout=settings["dropout"]
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_file=train_file,
        val_file=val_file,
        vocab_file=vocab_file,
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        device=pipeline.device,
        checkpoint_dir=os.path.join(pipeline.pipeline_dir, "train"),
        num_epochs=args.num_epochs,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Train the model
    checkpoint_path, best_val_loss = trainer.train()
    
    return {
        "checkpoint_path": checkpoint_path,
        "best_validation_loss": best_val_loss,
        "epochs_completed": args.num_epochs,
        "training_time": trainer.total_training_time,
        "settings_used": settings
    }

def run_generate_stage(pipeline) -> Dict[str, Any]:
    """
    Run the text generation stage.
    
    Args:
        pipeline: Pipeline instance
        
    Returns:
        Dictionary of stage results
    """
    logging.info("Starting text generation stage")
    
    # Import generator
    from src.generator import TextGenerator
    
    args = pipeline.args
    
    # Get data paths from process stage
    process_data = pipeline.state["stages"]["process"]
    if not process_data.get("completed", False):
        raise ValueError("Processing stage must be completed before generating")
    
    # Get model checkpoint from train stage
    train_data = pipeline.state["stages"]["train"]
    if not train_data.get("completed", False):
        raise ValueError("Training stage must be completed before generating")
    
    checkpoint_path = train_data["checkpoint_path"]
    vocab_file = process_data["vocab_file"]
    
    # Create generator
    generator = TextGenerator(
        checkpoint_path=checkpoint_path,
        vocab_file=vocab_file,
        device=pipeline.device
    )
    
    # Generate text
    texts = []
    for i in range(args.num_samples):
        seed = args.seed_text if args.seed_text else None
        text = generator.generate(
            seed_text=seed,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        texts.append(text)
    
    # Save generated texts
    output_file = os.path.join(pipeline.pipeline_dir, "generate", "generated_texts.txt")
    with open(output_file, 'w') as f:
        for i, text in enumerate(texts):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(text)
            f.write("\n\n")
    
    return {
        "output_file": output_file,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "generated_texts": texts
    } 