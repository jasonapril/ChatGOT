"""Data processing module for preparing datasets."""
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig

from chatgot.core.config import get_full_path
from chatgot.data.dataset import load_text_from_file
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


def process_text(
    text: str,
    lowercase: bool = False,
    remove_special_chars: bool = False,
    min_frequency: int = 1,
) -> str:
    """
    Process text for character-level modeling.
    
    Args:
        text: Input text
        lowercase: Whether to convert text to lowercase
        remove_special_chars: Whether to remove special characters
        min_frequency: Minimum character frequency to keep
        
    Returns:
        Processed text
    """
    # Convert to lowercase if requested
    if lowercase:
        logger.info("Converting text to lowercase")
        text = text.lower()
    
    # Remove special characters if requested
    if remove_special_chars:
        logger.info("Removing special characters")
        text = re.sub(r'[^\w\s]', '', text)
    
    # Filter characters by frequency if min_frequency > 1
    if min_frequency > 1:
        logger.info(f"Filtering characters with frequency < {min_frequency}")
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1
        
        # Replace rare characters with space
        filtered_text = ""
        for c in text:
            if char_counts[c] >= min_frequency:
                filtered_text += c
            else:
                filtered_text += " "
        
        text = filtered_text
        
        logger.info(f"Character vocabulary size after filtering: {len(set(text))}")
    
    return text


def analyze_text(text: str) -> Dict[str, any]:
    """
    Analyze text and return statistics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of statistics
    """
    # Character counts
    char_counts = {}
    for c in text:
        char_counts[c] = char_counts.get(c, 0) + 1
    
    # Sort characters by frequency
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate statistics
    stats = {
        "total_chars": len(text),
        "unique_chars": len(char_counts),
        "most_common_chars": sorted_chars[:10],
        "least_common_chars": sorted_chars[-10:],
        "avg_char_frequency": len(text) / len(char_counts),
        "char_frequency_distribution": char_counts,
    }
    
    return stats


def process_data(
    cfg: DictConfig,
    save_analysis: bool = True,
) -> int:
    """
    Process data based on configuration.
    
    Args:
        cfg: Configuration
        save_analysis: Whether to save analysis to a file
        
    Returns:
        0 for success, non-zero for failure
    """
    try:
        import sys
        print(f"DEBUG: Starting process_data with cfg: {cfg}", file=sys.stderr)
        print(f"DEBUG: cfg keys: {list(cfg.keys())}", file=sys.stderr)
        print(f"DEBUG: cfg paths keys: {list(cfg.paths.keys()) if hasattr(cfg, 'paths') else 'No paths'}", file=sys.stderr)
        
        # Extract parameters from config
        data_cfg = cfg.data
        
        # Get file paths
        data_path = get_full_path(cfg.paths.data_file, cfg)
        print(f"DEBUG: data_path: {data_path}", file=sys.stderr)
        processed_cache_path = get_full_path(cfg.paths.processed_data, cfg)
        print(f"DEBUG: processed_cache_path: {processed_cache_path}", file=sys.stderr)
        
        logger.info(f"Processing text data from {data_path}")
        
        # Load text
        text = load_text_from_file(data_path)
        logger.info(f"Loaded text with {len(text)} characters")
        
        # Process text
        processed_text = process_text(
            text=text,
            lowercase=data_cfg.lowercase,
            remove_special_chars=data_cfg.remove_special_chars,
            min_frequency=data_cfg.min_frequency,
        )
        
        logger.info(f"Processed text has {len(processed_text)} characters")
        
        # Analyze text
        logger.info("Analyzing processed text")
        analysis = analyze_text(processed_text)
        
        logger.info(f"Text statistics:")
        logger.info(f"  - Total characters: {analysis['total_chars']}")
        logger.info(f"  - Unique characters: {analysis['unique_chars']}")
        logger.info(f"  - 10 most common characters: {analysis['most_common_chars'][:10]}")
        
        # Save processed data
        os.makedirs(os.path.dirname(processed_cache_path), exist_ok=True)
        
        processed_data = {
            "text": processed_text,
            "config": {
                "lowercase": data_cfg.lowercase,
                "remove_special_chars": data_cfg.remove_special_chars,
                "min_frequency": data_cfg.min_frequency,
            },
            "analysis": analysis,
        }
        
        logger.info(f"Saving processed data to {processed_cache_path}")
        with open(processed_cache_path, "wb") as f:
            pickle.dump(processed_data, f)
        
        # Save analysis if requested
        if save_analysis:
            analysis_path = get_full_path(cfg.paths.analysis_dir, cfg) / "text_analysis.pkl"
            os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
            
            logger.info(f"Saving text analysis to {analysis_path}")
            with open(analysis_path, "wb") as f:
                pickle.dump(analysis, f)
        
        logger.info("Data processing completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1 