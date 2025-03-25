#!/usr/bin/env python
"""
Process Text Data Script

This script combines multiple text files into a single training file.
Can be used for any text dataset organized in multiple files.
"""

import os
import glob
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_text_data(input_dir, output_file, file_pattern="*.txt", sort_key=None, 
                     include_headers=True, header_prefix="## "):
    """
    Combine multiple text files into a single training file.
    
    Args:
        input_dir: Directory containing individual text files
        output_file: Path to the output combined file
        file_pattern: Glob pattern to match files (default: "*.txt")
        sort_key: Function to sort files (default: alphabetical)
        include_headers: Whether to include file names as section headers
        header_prefix: Prefix for section headers (default: "## ")
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all matching files
    input_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not input_files:
        logger.error(f"No files matching pattern '{file_pattern}' found in '{input_dir}'")
        return
    
    # Sort files if a sort key is provided
    if sort_key:
        input_files.sort(key=sort_key)
    else:
        input_files.sort()
    
    logger.info(f"Found {len(input_files)} files matching pattern '{file_pattern}'")
    
    # Initialize with a header for the dataset
    dataset_name = os.path.basename(os.path.normpath(input_dir))
    combined_content = f"# {dataset_name.capitalize()} Combined Text\n\n"
    
    # Process each file
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing file: {file_name}")
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add file header and content
        if include_headers:
            combined_content += f"\n{header_prefix}{file_name}\n\n"
        
        combined_content += content
        combined_content += "\n\n"
    
    # Write the combined content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    # Get file size in MB
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    logger.info(f"Successfully created combined file: {output_file}")
    logger.info(f"Total file size: {file_size_mb:.2f} MB")
    logger.info(f"Total character count: {len(combined_content)}")
    
    return combined_content

def got_sort_key(file_path):
    """Custom sort key for Game of Thrones episodes."""
    file_name = os.path.basename(file_path)
    if file_name.startswith('got_s') and 'e' in file_name:
        try:
            # Extract season and episode numbers
            season_part = file_name.split('_')[1]
            season = int(season_part[1:3])
            episode = int(season_part[4:6])
            return (season, episode)
        except (IndexError, ValueError):
            pass
    # Return a default tuple for non-matching files to ensure consistent comparison
    return (999, 999)

def main():
    parser = argparse.ArgumentParser(description="Process text data files into a combined file")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input text files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to the output combined file")
    parser.add_argument("--file_pattern", type=str, default="*.txt",
                       help="Glob pattern to match files (default: *.txt)")
    parser.add_argument("--dataset_type", type=str, default="generic",
                       choices=["generic", "got"],
                       help="Dataset type for specialized processing")
    parser.add_argument("--include_headers", action="store_true", default=True,
                       help="Include file names as section headers")
    parser.add_argument("--header_prefix", type=str, default="## ",
                       help="Prefix for section headers")
    
    args = parser.parse_args()
    
    # Select sort key based on dataset type
    sort_key = None
    if args.dataset_type == "got":
        sort_key = got_sort_key
    
    process_text_data(
        args.input_dir, 
        args.output_file,
        args.file_pattern,
        sort_key,
        args.include_headers,
        args.header_prefix
    )

if __name__ == "__main__":
    main() 