#!/usr/bin/env python
"""
Log file encoding fixer

This script helps fix encoding issues in log files by:
1. Detecting the encoding
2. Converting to UTF-8
3. Removing or replacing special control characters

Usage:
    python scripts/fix_log_encoding.py <log_file_path>
"""

import sys
import os
import re
import chardet
import argparse

def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        # Read first 10000 bytes for detection
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
    return result['encoding'], result['confidence']

def clean_ansi_escape_sequences(text):
    """Remove ANSI escape sequences from text."""
    # Pattern to match ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def fix_log_file(input_path, output_path=None, remove_ansi=True):
    """Fix encoding issues in a log file."""
    if output_path is None:
        # Create output filename based on input
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fixed{ext}"
    
    # Detect encoding
    encoding, confidence = detect_encoding(input_path)
    print(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
    
    try:
        # Read the file with detected encoding
        with open(input_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # Clean up ANSI escape sequences if requested
        if remove_ansi:
            content = clean_ansi_escape_sequences(content)
        
        # Replace other problematic characters
        content = content.replace('\u0000', '')  # Remove null bytes
        
        # Write with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed log saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error fixing log file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix encoding issues in log files")
    parser.add_argument("log_file", help="Path to the log file to fix")
    parser.add_argument("--output", "-o", help="Output file path (default: input_fixed.ext)")
    parser.add_argument("--keep-ansi", action="store_true", help="Keep ANSI escape sequences")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: File not found: {args.log_file}")
        return 1
    
    success = fix_log_file(args.log_file, args.output, not args.keep_ansi)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 