#!/usr/bin/env python
"""
Training Progress Monitor
This script monitors the training log file and provides real-time updates.
Run this script in a separate terminal window while training is in progress.
"""

import os
import time
import sys
import argparse
from datetime import datetime
import re

def get_terminal_width():
    """Get the width of the terminal window."""
    try:
        return os.get_terminal_size().columns
    except (AttributeError, OSError):
        return 80

def format_time_elapsed(seconds):
    """Format seconds into a readable time string."""
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} h"

def tail_file(filename, n=10):
    """Return the last n lines of a file."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    except FileNotFoundError:
        return []

def extract_progress(line):
    """Extract progress percentage from a line."""
    match = re.search(r'(\d+\.\d+)%', line)
    if match:
        return float(match.group(1))
    return None

def extract_loss(line):
    """Extract loss value from a line."""
    match = re.search(r'Loss:\s+(\d+\.\d+)', line)
    if match:
        return float(match.group(1))
    return None

def monitor_progress(log_file, refresh_interval=1.0):
    """Monitor the training progress by watching the log file."""
    last_modification_time = 0
    width = get_terminal_width()
    start_time = time.time()
    
    # Print header
    print("\nTraining Progress Monitor".center(width))
    print("=" * width)
    print(f"Log file: {log_file}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("=" * width)
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            try:
                current_mtime = os.path.getmtime(log_file)
            except FileNotFoundError:
                print(f"Waiting for log file {log_file} to be created...", end="\r")
                time.sleep(refresh_interval)
                continue
            
            # Check if file has been modified
            if current_mtime > last_modification_time:
                # Clear screen
                if os.name == 'nt':  # Windows
                    os.system('cls')
                else:  # Unix/Linux/MacOS
                    os.system('clear')
                
                # Get last few lines of the log file
                last_lines = tail_file(log_file, 20)
                
                # Find the most recent progress percentage and loss
                progress = None
                loss = None
                recent_info = []
                
                for line in reversed(last_lines):
                    if len(recent_info) < 5 and line.strip():
                        recent_info.insert(0, line.strip())
                    
                    if progress is None:
                        progress = extract_progress(line)
                    
                    if loss is None:
                        loss = extract_loss(line)
                    
                    if progress is not None and loss is not None and len(recent_info) >= 3:
                        break
                
                # Print header
                elapsed = time.time() - start_time
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print("\nTraining Progress Monitor".center(width))
                print("=" * width)
                print(f"Log file: {log_file}")
                print(f"Current time: {current_time}")
                print(f"Monitoring time: {format_time_elapsed(elapsed)}")
                print("=" * width)
                
                # Print progress bar if available
                if progress is not None:
                    bar_width = width - 20
                    filled_len = int(round(bar_width * progress / 100))
                    bar = '█' * filled_len + '░' * (bar_width - filled_len)
                    print(f"Progress: |{bar}| {progress:.1f}%")
                    
                    if loss is not None:
                        print(f"Current loss: {loss:.4f}")
                    
                    print("=" * width)
                
                # Print recent log lines
                print("\nRecent log entries:")
                for i, log_line in enumerate(recent_info):
                    print(f"{i+1}. {log_line}")
                
                # Print tail of the log file
                print("\nLog tail:")
                for i, line in enumerate(last_lines[-10:]):
                    print(f"{i+1}: {line.strip()}")
                
                last_modification_time = current_mtime
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log_file", type=str, default="training_log.txt",
                        help="Path to the training log file")
    parser.add_argument("--refresh", type=float, default=1.0,
                        help="Refresh interval in seconds")
    
    args = parser.parse_args()
    monitor_progress(args.log_file, args.refresh) 