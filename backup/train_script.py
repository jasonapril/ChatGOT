#!/usr/bin/env python3
"""
ChatGoT Training Script
=======================

A simple wrapper script that calls the main training function.
This script demonstrates how to use the refactored modules for training.

Usage:
    python train_script.py --data_path processed_data/got_char_data.pkl --epochs 10
"""

import sys
import os
from src.train import main

if __name__ == "__main__":
    # This script simply calls the main function from src.train
    # All command line arguments will be passed through automatically
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 