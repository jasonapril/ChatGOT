#!/usr/bin/env python
"""
Test Runner for ChatGoT

This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os

def run_tests():
    """Discover and run all tests."""
    # Add the project root to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests()) 