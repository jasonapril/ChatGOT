#!/usr/bin/env python
"""Run all Craft tests.

This script discovers and runs all tests for the Craft project,
providing options for running specific test types (unit, integration)
and filtering by test name.
"""
import os
import sys
import argparse
import unittest
import importlib
from pathlib import Path
from unittest import TestLoader, TextTestRunner, TestSuite

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def discover_tests(start_dir):
    """
    Discover all test modules in a directory.
    
    Args:
        start_dir: Directory path (string or Path) to search for tests
        
    Returns:
        TestSuite with all discovered tests
    """
    loader = TestLoader()
    # Discover tests recursively starting from the given directory
    print(f"Discovering tests in: {start_dir}")
    suite = loader.discover(str(start_dir), pattern='test_*.py', top_level_dir=str(project_root))
    print(f"Discovered {suite.countTestCases()} tests.")
    return suite

def run_tests(test_suite, verbosity=2):
    """
    Run a test suite.
    
    Args:
        test_suite: Test suite to run
        verbosity: Verbosity level (1-3)
        
    Returns:
        Test results
    """
    runner = TextTestRunner(verbosity=verbosity)
    return runner.run(test_suite)

def filter_tests_by_name(suite, test_name):
    """
    Filter tests by name.
    
    Args:
        suite: TestSuite to filter
        test_name: Name of the test to filter for
        
    Returns:
        Filtered TestSuite
    """
    filtered_suite = unittest.TestSuite()
    
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            # Recursively filter nested test suites
            filtered_sub_suite = filter_tests_by_name(test, test_name)
            if filtered_sub_suite.countTestCases() > 0:
                filtered_suite.addTest(filtered_sub_suite)
        elif test_name.lower() in test.id().lower():
            # Add test if the name matches
            filtered_suite.addTest(test)
    
    return filtered_suite

def create_test_suite(test_type=None, test_name=None):
    """
    Create a test suite based on discovery and optional filters.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', or None for all)
                     Note: 'integration' tests were removed.
        test_name: Specific test name to run (or None for all)
        
    Returns:
        TestSuite instance
    """
    loader = TestLoader()
    suite = TestSuite()
    tests_dir = Path(__file__).parent # Get the directory containing this script (tests/)

    # Discover all tests first
    # Use project_root as top_level_dir to ensure correct module paths
    all_tests_suite = loader.discover(str(tests_dir), pattern="test_*.py", top_level_dir=str(project_root))

    # --- Apply Filters --- 
    filtered_suite = all_tests_suite # Start with all discovered tests

    # Filter by type (Note: 'integration' might not match anything now)
    if test_type:
        temp_suite = TestSuite()
        for test in filtered_suite:
            # Check if the test module path contains the type string
            # This is a simple way; might need refinement based on actual structure
            module_path = getattr(test, '__module__', '').lower()
            if f'.{test_type}.' in module_path or f'tests.{test_type}_' in module_path:
                 temp_suite.addTest(test)
        filtered_suite = temp_suite

    # Filter by name
    if test_name:
        filtered_suite = filter_tests_by_name(filtered_suite, test_name)

    suite.addTests(filtered_suite)
    print(f"Final suite contains {suite.countTestCases()} tests after filtering.")
    return suite

def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(description='Run Craft tests')
    parser.add_argument('--type', choices=['unit', 'integration'], 
                        help='Type of tests to run (e.g., based on path containing unit/integration)')
    parser.add_argument('--name', help='Filter tests by name substring (case-insensitive)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Verbose output')
    args = parser.parse_args()
    
    # Create test suite using discovery and filters
    suite = create_test_suite(args.type, args.name)
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    print(f"Running tests with verbosity {verbosity}...")
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    print(f"\nTests run: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main() 