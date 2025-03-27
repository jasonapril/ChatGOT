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

# Import test modules
from tests.craft_test_cases import create_unit_tests
from tests.integration_tests import create_integration_tests

def discover_tests(dir_path):
    """
    Discover all test modules in a directory.
    
    Args:
        dir_path: Directory to search for tests
        
    Returns:
        TestSuite with all discovered tests
    """
    loader = TestLoader()
    return loader.discover(dir_path, pattern='test_*.py')

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
    Create a test suite based on test type and name filters.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', or None for all)
        test_name: Specific test name to run (or None for all)
        
    Returns:
        TestSuite instance
    """
    suite = TestSuite()
    
    # Add unit tests
    if test_type is None or test_type == 'unit':
        # Add base unit tests
        unit_tests = create_unit_tests()
        
        # Add new unit tests for checkpoint, io, and CLI
        unit_module_names = [
            'tests.unit.test_checkpoint',
            'tests.unit.test_io',
            'tests.unit.test_cli'
        ]
        
        for module_name in unit_module_names:
            try:
                module = importlib.import_module(module_name)
                module_tests = unittest.defaultTestLoader.loadTestsFromModule(module)
                unit_tests.addTests(module_tests)
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
        
        if test_name:
            filtered_tests = filter_tests_by_name(unit_tests, test_name)
            suite.addTests(filtered_tests)
        else:
            suite.addTest(unit_tests)
    
    # Add integration tests
    if test_type is None or test_type == 'integration':
        integration_tests = create_integration_tests()
        if test_name:
            filtered_tests = filter_tests_by_name(integration_tests, test_name)
            suite.addTests(filtered_tests)
        else:
            suite.addTest(integration_tests)
    
    return suite

def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(description='Run Craft tests')
    parser.add_argument('--type', choices=['unit', 'integration'], 
                        help='Type of tests to run')
    parser.add_argument('--name', help='Filter tests by name')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Verbose output')
    args = parser.parse_args()
    
    # Create test suite
    suite = create_test_suite(args.type, args.name)
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    main() 