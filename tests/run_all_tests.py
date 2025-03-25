#!/usr/bin/env python
"""
Run all ChatGoT tests.

This script discovers and runs all tests for the ChatGoT project, 
including unit tests, integration tests, and optionally benchmarks.
"""

import os
import sys
import time
import unittest
import argparse
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import custom test modules
from tests.chatgot_test_cases import create_unit_tests
from tests.integration_tests import create_integration_tests


class ChatGoTTestLoader(unittest.TestLoader):
    """Custom test loader that can filter slow tests."""
    
    def __init__(self, *args, **kwargs):
        self.skip_slow = kwargs.pop('skip_slow', False)
        super().__init__(*args, **kwargs)
    
    def loadTestsFromTestCase(self, testCaseClass):
        """Load tests from test case, excluding slow tests if requested."""
        test_case_names = self.getTestCaseNames(testCaseClass)
        if not test_case_names and hasattr(testCaseClass, 'runTest'):
            test_case_names = ['runTest']
        
        if self.skip_slow:
            # Filter out tests marked as slow
            test_case_names = [name for name in test_case_names 
                              if not name.lower().startswith('slow_')]
        
        return self.suiteClass(map(testCaseClass, test_case_names))


class ChatGoTTestResult(unittest.TextTestResult):
    """Custom test result class that tracks test execution time."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings = {}
    
    def startTest(self, test):
        """Start timing the test."""
        self._start_time = time.time()
        super().startTest(test)
    
    def stopTest(self, test):
        """Stop timing the test and record the time."""
        elapsed = time.time() - self._start_time
        name = self.getDescription(test)
        self.test_timings[name] = elapsed
        super().stopTest(test)


def run_tests(run_unit=True, run_integration=False, run_benchmarks=False, verbose=1, skip_slow=True):
    """Run the specified tests."""
    loader = ChatGoTTestLoader(skip_slow=skip_slow)
    
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    if run_unit:
        print(f"Running unit tests...")
        
        # Add custom unit tests only
        test_suite.addTests(create_unit_tests())
            
    if run_integration:
        print(f"Running integration tests...")
        
        # Add custom integration tests only
        test_suite.addTests(create_integration_tests())
    
    if run_benchmarks:
        print(f"Running benchmarks...")
        try:
            benchmark_dir = os.path.join(project_root, 'tests', 'benchmarks')
            if os.path.isdir(benchmark_dir):
                benchmark_tests = loader.discover(benchmark_dir, pattern='benchmark_*.py')
                test_suite.addTests(benchmark_tests)
        except Exception as e:
            print(f"Error loading benchmarks: {e}")
    
    # Create a test runner with custom result class
    runner = unittest.TextTestRunner(
        verbosity=verbose,
        resultclass=ChatGoTTestResult
    )
    
    # Run the tests
    print(f"Running tests with Python {sys.version}")
    print(f"Running {test_suite.countTestCases()} tests...")
    result = runner.run(test_suite)
    
    # Print the slowest tests
    if hasattr(result, 'test_timings') and result.test_timings:
        print("\nTop 5 slowest tests:")
        sorted_times = sorted(result.test_timings.items(), key=lambda x: x[1], reverse=True)
        for test_name, timing in sorted_times[:5]:
            print(f"{test_name}: {timing:.3f}s")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    test_counts = test_suite.countTestCases()
    error_count = len(result.errors)
    failure_count = len(result.failures)
    
    test_types = []
    if run_unit:
        test_types.append(f"Unit tests: {test_counts} run, {error_count} errors, {failure_count} failures")
    if run_integration:
        test_types.append(f"Integration tests: {test_counts} run, {error_count} errors, {failure_count} failures")
    if run_benchmarks:
        test_types.append(f"Benchmarks: {test_counts} run, {error_count} errors, {failure_count} failures")
    
    for test_type in test_types:
        print(test_type)
    
    # Return non-zero exit code if there were errors or failures
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChatGoT tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--verbose", "-v", action="count", default=1, help="Increase verbosity")
    parser.add_argument("--include-slow", action="store_true", help="Include tests marked as slow")
    
    args = parser.parse_args()
    
    # If no test type is specified, run unit tests by default
    if not (args.unit or args.integration or args.benchmarks or args.all):
        args.unit = True
    
    # If --all is specified, run all test types
    if args.all:
        args.unit = True
        args.integration = True
        args.benchmarks = True
    
    exit_code = run_tests(
        run_unit=args.unit,
        run_integration=args.integration,
        run_benchmarks=args.benchmarks,
        verbose=args.verbose,
        skip_slow=not args.include_slow
    )
    
    sys.exit(exit_code) 