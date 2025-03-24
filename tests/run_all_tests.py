#!/usr/bin/env python
"""
Comprehensive Test Runner for ChatGoT

This script discovers and runs all tests, including:
1. Unit tests for each module
2. Integration tests
3. Reporting test coverage

Usage:
    python run_all_tests.py
    python run_all_tests.py --unit-only
    python run_all_tests.py --integration-only
    python run_all_tests.py --timeout 10  # Set timeout to 10 seconds per test
    python run_all_tests.py --test test_save_monitor_stats  # Run a specific test by name
    python run_all_tests.py --module test_instrumentation  # Run all tests in a module
"""

import unittest
import sys
import os
import argparse
import time
import signal
import re
from collections import defaultdict
from contextlib import contextmanager

class TimeoutError(Exception):
    """Exception raised when a test takes too long."""
    pass

@contextmanager
def timeout(seconds):
    """Context manager to timeout a block of code."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")
    
    # Only set timeout on systems that support SIGALRM
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm
    else:
        # On systems without SIGALRM (like Windows), skip timeout
        yield

class VerboseTestResult(unittest.TextTestResult):
    """Custom test result class that shows progress for each test."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.slow_tests = []
        self.current_test = None
        self.current_test_start_time = 0
        self.tests_run_so_far = 0
        self.total_tests = 0
        self.test_timeout = None
    
    def startTest(self, test):
        """Called when a test starts."""
        self.current_test = test
        self.current_test_start_time = time.time()
        self.tests_run_so_far += 1
        
        # Print progress information
        test_name = self.getDescription(test)
        progress = f"[{self.tests_run_so_far}/{self.total_tests}]" if self.total_tests else ""
        self.stream.write(f"\r{progress} Running: {test_name}".ljust(100))
        self.stream.flush()
        
        # Apply timeout if specified
        if self.test_timeout:
            if hasattr(signal, 'SIGALRM'):  # Check if timeout is supported
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.test_timeout)
            
        super().startTest(test)
    
    def _timeout_handler(self, signum, frame):
        """Handle test timeout."""
        test_name = self.getDescription(self.current_test)
        self.stream.writeln(f"\nTEST TIMEOUT: {test_name} took longer than {self.test_timeout} seconds!")
        raise TimeoutError(f"Test {test_name} timed out after {self.test_timeout} seconds")
    
    def stopTest(self, test):
        """Called when a test finishes."""
        elapsed = time.time() - self.current_test_start_time
        
        # Disable timeout alarm
        if self.test_timeout and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        # Record slow tests (taking more than 1 second)
        if elapsed > 1.0:
            self.slow_tests.append((self.getDescription(test), elapsed))
        
        # Clear the current progress line
        self.stream.write("\r" + " " * 100 + "\r")
        
        # Always show test result with timing
        self.stream.writeln(f"  {self.getDescription(test)} ... {elapsed:.2f}s")
            
        super().stopTest(test)

class VerboseTestRunner(unittest.TextTestRunner):
    """Custom test runner that shows progress for each test."""
    
    def __init__(self, stream=None, descriptions=True, verbosity=1, failfast=False, buffer=False, 
                 warnings=None, *, tb_locals=False, test_timeout=None):
        super().__init__(stream, descriptions, verbosity, failfast, buffer, warnings, tb_locals=tb_locals)
        self.test_timeout = test_timeout
    
    def _makeResult(self):
        """Create a test result object."""
        result = VerboseTestResult(self.stream, self.descriptions, self.verbosity)
        result.test_timeout = self.test_timeout
        return result
    
    def run(self, test):
        """Run the test suite with progress reporting."""
        result = super()._makeResult()
        result.test_timeout = self.test_timeout
        
        # Count total tests for progress reporting
        test_count = test.countTestCases()
        result.total_tests = test_count
        
        self.stream.writeln(f"Running {test_count} tests...")
        self.stream.writeln("-" * 70)
        
        startTime = time.time()
        try:
            test(result)
        finally:
            stopTime = time.time()
        
        timeTaken = stopTime - startTime
        result.printErrors()
        
        self.stream.writeln("-" * 70)
        self.stream.writeln(f"Ran {result.testsRun} tests in {timeTaken:.3f}s")
        
        # Report slow tests
        if result.slow_tests:
            self.stream.writeln("\nSlow Tests (>1s):")
            for test_name, elapsed in sorted(result.slow_tests, key=lambda x: x[1], reverse=True):
                self.stream.writeln(f"  {test_name}: {elapsed:.2f}s")
        
        if not result.wasSuccessful():
            self.stream.writeln("FAILED")
        else:
            self.stream.writeln("OK")
        
        return result

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run tests for ChatGoT')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--pattern', '-p', type=str, default='test_*.py', help='Test file pattern')
    parser.add_argument('--failfast', '-f', action='store_true', help='Stop on first failure')
    parser.add_argument('--timeout', '-t', type=int, default=None, 
                        help='Timeout in seconds for each individual test')
    parser.add_argument('--skip-slow', action='store_true', 
                        help='Skip known slow tests (tests in skip_slow_tests list)')
    parser.add_argument('--test', type=str, 
                        help='Run a specific test by name (e.g., test_save_monitor_stats)')
    parser.add_argument('--module', type=str, 
                        help='Run all tests in a specific module (e.g., test_instrumentation)')
    parser.add_argument('--list', action='store_true',
                        help='List all discovered tests without running them')
    return parser.parse_args()

# List of tests known to be slow - update as needed
SKIP_SLOW_TESTS = [
    'test_train_epoch_with_amp',  # Example of a slow test pattern
    'test_batch_generate',        # Example of another slow test
    'test_evaluate_with_metrics'  # Another slow test
]

def filter_test_suite(suite, test_filter=None, module_filter=None):
    """Filter a test suite by test name or module name."""
    if not test_filter and not module_filter:
        return suite
        
    filtered_suite = unittest.TestSuite()
    
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            # Recursively filter sub-suites
            sub_filtered = filter_test_suite(test, test_filter, module_filter)
            if sub_filtered.countTestCases() > 0:
                filtered_suite.addTest(sub_filtered)
        else:
            # Check if this test matches the filter
            test_id = test.id()
            test_name = test_id.split('.')[-1]
            module_name = test_id.split('.')[-2] if len(test_id.split('.')) > 1 else ""
            
            if test_filter and test_name == test_filter:
                filtered_suite.addTest(test)
            elif module_filter and module_filter in module_name:
                filtered_suite.addTest(test)
                
    return filtered_suite

def discover_and_list_tests():
    """Discover all tests and list them without running."""
    loader = unittest.TestLoader()
    
    print("\nDISCOVERED TESTS:")
    print("-" * 70)
    
    # Discover and list unit tests
    print("\nUNIT TESTS:")
    unit_suite = loader.discover('tests/unit', pattern='test_*.py')
    list_tests_in_suite(unit_suite)
    
    # Discover and list integration tests
    print("\nINTEGRATION TESTS:")
    integration_suite = loader.discover('tests/integration', pattern='test_*.py')
    list_tests_in_suite(integration_suite)

def list_tests_in_suite(suite, indent=0):
    """List all tests in a test suite with indentation."""
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            # For test suites, list the name and recurse
            if test.countTestCases() > 0:
                suite_name = test._tests[0].__class__.__name__ if test._tests else "Unknown"
                print(f"{'  ' * indent}{suite_name} ({test.countTestCases()} tests)")
                list_tests_in_suite(test, indent + 1)
        else:
            # For individual tests, list the test name
            test_name = test.id().split('.')[-1]
            print(f"{'  ' * indent}- {test_name}")

def run_tests(args):
    """Discover and run tests based on arguments."""
    # Add the project root to sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Just list tests if requested
    if args.list:
        discover_and_list_tests()
        return 0
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    # Set up test discovery
    loader = unittest.TestLoader()
    
    # Optionally skip slow tests
    if args.skip_slow:
        def skip_slow_tests(test_suite):
            # Recursively filter out slow tests
            if isinstance(test_suite, unittest.TestSuite):
                filtered_suite = unittest.TestSuite()
                for test in test_suite:
                    # If it's a suite, recursively filter it
                    if isinstance(test, unittest.TestSuite):
                        filtered_sub_suite = skip_slow_tests(test)
                        if filtered_sub_suite.countTestCases() > 0:
                            filtered_suite.addTest(filtered_sub_suite)
                    # If it's a test, check if it should be skipped
                    else:
                        test_name = test.id().split('.')[-1]
                        if not any(slow_pattern in test_name for slow_pattern in SKIP_SLOW_TESTS):
                            filtered_suite.addTest(test)
                return filtered_suite
            return test_suite
        
        # Apply the filter to the loader
        original_load_tests = loader.loadTestsFromTestCase
        def filtered_load_tests(testCaseClass):
            suite = original_load_tests(testCaseClass)
            return skip_slow_tests(suite)
        loader.loadTestsFromTestCase = filtered_load_tests
    
    # Tracks test results by category
    results = defaultdict(lambda: {'total': 0, 'failures': 0, 'errors': 0, 'skipped': 0})
    all_failures = []
    all_errors = []
    total_time = 0
    
    # Print header
    print("\n" + "=" * 70)
    print("RUNNING CHATGOT TESTS")
    print("=" * 70)
    
    # If running a specific test or module, create a combined test suite
    if args.test or args.module:
        combined_suite = unittest.TestSuite()
        
        # Discover all unit tests
        if not args.integration_only:
            unit_suite = loader.discover('tests/unit', pattern='test_*.py')
            filtered_unit_suite = filter_test_suite(unit_suite, args.test, args.module)
            combined_suite.addTest(filtered_unit_suite)
        
        # Discover all integration tests
        if not args.unit_only:
            integration_suite = loader.discover('tests/integration', pattern='test_*.py')
            filtered_integration_suite = filter_test_suite(integration_suite, args.test, args.module)
            combined_suite.addTest(filtered_integration_suite)
        
        if combined_suite.countTestCases() == 0:
            print(f"\nNo tests found matching: {args.test or args.module}")
            return 1
        
        print(f"\nRUNNING FILTERED TESTS: {args.test or args.module}")
        start_time = time.time()
        
        runner = VerboseTestRunner(verbosity=verbosity, failfast=args.failfast, test_timeout=args.timeout)
        combined_result = runner.run(combined_suite)
        
        # Update results
        elapsed = time.time() - start_time
        results['filtered']['total'] = combined_result.testsRun
        results['filtered']['failures'] = len(combined_result.failures)
        results['filtered']['errors'] = len(combined_result.errors)
        results['filtered']['skipped'] = len(combined_result.skipped)
        results['filtered']['time'] = elapsed
        
        all_failures.extend(combined_result.failures)
        all_errors.extend(combined_result.errors)
        total_time += elapsed
    else:
        # Run unit tests if requested
        if not args.integration_only:
            print("\nRUNNING UNIT TESTS")
            start_time = time.time()
            
            unit_suite = loader.discover('tests/unit', pattern=args.pattern)
            runner = VerboseTestRunner(verbosity=verbosity, failfast=args.failfast, test_timeout=args.timeout)
            unit_result = runner.run(unit_suite)
            
            # Update results
            elapsed = time.time() - start_time
            results['unit']['total'] = unit_result.testsRun
            results['unit']['failures'] = len(unit_result.failures)
            results['unit']['errors'] = len(unit_result.errors)
            results['unit']['skipped'] = len(unit_result.skipped)
            results['unit']['time'] = elapsed
            
            all_failures.extend(unit_result.failures)
            all_errors.extend(unit_result.errors)
            total_time += elapsed
            
            # Break on failure if requested
            if args.failfast and (unit_result.failures or unit_result.errors):
                return 1
        
        # Run integration tests if requested
        if not args.unit_only:
            print("\nRUNNING INTEGRATION TESTS")
            start_time = time.time()
            
            integration_suite = loader.discover('tests/integration', pattern=args.pattern)
            runner = VerboseTestRunner(verbosity=verbosity, failfast=args.failfast, test_timeout=args.timeout)
            integration_result = runner.run(integration_suite)
            
            # Update results
            elapsed = time.time() - start_time
            results['integration']['total'] = integration_result.testsRun
            results['integration']['failures'] = len(integration_result.failures)
            results['integration']['errors'] = len(integration_result.errors)
            results['integration']['skipped'] = len(integration_result.skipped)
            results['integration']['time'] = elapsed
            
            all_failures.extend(integration_result.failures)
            all_errors.extend(integration_result.errors)
            total_time += elapsed
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    grand_total = 0
    grand_failures = 0
    grand_errors = 0
    grand_skipped = 0
    
    for category, result in results.items():
        print(f"{category.upper()} TESTS:")
        print(f"  Ran {result['total']} tests in {result['time']:.2f}s")
        print(f"  Failures: {result['failures']}")
        print(f"  Errors: {result['errors']}")
        print(f"  Skipped: {result['skipped']}")
        print()
        
        grand_total += result['total']
        grand_failures += result['failures']
        grand_errors += result['errors']
        grand_skipped += result['skipped']
    
    # Print grand total
    print("OVERALL:")
    print(f"  Ran {grand_total} tests in {total_time:.2f}s")
    print(f"  Failures: {grand_failures}")
    print(f"  Errors: {grand_errors}")
    print(f"  Skipped: {grand_skipped}")
    
    # Print summary status
    if grand_failures == 0 and grand_errors == 0:
        print("\nOVERALL STATUS: ✓ PASS")
    else:
        print("\nOVERALL STATUS: ✗ FAIL")
    
    # Return non-zero exit code if any tests failed
    return 0 if grand_failures == 0 and grand_errors == 0 else 1

if __name__ == '__main__':
    args = parse_args()
    sys.exit(run_tests(args)) 