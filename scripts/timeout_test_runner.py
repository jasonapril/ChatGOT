#!/usr/bin/env python
"""
Test runner with timeout functionality
"""

import unittest
import sys
import os
import time
import threading
import traceback

def run_test_with_timeout(test_name, timeout=5):
    """Run a specific test with a timeout."""
    # Parse test path
    parts = test_name.split('.')
    if len(parts) < 3:
        print(f"Invalid test path: {test_name}")
        print("Format should be: package.module.TestClass.test_method")
        return 1
    
    module_path = '.'.join(parts[:-2])
    class_name = parts[-2]
    method_name = parts[-1]
    
    # Try to import the module
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as e:
        print(f"Error importing module {module_path}: {e}")
        return 1
    
    # Get the test class
    try:
        test_class = getattr(module, class_name)
    except AttributeError:
        print(f"Test class {class_name} not found in module {module_path}")
        return 1
    
    # Create test suite with just the specified test
    suite = unittest.TestSuite()
    try:
        suite.addTest(test_class(method_name))
    except AttributeError:
        print(f"Test method {method_name} not found in class {class_name}")
        return 1
    
    # Create a runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Create a result object to store the results
    result = unittest.TestResult()
    
    # Run the test in a separate thread with a timeout
    print("\nStarting test with timeout...")
    
    def run_test():
        try:
            print("Test thread started")
            suite.run(result)
            print("Test thread finished")
        except Exception as e:
            print(f"Exception in test thread: {e}")
            traceback.print_exc()
    
    test_thread = threading.Thread(target=run_test)
    test_thread.daemon = True
    
    start_time = time.time()
    test_thread.start()
    
    # Wait for the test to complete or timeout
    test_thread.join(timeout)
    elapsed = time.time() - start_time
    
    if test_thread.is_alive():
        print(f"\n*** TEST TIMEOUT after {elapsed:.2f}s ***")
        print("The test is still running but will be forcibly terminated.")
        return 1
    else:
        print(f"\nTest completed in {elapsed:.2f}s")
        
        # Check for failures
        if result.wasSuccessful():
            print("Test passed!")
            return 0
        else:
            print("Test failed:")
            for error in result.errors:
                print(f"ERROR: {error[0]}")
                print(error[1])
            for failure in result.failures:
                print(f"FAILURE: {failure[0]}")
                print(failure[1])
            return 1
