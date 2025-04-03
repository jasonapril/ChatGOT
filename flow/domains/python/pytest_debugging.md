# Pytest Debugging Notes (Python Project)

This document captures common issues, patterns, and solutions encountered while debugging `pytest` failures in this Python project, particularly those involving `unittest.mock`.

## Common Issues & Solutions

*   **Mock Call Assertions (`assert_called_once_with`, `assert_any_call`, etc.):**
    *   **Incorrect Arguments:** Carefully check the *exact* arguments (including keyword arguments like `weights_only=False` for `torch.load`) expected by the mock.
    *   **Unexpected Extra Positional Arguments:** Be aware that methods might receive implicit arguments like `self` or a `trainer` instance, which need to be included in the assertion (`assert_called_once_with(mock_trainer, arg1, ...)`) if the mock is on an instance method called externally.
    *   **Incorrect Call Count:** If `assert_called_once` fails with >1 calls, use `assert_called()` or check `mock.call_count`. If other calls are expected (e.g., logger calls during setup/teardown), use `assert_any_call` for the specific call you need to verify, rather than asserting the *only* call was the expected one.
    *   **Mock not Called:** Ensure the code path leading to the mock call is actually executed. Use logging or simple `print` statements (temporarily) if needed. Check that mocks are patched correctly (target the right module path).

*   **Mock Spec Issues:**
    *   `InvalidSpecError: Cannot spec a Mock object`: Don't use a `MagicMock` instance as the `spec` for another `MagicMock`. Use the actual class/type (e.g., `spec=torch.nn.Module`, not `spec=mock_torch.nn.Module`). Ensure necessary imports (like `import torch`) are present.

*   **Configuration in Tests:**
    *   When testing code that reads from a configuration dictionary (e.g., `self.config['training']['max_steps']`), ensure the mock object or test setup correctly populates the *nested* structure the code expects. Simply setting `loop.config['max_steps']` might not work if the code looks for `loop.config['training']['max_steps']`.

*   **Assertion Logic Errors:**
    *   `KeyError` when checking results: Double-check what a function *actually* returns versus what the test assumes (e.g., `load_checkpoint` returning the full state dict vs. just the config). Simplify assertions to check only what's necessary and stable.
    *   `assert X is True`: Be careful using `is` with mocks or potentially complex objects. Prefer `assert X == True` or just `assert X` for truthiness checks unless identity is specifically required.

*   **Debugging Control Flow:**
    *   **Log Message Checks:** When unsure if a specific branch (like an early exit condition) is being hit, add assertions for specific log messages generated within that branch (`mock_logger.info.assert_any_call(...)`). This helps verify control flow without complex state introspection.

*   **TensorBoard / Logging Keys:**
    *   Be mindful that logging keys might change during refactoring (e.g., `train_loss` vs `Loss/train`). Ensure tests expect the current key format.

*   **General Tips:**
    *   Run specific tests using `pytest path/to/test_file.py::TestClass::test_method` to isolate failures.
    *   Use `pytest -v` for more verbose output and `-s` to show `print` statements.
    *   Temporarily add `print()` statements in the source code being tested to understand its state or flow. Remember to remove them afterward.
    *   Read the full `pytest` error output carefully, including the diff in `AssertionError` for mock calls. 