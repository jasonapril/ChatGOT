import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"Error importing numpy: {e}")

print("Test completed successfully!")
