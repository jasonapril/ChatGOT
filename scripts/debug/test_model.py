import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    from chatgot.models import gpt_model
    print(f"Successfully imported GPT model module")
except ImportError as e:
    print(f"Error importing GPT model: {e}")

print("Test completed successfully!") 