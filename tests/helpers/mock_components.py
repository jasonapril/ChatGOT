import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Moved from tests/training/trainer/test_trainer_init_optionals.py
class MockCallbackTarget:
    """Dummy target class for callback instantiation tests."""
    pass

# Define MockModel here to be importable by Hydra
class MockModel(nn.Module):
    """Minimal mock model for testing purposes."""
    def __init__(self, config: dict): # Accept a config dict
        super().__init__()
        self.config = config # Store config if needed
        self.layer = nn.Linear(10, 10) # Example layer

    def forward(self, x):
        return self.layer(x)

    # Add any other methods expected by the code using this mock
    # e.g., parameters() if optimizer initialization needs it
    def parameters(self):
        return self.layer.parameters()


class MockDataset(Dataset):
    """Minimal mock dataset for testing dataloader instantiation."""
    def __init__(self, length: int = 10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError
        return torch.tensor([idx]) # Return a simple tensor 