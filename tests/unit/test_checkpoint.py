"""
Unit tests for checkpoint utilities.
"""

import os
import shutil
import tempfile
import unittest
from time import sleep

import torch
import torch.nn as nn

from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    count_checkpoints,
    clean_old_checkpoints
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoints."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


class TestCheckpoint(unittest.TestCase):
    """Test cases for checkpoint utilities."""
    
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple model for testing
        self.model = SimpleModel()
        
        # Create a simple optimizer for testing
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        # Define checkpoint path
        checkpoint_path = os.path.join(self.test_dir, "checkpoint.pt")
        
        # Store the original parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        # Create checkpoint data
        epoch = 5
        loss = 0.1
        additional_data = {"custom_key": "custom_value"}
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            **additional_data
        }
        
        # Save checkpoint
        save_checkpoint(checkpoint, checkpoint_path)
        
        # Verify that the checkpoint file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Change the model parameters to verify that loading works
        for param in self.model.parameters():
            param.data.fill_(0.0)
        
        # Verify parameters were changed
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.all(param.data == 0.0))
        
        # Load the checkpoint
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        
        # Check that the loaded checkpoint has the correct data
        self.assertEqual(loaded_checkpoint['epoch'], epoch)
        self.assertEqual(loaded_checkpoint['loss'], loss)
        self.assertEqual(loaded_checkpoint['custom_key'], "custom_value")
        
        # Verify that the model can be restored from the loaded checkpoint
        new_model = SimpleModel()
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        # The parameters of the new model should match the original model parameters
        for name, param in new_model.named_parameters():
            self.assertFalse(torch.all(param.data == 0.0))
            self.assertTrue(torch.allclose(param.data, original_params[name]))
    
    def test_get_latest_checkpoint(self):
        """Test getting the latest checkpoint."""
        # Create multiple checkpoint files with delays to ensure different modification times
        for i in range(3):
            checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}.pt")
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch': i
            }
            
            save_checkpoint(checkpoint, checkpoint_path)
            sleep(0.1)  # Add a small delay to ensure different modification times
        
        # Get the latest checkpoint
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        # The latest checkpoint should be the one with i=2
        self.assertEqual(latest_checkpoint, os.path.join(checkpoint_dir, "checkpoint_2.pt"))
        
        # Test with non-existent directory
        latest_checkpoint = get_latest_checkpoint(os.path.join(self.test_dir, "non_existent"))
        self.assertIsNone(latest_checkpoint)
    
    def test_count_checkpoints(self):
        """Test counting checkpoints."""
        # Create multiple checkpoint files
        checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        num_checkpoints = 5
        for i in range(num_checkpoints):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}.pt")
            
            checkpoint = {
                'model_state_dict': self.model.state_dict()
            }
            
            save_checkpoint(checkpoint, checkpoint_path)
        
        # Count checkpoints
        count = count_checkpoints(checkpoint_dir)
        self.assertEqual(count, num_checkpoints)
        
        # Test with non-existent directory
        count = count_checkpoints(os.path.join(self.test_dir, "non_existent"))
        self.assertEqual(count, 0)
    
    def test_clean_old_checkpoints(self):
        """Test cleaning old checkpoints."""
        # Create multiple checkpoint files with delays to ensure different modification times
        checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        num_checkpoints = 10
        for i in range(num_checkpoints):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}.pt")
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch': i
            }
            
            save_checkpoint(checkpoint, checkpoint_path)
            sleep(0.1)  # Add a small delay to ensure different modification times
        
        # Clean old checkpoints, keeping the 3 most recent ones
        keep = 3
        clean_old_checkpoints(checkpoint_dir, keep=keep)
        
        # Count the remaining checkpoints
        remaining_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        self.assertEqual(len(remaining_checkpoints), keep)
        
        # The remaining checkpoints should be the most recent ones
        for i in range(num_checkpoints - keep, num_checkpoints):
            self.assertIn(f"checkpoint_{i}.pt", remaining_checkpoints)
        
        # Test with non-existent directory
        clean_old_checkpoints(os.path.join(self.test_dir, "non_existent"), keep=1)


if __name__ == "__main__":
    unittest.main() 