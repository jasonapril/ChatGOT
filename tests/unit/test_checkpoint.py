"""
Unit tests for the checkpoint utility module.
"""
import unittest
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import shutil
from datetime import datetime

from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    count_checkpoints,
    clean_old_checkpoints
)

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.output = nn.Linear(5, 2)
        
    def forward(self, x):
        x = torch.relu(self.linear(x))
        return self.output(x)


class TestCheckpoint(unittest.TestCase):
    """Tests for checkpoint.py functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple model, optimizer, and scheduler for testing
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Sample input for testing
        self.input_tensor = torch.randn(2, 10)
        
        # Get initial output before checkpoint
        self.initial_output = self.model(self.input_tensor)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        # Create checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.pt")
        
        # Force CPU device for testing to avoid device mismatch issues
        device = torch.device('cpu')
        self.input_tensor = self.input_tensor.to(device)
        self.model = self.model.to(device)
        
        # Additional data to include in checkpoint
        additional_data = {
            "test_data": "test_value",
            "timestamp": datetime.now().isoformat(),
            "config": {"model_type": "test", "layers": 2}
        }
        
        # Save checkpoint
        save_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=5,
            loss=0.123,
            additional_data=additional_data
        )
        
        # Verify checkpoint file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Modify model parameters to simulate training
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Verify model output has changed
        modified_output = self.model(self.input_tensor)
        self.assertFalse(torch.allclose(self.initial_output, modified_output))
        
        # Load checkpoint
        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=device
        )
        
        # Verify restored output matches initial output
        restored_output = self.model(self.input_tensor)
        self.assertTrue(torch.allclose(self.initial_output, restored_output))
        
        # Verify additional data was loaded correctly
        self.assertEqual(checkpoint["test_data"], "test_value")
        self.assertEqual(checkpoint["epoch"], 5)
        self.assertEqual(checkpoint["loss"], 0.123)
        self.assertEqual(checkpoint["config"]["model_type"], "test")
    
    def test_get_latest_checkpoint(self):
        """Test getting the latest checkpoint."""
        # Create multiple checkpoints with delays to ensure different modification times
        for i in range(3):
            checkpoint_path = os.path.join(self.temp_dir, f"checkpoint_{i}.pt")
            save_checkpoint(checkpoint_path, self.model, epoch=i)
            # Sleep briefly to ensure different modification times
            import time
            time.sleep(0.1)
        
        # Get the latest checkpoint
        latest = get_latest_checkpoint(self.temp_dir)
        
        # Verify it's the most recent one
        self.assertEqual(latest, os.path.join(self.temp_dir, "checkpoint_2.pt"))
    
    def test_count_checkpoints(self):
        """Test counting checkpoints."""
        # Create multiple checkpoints
        for i in range(5):
            checkpoint_path = os.path.join(self.temp_dir, f"checkpoint_{i}.pt")
            save_checkpoint(checkpoint_path, self.model)
        
        # Count checkpoints
        count = count_checkpoints(self.temp_dir)
        
        # Verify count
        self.assertEqual(count, 5)
        
        # Test with pattern
        count = count_checkpoints(self.temp_dir, pattern="checkpoint_[0-2].pt")
        self.assertEqual(count, 3)
    
    def test_clean_old_checkpoints(self):
        """Test cleaning old checkpoints."""
        # Create multiple checkpoints with delays to ensure different modification times
        for i in range(7):
            checkpoint_path = os.path.join(self.temp_dir, f"model_epoch_{i}.pt")
            save_checkpoint(checkpoint_path, self.model, epoch=i)
            # Sleep briefly to ensure different modification times
            import time
            time.sleep(0.1)
        
        # Clean old checkpoints, keeping only the 3 most recent
        clean_old_checkpoints(self.temp_dir, keep=3, pattern="model_epoch_*.pt")
        
        # Count remaining checkpoints
        remaining = [f for f in os.listdir(self.temp_dir) if f.startswith("model_epoch_")]
        
        # Verify only 3 checkpoints remain
        self.assertEqual(len(remaining), 3)
        
        # Verify the oldest were removed
        self.assertNotIn("model_epoch_0.pt", remaining)
        self.assertNotIn("model_epoch_1.pt", remaining)
        self.assertNotIn("model_epoch_2.pt", remaining)
        self.assertNotIn("model_epoch_3.pt", remaining)
        
        # Verify the newest remain
        self.assertIn("model_epoch_4.pt", remaining)
        self.assertIn("model_epoch_5.pt", remaining)
        self.assertIn("model_epoch_6.pt", remaining)
    
    def test_checkpoint_non_existent_directory(self):
        """Test handling of non-existent directory."""
        non_existent_dir = os.path.join(self.temp_dir, "non_existent")
        
        # These should not raise exceptions, even with a non-existent directory
        latest = get_latest_checkpoint(non_existent_dir)
        self.assertIsNone(latest)
        
        count = count_checkpoints(non_existent_dir)
        self.assertEqual(count, 0)
        
        # This should not raise an exception
        clean_old_checkpoints(non_existent_dir)


if __name__ == "__main__":
    unittest.main() 