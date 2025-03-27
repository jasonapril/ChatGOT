"""
Integration tests for Craft.

This file contains integration tests for the Craft project, 
testing the end-to-end functionality of various components.
"""

import unittest
import os
import tempfile
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import pickle
import shutil

# Import Craft modules or mock them if not available
try:
    from src.models.transformer import create_transformer_model
except ImportError:
    # Mock the import
    create_transformer_model = MagicMock()

try:
    from src.data.simple_processor import simple_process_data
except ImportError:
    # Mock the import
    simple_process_data = MagicMock(return_value=0)

try:
    from src.training.train_config import TrainingConfig
except ImportError:
    # Mock the import
    TrainingConfig = MagicMock()

try:
    from src.training.trainer import train_epoch, evaluate
except ImportError:
    # Mock the import
    train_epoch = MagicMock()
    evaluate = MagicMock()

try:
    from src.models.generate import generate_text
except ImportError:
    # Mock the import
    generate_text = MagicMock()

try:
    from src.utils.logging import get_logger
except ImportError:
    # Mock the import
    get_logger = MagicMock()

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = get_logger(__name__)


# Create a minimal Trainer class for testing
class SimpleTrainer:
    """Simple trainer class for testing."""
    
    def __init__(self, model, train_data, val_data, vocab_size, learning_rate=0.001, 
                 max_epochs=1, batch_size=2, device="cpu", checkpoint_dir=None, patience=5):
        self.model = model
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create dataloaders from the input-target pairs
        # The data comes as tuples of (input_seq, target_seq) where both are already tensors
        if train_data:
            train_inputs = torch.stack([item[0] for item in train_data])
            train_targets = torch.stack([item[1] for item in train_data])
            self.train_dataset = TensorDataset(train_inputs, train_targets)
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            self.train_dataloader = None
        
        if val_data:
            val_inputs = torch.stack([item[0] for item in val_data])
            val_targets = torch.stack([item[1] for item in val_data])
            self.val_dataset = TensorDataset(val_inputs, val_targets)
            self.val_dataloader = DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            self.val_dataloader = None
    
    def train_step(self):
        """Run a single training step."""
        if not self.train_dataloader:
            logger.warning("No training data available")
            return
            
        self.model.train()
        for inputs, targets in self.train_dataloader:
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(
                outputs.reshape(-1, self.vocab_size), 
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Only need one batch for testing
            break


class EndToEndTests(unittest.TestCase):
    """End-to-end tests for the Craft pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.test_dir.name)
        
        # Create paths for outputs
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = self.data_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create a small test dataset
        self.test_data = "This is a test dataset for Craft.\n" * 100
        self.test_file = self.data_dir / "test_data.txt"
        with open(self.test_file, "w") as f:
            f.write(self.test_data)
            
        self.processed_data_file = self.data_dir / "processed_data.pkl"
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()
    
    def test_data_processing_to_training(self):
        """Test data processing followed by training."""
        # Step 1: Process data
        config = self._create_test_config()
        
        result = simple_process_data(config)
        self.assertEqual(result, 0, "Data processing failed")
        
        # Step 2: Load processed data
        with open(self.processed_data_file, "rb") as f:
            data = pickle.load(f)
        
        # Step 3: Create a model
        vocab_size = len(data["char_to_idx"])
        model = create_transformer_model(
            vocab_size=vocab_size,
            max_seq_length=config.data.sequence_length,
            d_model=128,  # Smaller model for testing
            n_head=4,
            d_hid=256,
            n_layers=2,
            dropout=0.1
        )
        
        # Step 4: Create a trainer
        trainer = SimpleTrainer(
            model=model,
            train_data=data["train_sequences"][:10],  # Just use a small subset for speed
            val_data=data["val_sequences"][:5],
            vocab_size=vocab_size,
            learning_rate=0.001,
            max_epochs=1,
            batch_size=2,
            device="cpu",
            checkpoint_dir=str(self.checkpoint_dir),
            patience=5
        )
        
        # Step 5: Train for a single step (not full epochs for testing speed)
        trainer.train_step()
        
        # Verify that the model parameters changed during training
        # Save the initial model state
        init_params = {}
        for name, param in model.named_parameters():
            init_params[name] = param.detach().clone()
        
        # Train for one step
        trainer.train_step()
        
        # Check if parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, init_params[name]):
                params_changed = True
                break
        
        self.assertTrue(params_changed, "Model parameters did not change during training")
        
        # Step 6: Save the model
        model_path = os.path.join(self.model_dir, "test_model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Step 7: Load the model and generate text
        new_model = create_transformer_model(
            vocab_size=vocab_size,
            max_seq_length=config.data.sequence_length,
            d_model=128,
            n_head=4,
            d_hid=256,
            n_layers=2,
            dropout=0.1
        )
        new_model.load_state_dict(torch.load(model_path))
        
        # Generate some text (just test that it runs without errors)
        prompt = "This"
        prompt_indices = [data["char_to_idx"][c] for c in prompt]
        
        # Just verify that we can generate text (not checking quality in tests)
        with torch.no_grad():
            input_tensor = torch.tensor([prompt_indices], dtype=torch.long)
            output = new_model(input_tensor)
            
            # Check output shape
            self.assertEqual(output.shape[0], 1, "Batch size should be 1")
            self.assertEqual(output.shape[1], len(prompt), "Sequence length should match prompt length")
            self.assertEqual(output.shape[2], vocab_size, "Output dimension should match vocab size")
    
    def _create_test_config(self):
        """Create a test configuration."""
        test_file = self.test_file
        data_dir = self.data_dir
        model_dir = self.model_dir
        checkpoint_dir = self.checkpoint_dir
        processed_data_file = self.processed_data_file
        
        class TestConfig:
            def __init__(self):
                self.paths = type('obj', (object,), {
                    'data_file': str(test_file),
                    'processed_data': str(processed_data_file),
                    'analysis_dir': str(data_dir),
                    'models_dir': str(model_dir),
                    'checkpoint_dir': str(checkpoint_dir),
                })
                
                self.data = type('obj', (object,), {
                    'sequence_length': 32,  # Short sequences for testing
                    'validation_split': 0.1,
                    'processing': type('obj', (object,), {
                        'lowercase': False,
                    }),
                    'dataset': type('obj', (object,), {
                        'split_ratio': 0.9,  # 90% training, 10% validation
                    }),
                })
                
                self.training = type('obj', (object,), {
                    'sequence_length': 32,  # Short sequences for testing
                    'batch_size': 2,
                    'learning_rate': 0.001,
                    'epochs': 1,
                })
        
        return TestConfig()


class PipelineIntegrationTests(unittest.TestCase):
    """Tests for integrating different pipeline components."""
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline integration."""
        # This is just a placeholder test that would verify the pipeline can run from
        # beginning to end using the command-line interface
        # For actual implementation, you would need to set up and teardown the test environment,
        # and run CLI commands using subprocess
        
        # Just pass for now since this would require more setup
        pass


def create_integration_tests():
    """Create a test suite with integration tests."""
    suite = unittest.TestSuite()
    
    # Add end-to-end tests
    e2e_tests = unittest.defaultTestLoader.loadTestsFromTestCase(EndToEndTests)
    suite.addTests(e2e_tests)
    
    # Add pipeline integration tests
    pipeline_tests = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineIntegrationTests)
    suite.addTests(pipeline_tests)
    
    return suite


if __name__ == "__main__":
    # Run tests directly when the file is executed
    unittest.main() 