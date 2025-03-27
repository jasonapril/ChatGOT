"""
Unit tests for trainer base classes.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.base import Trainer, LanguageModelTrainer, create_trainer_from_config
from src.models.base import LanguageModel


class MockLanguageModel(LanguageModel):
    """Mock language model for testing."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 10)
        self.linear = nn.Linear(10, 100)
    
    def forward(self, x, targets=None):
        x = self.embedding(x)
        logits = self.linear(x)
        
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return logits, loss
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=10):
        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, input_ids.shape[1] + max_new_tokens)


class MockTrainer(Trainer):
    """Mock implementation of Trainer for testing."""
    
    def train_epoch(self):
        return {"loss": 1.0, "tokens": 100}
    
    def evaluate(self):
        return {"loss": 0.9, "perplexity": 2.5, "tokens": 50}


class TestTrainer(unittest.TestCase):
    """Tests for the Trainer class."""
    
    def setUp(self):
        # Create a simple model
        self.model = MockLanguageModel()
        
        # Create a simple dataset and dataloader
        inputs = torch.randint(0, 100, (100, 10))
        targets = torch.randint(0, 100, (100, 10))
        dataset = TensorDataset(inputs, targets)
        self.train_dataloader = DataLoader(dataset, batch_size=16)
        self.val_dataloader = DataLoader(dataset, batch_size=16)
        
        # Create a trainer
        self.trainer = MockTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            config={"epochs": 2}
        )
    
    def test_initialization(self):
        """Test that the trainer initializes correctly."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.train_dataloader, self.train_dataloader)
        self.assertEqual(self.trainer.val_dataloader, self.val_dataloader)
        self.assertEqual(self.trainer.epochs, 2)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.scheduler)
    
    def test_train(self):
        """Test the train method."""
        metrics = self.trainer.train()
        
        # Check metrics
        self.assertIn("train_loss", metrics)
        self.assertIn("val_loss", metrics)
        self.assertEqual(len(metrics["train_loss"]), 2)  # 2 epochs
        self.assertEqual(len(metrics["val_loss"]), 2)    # 2 epochs
    
    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            
            # Train for one epoch
            self.trainer.train()
            
            # Save checkpoint
            self.trainer.save_checkpoint(checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Create new trainer and load checkpoint
            new_trainer = MockTrainer(
                model=MockLanguageModel(),
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
            )
            new_trainer.load_checkpoint(checkpoint_path)
            
            # Check that metrics were loaded
            self.assertEqual(len(new_trainer.metrics["train_loss"]), 2)  # 2 epochs
            self.assertEqual(len(new_trainer.metrics["val_loss"]), 2)    # 2 epochs


class TestLanguageModelTrainer(unittest.TestCase):
    """Tests for the LanguageModelTrainer class."""
    
    def setUp(self):
        # Create a simple model
        self.model = MockLanguageModel()
        
        # Create a simple dataset and dataloader
        inputs = torch.randint(0, 100, (100, 10))
        targets = torch.randint(0, 100, (100, 10))
        dataset = TensorDataset(inputs, targets)
        self.train_dataloader = DataLoader(dataset, batch_size=16)
        self.val_dataloader = DataLoader(dataset, batch_size=16)
        
        # Create a trainer
        self.trainer = LanguageModelTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            config={"epochs": 1}
        )
    
    @patch("torch.nn.functional.cross_entropy")
    def test_train_epoch(self, mock_cross_entropy):
        """Test the train_epoch method."""
        # Set up mock
        mock_cross_entropy.return_value = torch.tensor(1.0)
        
        # Call train_epoch
        metrics = self.trainer.train_epoch()
        
        # Check metrics
        self.assertIn("loss", metrics)
        self.assertIn("tokens", metrics)
    
    @patch("torch.nn.functional.cross_entropy")
    def test_evaluate(self, mock_cross_entropy):
        """Test the evaluate method."""
        # Set up mock
        mock_cross_entropy.return_value = torch.tensor(0.5)
        
        # Call evaluate
        metrics = self.trainer.evaluate()
        
        # Check metrics
        self.assertIn("loss", metrics)
        self.assertIn("perplexity", metrics)
        self.assertIn("tokens", metrics)
        
        # Check perplexity calculation
        self.assertAlmostEqual(metrics["perplexity"], torch.exp(torch.tensor(0.5)).item())


class TestTrainerCreation(unittest.TestCase):
    """Tests for the trainer creation function."""
    
    def setUp(self):
        # Create a simple model
        self.model = MockLanguageModel()
        
        # Create a simple dataset and dataloader
        inputs = torch.randint(0, 100, (100, 10))
        targets = torch.randint(0, 100, (100, 10))
        dataset = TensorDataset(inputs, targets)
        self.train_dataloader = DataLoader(dataset, batch_size=16)
        self.val_dataloader = DataLoader(dataset, batch_size=16)
    
    def test_create_language_model_trainer(self):
        """Test creating a language model trainer."""
        trainer = create_trainer_from_config(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            config={"epochs": 1}
        )
        
        # Check that the correct trainer type was created
        self.assertIsInstance(trainer, LanguageModelTrainer)
    
    def test_create_unsupported_model_type(self):
        """Test that an error is raised for unsupported model types."""
        model = MagicMock()
        model.model_type = "unsupported"
        
        with self.assertRaises(ValueError):
            create_trainer_from_config(
                model=model,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
            )


if __name__ == "__main__":
    unittest.main() 