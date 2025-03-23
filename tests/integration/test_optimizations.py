import unittest
import torch
import time
import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path
import numpy as np

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import create_transformer_model
from src.trainer import train_epoch, evaluate
from src.data_handler import create_dataloader

class TestTorchCompileIntegration(unittest.TestCase):
    """Integration tests for torch.compile optimization."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Disable logging for tests
        logging.basicConfig(level=logging.ERROR)
        
        # Create a temporary directory for test artifacts
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create a small synthetic dataset for training
        cls.batch_size = 4
        cls.seq_length = 32
        cls.vocab_size = 65  # basic ASCII character set
        
        # Set up device - prefer CUDA if available
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate synthetic character-level dataset
        def generate_synthetic_data(num_samples=100):
            """Generate synthetic character-level data."""
            chars = [chr(i) for i in range(32, 97)]  # ASCII characters
            
            # Create character to index mapping
            char_to_idx = {char: i for i, char in enumerate(chars)}
            idx_to_char = {i: char for i, char in enumerate(chars)}
            
            # Generate random sequences
            data = []
            for _ in range(num_samples):
                # Generate random sequence of indices
                indices = np.random.randint(0, cls.vocab_size, cls.seq_length + 1)
                x = torch.tensor(indices[:-1], dtype=torch.long)
                y = torch.tensor(indices[1:], dtype=torch.long)
                data.append((x, y))
            
            return data, char_to_idx, idx_to_char
        
        # Generate data
        cls.data, cls.char_to_idx, cls.idx_to_char = generate_synthetic_data()
        
        # Create model with small dimensions for testing
        cls.model = create_transformer_model(
            vocab_size=cls.vocab_size,
            max_seq_length=cls.seq_length,
            d_model=128,
            n_head=4,
            d_hid=256,
            n_layers=2,
            dropout=0.1
        )
        
        # Create optimizer
        cls.optimizer = torch.optim.AdamW(cls.model.parameters(), lr=5e-5)
        
        # Create learning rate scheduler
        cls.scheduler = torch.optim.lr_scheduler.LambdaLR(
            cls.optimizer, lambda _: 1.0
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def _create_dataloaders(self):
        """Create training and validation data loaders from synthetic data."""
        # Split data into train and validation
        train_size = int(0.8 * len(self.data))
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]
        
        # Custom dataset class
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create datasets
        train_dataset = SyntheticDataset(train_data)
        val_dataset = SyntheticDataset(val_data)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def _train_model(self, use_compile, compile_mode='reduce-overhead', epochs=2):
        """Train model with or without compilation."""
        # Reset model
        self.model = create_transformer_model(
            vocab_size=self.vocab_size,
            max_seq_length=self.seq_length,
            d_model=128,
            n_head=4,
            d_hid=256,
            n_layers=2,
            dropout=0.1
        )
        self.model.to(self.device)
        
        # Reset optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _: 1.0
        )
        
        # Create data loaders
        train_loader, val_loader = self._create_dataloaders()
        
        # Train for specified number of epochs
        train_losses = []
        train_times = []
        tokens_per_second = []
        
        for epoch in range(epochs):
            start_time = time.time()
            
            epoch_loss, epoch_tokens_per_sec = train_epoch(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                dataloader=train_loader,
                device=self.device,
                epoch=epoch,
                use_torch_compile=use_compile,
                compile_mode=compile_mode
            )
            
            epoch_time = time.time() - start_time
            
            train_losses.append(epoch_loss)
            train_times.append(epoch_time)
            tokens_per_second.append(epoch_tokens_per_sec)
        
        # Evaluate on validation set
        val_loss = evaluate(self.model, val_loader, self.device)
        
        return {
            'train_losses': train_losses,
            'train_times': train_times,
            'tokens_per_second': tokens_per_second,
            'val_loss': val_loss
        }
    
    @unittest.skipIf(not hasattr(torch, 'compile'), "torch.compile not available")
    def test_torch_compile_integration(self):
        """Test that a model trained with torch.compile produces reasonable results."""
        # Check if we should skip CUDA tests
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for torch.compile test")
        
        # Train with and without compilation
        metrics_without_compile = self._train_model(use_compile=False)
        metrics_with_compile = self._train_model(use_compile=True)
        
        # Output results for comparison
        print("\nTraining without torch.compile:")
        print(f"Final train loss: {metrics_without_compile['train_losses'][-1]:.6f}")
        print(f"Validation loss: {metrics_without_compile['val_loss']:.6f}")
        print(f"Avg tokens/sec: {np.mean(metrics_without_compile['tokens_per_second']):.2f}")
        print(f"Total training time: {sum(metrics_without_compile['train_times']):.2f}s")
        
        print("\nTraining with torch.compile:")
        print(f"Final train loss: {metrics_with_compile['train_losses'][-1]:.6f}")
        print(f"Validation loss: {metrics_with_compile['val_loss']:.6f}")
        print(f"Avg tokens/sec: {np.mean(metrics_with_compile['tokens_per_second']):.2f}")
        print(f"Total training time: {sum(metrics_with_compile['train_times']):.2f}s")
        
        # Calculate speedup or slowdown
        speedup = np.mean(metrics_with_compile['tokens_per_second']) / np.mean(metrics_without_compile['tokens_per_second'])
        print(f"\nSpeedup from torch.compile: {speedup:.2f}x")
        
        # Verify results are reasonable
        # Train and validation losses should be reasonable (not NaN or extremely high)
        self.assertFalse(np.isnan(metrics_with_compile['train_losses'][-1]))
        self.assertFalse(np.isnan(metrics_with_compile['val_loss']))
        
        # Loss should decrease over epochs
        self.assertLess(metrics_with_compile['train_losses'][-1], metrics_with_compile['train_losses'][0])
    
    def test_torch_compile_multiple_modes(self):
        """Test model training with different torch.compile modes."""
        # Skip if torch.compile not available or CUDA not available
        if not hasattr(torch, 'compile') or not torch.cuda.is_available():
            self.skipTest("torch.compile or CUDA not available")
        
        # Test available compilation modes
        modes = ['default', 'reduce-overhead', 'max-autotune']
        results = {}
        
        for mode in modes:
            print(f"\nTesting torch.compile with mode: {mode}")
            results[mode] = self._train_model(use_compile=True, compile_mode=mode, epochs=1)
            
            print(f"Mode {mode}:")
            print(f"Train loss: {results[mode]['train_losses'][-1]:.6f}")
            print(f"Tokens/sec: {results[mode]['tokens_per_second'][-1]:.2f}")
            print(f"Time: {results[mode]['train_times'][-1]:.2f}s")
        
        # Compare modes (no assertions, just output for analysis)
        best_mode = max(modes, key=lambda m: results[m]['tokens_per_second'][-1])
        print(f"\nBest performing mode: {best_mode} with {results[best_mode]['tokens_per_second'][-1]:.2f} tokens/sec")

if __name__ == "__main__":
    unittest.main() 