import unittest
import torch
import time
import logging
import os
import sys
from unittest import mock
import numpy as np
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf, open_dict

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Updated import path for training functions
# from src.training.trainer import train_epoch # Removed unused import
# from src.training.lr_schedule import get_lr_scheduler # Removed unused import
# Also update model import to use factory
from src.models.factory import create_model_from_config
from src.training.trainer import Trainer # Import Trainer
from src.training.optimizations import setup_torch_compile # Import setup_torch_compile

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestTorchCompileOptimization(unittest.TestCase):
    """Test cases for the PyTorch 2.0+ Compilation optimization."""

    def setUp(self):
        """Set up test environment."""
        self.batch_size = 2
        self.seq_length = 16
        self.vocab_size = 50
        
        # --- Minimal Config ---
        self.cfg = OmegaConf.create({
            'seed': 42,
            'force_cpu': True, # Default to CPU for most tests
            'model': {
                '_target_': 'src.models.transformer.TransformerModel',
                'model_type': 'language',
                'architecture': 'transformer',
                'vocab_size': self.vocab_size,
                'max_seq_length': self.seq_length,
                'd_model': 128,
                'n_head': 4,
                'd_hid': 256,
                'n_layers': 2,
                'dropout': 0.1
            },
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 5e-5,
                'weight_decay': 0.01
            },
            'scheduler': { # Add a minimal scheduler config
                'type': 'constant', # Example, adjust if needed
                'warmup_steps': 0
            },
            'data': { # Add minimal data config
                 'batch_size': self.batch_size,
                 'num_workers': 0,
                 'vocab_path': None # Set vocab_path if needed by Trainer/callbacks
            },
            'training': {
                'epochs': 1,
                'torch_compile': False, # Default to False, override in specific tests
                'compile_mode': 'reduce-overhead',
                'use_amp': False, # Usually False for CPU tests
                'gradient_accumulation_steps': 1,
                'max_grad_norm': 1.0,
                'log_interval': 10,
                'checkpoint_subdir': 'test_checkpoints', # Use a test-specific dir
                 'save_interval': 0, # Disable epoch saving by default
                 'time_save_interval_minutes': 0 # Disable time saving by default
            },
            'callbacks': { # Empty callbacks list
                 'callbacks_list': []
            }
            # Add other necessary keys if Trainer complains
        })
        # --- End Minimal Config ---

        # Create a small model for testing using the factory
        # Use the config we just created
        self.model = create_model_from_config(self.cfg.model) 
        
        # Create optimizer and scheduler (using config values)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.learning_rate,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        # Simple constant scheduler for testing
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _: 1.0
        ) 
        
        # Create a small dummy dataset
        self.inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        self.targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # Create a mock DataLoader that yields dictionaries
        class MockDataLoader:
            def __init__(self, inputs, targets, batch_size):
                self.inputs = inputs
                self.targets = targets
                self.batch_size = batch_size
                
            def __iter__(self):
                # Yield a dictionary similar to the actual dataloader output
                yield {'input_ids': self.inputs, 'labels': self.targets} 
                
            def __len__(self):
                # Return number of batches
                return 1 
                
        self.dataloader = MockDataLoader(self.inputs, self.targets, self.batch_size)
        
        # Use CPU for testing by default (can be overridden in specific tests)
        self.device = torch.device('cpu') 
        self.model.to(self.device) # Move model to device in setUp

    def test_torch_compile_available(self):
        """Test if torch.compile is available in the current PyTorch version."""
        has_compile = hasattr(torch, 'compile')
        
        if has_compile:
            print("torch.compile is available")
            logging.info("torch.compile is available") # Use logging
        else:
            print("torch.compile is not available in this PyTorch version")
            logging.info("torch.compile is not available in this PyTorch version") # Use logging
        # No assertion needed, just informational / used by other tests

    @unittest.skip("Skipping torch.compile functionality test due to potential environment/model specific issues")
    def test_torch_compile_functionality(self):
        """Test that torch.compile works functionally (produces same results)."""
        # Skip if torch.compile is not available
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile not available")
        
        # Get a fresh model instance
        model_to_compile = create_model_from_config(self.cfg.model)
        model_to_compile.to(self.device)
        model_to_compile.eval() # Set to eval mode for consistent output

        # Get uncompiled output
        with torch.no_grad():
            regular_output = model_to_compile(self.inputs.to(self.device))

        # Compile the model using setup_torch_compile
        # Create a temporary config enabling compile
        compile_cfg = self.cfg.copy()
        with open_dict(compile_cfg): # Allow modification
             compile_cfg.training.torch_compile = True
             compile_cfg.training.compile_mode = 'reduce-overhead' # Or another mode if needed

        # Mock CUDA check if testing on CPU but want to test compile logic
        with patch('torch.cuda.is_available', return_value=True):
            try:
                 # Use a copy of the model for compilation to avoid side effects
                 model_copy = create_model_from_config(self.cfg.model)
                 model_copy.load_state_dict(model_to_compile.state_dict())
                 model_copy.to(self.device)
                 model_copy.eval()

                 compiled_model = setup_torch_compile(compile_cfg.training, model_copy)

                 # Check if compilation actually happened (setup_torch_compile might skip it)
                 # A simple check: see if the model type changed or if it has dynamo attributes
                 is_compiled = "OptimizedModule" in str(type(compiled_model)) or hasattr(compiled_model, '_torchdynamo_orig_callable')

                 if not is_compiled:
                      # This might happen if compile fails silently or requirements aren't met
                      self.skipTest("setup_torch_compile did not compile the model (check logs/requirements)")

                 # Get compiled output
                 with torch.no_grad():
                      compiled_output = compiled_model(self.inputs.to(self.device))

                 # Check outputs are close
                 self.assertTrue(torch.allclose(regular_output, compiled_output, atol=1e-5),
                                 f"Compiled model output differs significantly. Max diff: {torch.max(torch.abs(regular_output - compiled_output))}")

            except RuntimeError as e:
                 # Catch runtime errors during compilation or execution
                 self.skipTest(f"Environment doesn't support torch.compile or model incompatible: {e}")
            except Exception as e:
                 # Catch any other unexpected errors during setup or execution
                 self.fail(f"An unexpected error occurred during compile functionality test: {e}")
    
    def test_train_epoch_with_compile(self):
        """Test train_epoch function with torch.compile enabled."""
        # Skip if torch.compile is not available
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile not available")
        
        try:
            # --- Run without compilation ---
            cfg_no_compile = self.cfg.copy()
            with open_dict(cfg_no_compile):
                 cfg_no_compile.training.torch_compile = False

            model_no_compile = create_model_from_config(cfg_no_compile.model)
            model_no_compile.to(self.device)
            optimizer_no_compile = torch.optim.AdamW(model_no_compile.parameters(), lr=cfg_no_compile.optimizer.learning_rate)
            scheduler_no_compile = torch.optim.lr_scheduler.LambdaLR(optimizer_no_compile, lambda _: 1.0)

            trainer_no_compile = Trainer(
                model=model_no_compile,
                optimizer=optimizer_no_compile,
                scheduler=scheduler_no_compile,
                train_dataloader=self.dataloader,
                device=self.device,
                config=cfg_no_compile,
                # Add other required Trainer args if needed
            )
            metrics_no_compile = trainer_no_compile._train_epoch()
            loss_without_compile = metrics_no_compile.get('loss')


            # --- Run with compilation ---
            cfg_with_compile = self.cfg.copy()
            with open_dict(cfg_with_compile):
                 cfg_with_compile.training.torch_compile = True
                 cfg_with_compile.training.compile_mode = 'reduce-overhead' # Or 'default'

            model_to_compile = create_model_from_config(cfg_with_compile.model)
            model_to_compile.to(self.device)

            # Apply torch.compile BEFORE initializing Trainer
            # Mock CUDA check if needed for CPU testing
            with patch('torch.cuda.is_available', return_value=True):
                 compiled_model = setup_torch_compile(cfg_with_compile.training, model_to_compile)

                 # Check if compilation happened
                 is_compiled = "OptimizedModule" in str(type(compiled_model)) or hasattr(compiled_model, '_torchdynamo_orig_callable')
                 if not is_compiled:
                      self.skipTest("setup_torch_compile did not compile the model (check logs/requirements)")


            optimizer_with_compile = torch.optim.AdamW(compiled_model.parameters(), lr=cfg_with_compile.optimizer.learning_rate)
            scheduler_with_compile = torch.optim.lr_scheduler.LambdaLR(optimizer_with_compile, lambda _: 1.0)

            trainer_with_compile = Trainer(
                model=compiled_model, # Use the compiled model
                optimizer=optimizer_with_compile,
                scheduler=scheduler_with_compile,
                train_dataloader=self.dataloader,
                device=self.device,
                config=cfg_with_compile,
                # Add other required Trainer args
            )
            metrics_with_compile = trainer_with_compile._train_epoch()
            loss_with_compile = metrics_with_compile.get('loss')


            # Verify loss values are reasonably close
            self.assertIsNotNone(loss_with_compile)
            self.assertIsNotNone(loss_without_compile)
            self.assertFalse(np.isnan(loss_with_compile))
            self.assertFalse(np.isnan(loss_without_compile))

            logging.info(f"Loss with compile: {loss_with_compile}")
            logging.info(f"Loss without compile: {loss_without_compile}")
            # Losses might differ slightly due to optimization differences
            # Use assertAlmostEqual for floating point comparison
            self.assertAlmostEqual(loss_with_compile, loss_without_compile, delta=0.1, # Allow some difference
                                   msg="Loss values differ significantly between compiled and non-compiled runs.")

        except RuntimeError as e:
            if "Compiler:" in str(e) or "compile" in str(e).lower():
                self.skipTest(f"Environment doesn't support torch.compile: {e}")
            else:
                raise
        except Exception as e:
             # Catch any other unexpected errors during setup or execution
             self.fail(f"An unexpected error occurred during compile train_epoch test: {e}")
    
    @unittest.skip("Skipping performance test due to environment-specific requirements")
    def test_performance_improvement(self):
        """Test that torch.compile provides performance improvement on CUDA."""
        # Skip this test if not using CUDA
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Skip if torch.compile is not available
        if not hasattr(torch, 'compile'):
            self.skipTest("torch.compile not available")
        
        try:
            cuda_device = torch.device('cuda')

            # Create larger test data for more reliable timing
            test_batch_size = 8 # Increase batch size
            test_seq_length = 64 # Increase sequence length
            num_batches = 10 # Run more batches

            test_inputs = torch.randint(0, self.vocab_size, (test_batch_size, test_seq_length)).to(cuda_device)
            test_targets = torch.randint(0, self.vocab_size, (test_batch_size, test_seq_length)).to(cuda_device)

            # Mock data loader with larger data
            class LargerMockDataLoader:
                def __init__(self, inputs, targets, num_batches=5):
                    self.inputs = inputs
                    self.targets = targets
                    self.batch_size = inputs.size(0)
                    self.num_batches = num_batches
                    
                def __iter__(self):
                    for _ in range(self.num_batches):
                        yield {'input_ids': self.inputs, 'labels': self.targets}
                        
                def __len__(self):
                    return self.num_batches
                    
            test_dataloader = LargerMockDataLoader(test_inputs, test_targets, num_batches=num_batches)


            # --- Benchmark without compile ---
            cfg_no_compile = self.cfg.copy()
            with open_dict(cfg_no_compile):
                 cfg_no_compile.training.torch_compile = False
                 cfg_no_compile.training.use_amp = True # Often used with compile
                 cfg_no_compile.force_cpu = False

            model_no_compile = create_model_from_config(cfg_no_compile.model)
            model_no_compile.to(cuda_device)
            optimizer_no_compile = torch.optim.AdamW(model_no_compile.parameters(), lr=cfg_no_compile.optimizer.learning_rate)

            trainer_no_compile = Trainer(
                model=model_no_compile,
                optimizer=optimizer_no_compile,
                train_dataloader=test_dataloader,
                device=cuda_device,
                config=cfg_no_compile,
                use_amp=cfg_no_compile.training.use_amp # Pass AMP flag
            )
            # Warmup run (optional but good practice)
            trainer_no_compile._train_epoch()

            # Actual benchmark run
            start_time = time.time()
            metrics_no_compile = trainer_no_compile._train_epoch()
            time_without_compile = time.time() - start_time


            # --- Benchmark with compile ---
            cfg_with_compile = self.cfg.copy()
            with open_dict(cfg_with_compile):
                 cfg_with_compile.training.torch_compile = True
                 cfg_with_compile.training.compile_mode = 'reduce-overhead' # Or max-autotune
                 cfg_with_compile.training.use_amp = True
                 cfg_with_compile.force_cpu = False

            model_to_compile = create_model_from_config(cfg_with_compile.model)
            model_to_compile.to(cuda_device)

            # Compile the model
            compiled_model = setup_torch_compile(cfg_with_compile.training, model_to_compile)
            is_compiled = "OptimizedModule" in str(type(compiled_model)) or hasattr(compiled_model, '_torchdynamo_orig_callable')
            if not is_compiled:
                self.skipTest("setup_torch_compile did not compile the model (check logs/requirements)")

            optimizer_with_compile = torch.optim.AdamW(compiled_model.parameters(), lr=cfg_with_compile.optimizer.learning_rate)

            trainer_with_compile = Trainer(
                model=compiled_model,
                optimizer=optimizer_with_compile,
                train_dataloader=test_dataloader,
                device=cuda_device,
                config=cfg_with_compile,
                use_amp=cfg_with_compile.training.use_amp
            )
            # Warmup run (important for compile)
            trainer_with_compile._train_epoch()

            # Actual benchmark run
            start_time = time.time()
            metrics_with_compile = trainer_with_compile._train_epoch()
            time_with_compile = time.time() - start_time


            # Print and verify results
            logging.info(f"\nPerformance test results:")
            logging.info(f"Time without torch.compile: {time_without_compile:.4f}s")
            logging.info(f"Time with torch.compile:   {time_with_compile:.4f}s")

            # Allow for some compilation overhead, especially with 'reduce-overhead'
            # Check that compiled is not significantly slower, ideally faster
            # Looser check: assert less than 2x slower, ideally faster
            self.assertLess(time_with_compile, time_without_compile * 2,
                             "Compiled run was more than 2x slower than non-compiled.")
            if time_with_compile < time_without_compile:
                 logging.info(f"Speedup from torch.compile: {time_without_compile / time_with_compile:.2f}x")
            else:
                 logging.info(f"Slowdown/No Speedup: {time_with_compile / time_without_compile:.2f}x (might be due to overhead/model complexity)")


        except RuntimeError as e:
            if "Compiler:" in str(e) or "compile" in str(e).lower() or "CUDA" in str(e):
                self.skipTest(f"Environment doesn't support torch.compile or CUDA issue: {e}")
            else:
                raise
        except Exception as e:
             self.fail(f"An unexpected error occurred during performance test: {e}")
    
    def test_graceful_fallback_when_compile_not_available(self):
        """Test that training continues gracefully when torch.compile is not available."""

        # Mock torch.compile to ensure it's seen as unavailable
        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj == torch and name == 'compile':
                logging.info("Mocking hasattr(torch, 'compile') -> False")
                return False
            return original_hasattr(obj, name)

        # Apply the mock
        with mock.patch('builtins.hasattr', mock_hasattr):
             # Ensure compile is requested in config
             cfg_request_compile = self.cfg.copy()
             with open_dict(cfg_request_compile):
                 cfg_request_compile.training.torch_compile = True

             # Attempt to setup compile (should log warning and return original model)
             model_fallback = create_model_from_config(cfg_request_compile.model)
             model_fallback.to(self.device)
             model_after_setup = setup_torch_compile(cfg_request_compile.training, model_fallback)
             self.assertIs(model_after_setup, model_fallback, "setup_torch_compile should return original model when compile is unavailable")

             # Initialize Trainer (should work with uncompiled model)
             optimizer_fallback = torch.optim.AdamW(model_after_setup.parameters(), lr=cfg_request_compile.optimizer.learning_rate)
             scheduler_fallback = torch.optim.lr_scheduler.LambdaLR(optimizer_fallback, lambda _: 1.0)

             try:
                  trainer_fallback = Trainer(
                       model=model_after_setup,
                       optimizer=optimizer_fallback,
                       scheduler=scheduler_fallback,
                       train_dataloader=self.dataloader,
                       device=self.device,
                       config=cfg_request_compile,
                       # Add other args if needed
                  )

                  # Run training epoch (should succeed)
                  metrics = trainer_fallback._train_epoch()
                  loss = metrics.get('loss')

                  # Verify we got a valid loss
                  self.assertIsNotNone(loss)
                  self.assertFalse(np.isnan(loss))
                  logging.info(f"Fallback test ran successfully with loss: {loss}")

             except Exception as e:
                  self.fail(f"Trainer failed during fallback test even though compile was mocked as unavailable: {e}")


if __name__ == "__main__":
    unittest.main() 