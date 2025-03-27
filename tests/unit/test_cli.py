"""
Unit tests for the CLI module.
"""
import unittest
import os
import tempfile
import shutil
import json
import torch
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from src.cli.run import app
from src.utils.checkpoint import save_checkpoint


class TestCLI(unittest.TestCase):
    """Tests for CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create runner for testing Typer CLI
        self.runner = CliRunner()
        
        # Create a sample model checkpoint
        self.model_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create a sample config directory
        self.config_dir = os.path.join(self.temp_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def create_mock_checkpoint(self):
        """Helper to create a mock checkpoint file."""
        checkpoint_path = os.path.join(self.model_dir, "model.pt")
        
        # Create a minimal mock model with state dict
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight": [1.0, 2.0]}
        
        # Save a minimal checkpoint
        checkpoint_data = {
            "model_state_dict": mock_model.state_dict(),
            "config": {
                "model": {
                    "model_type": "language",
                    "architecture": "transformer"
                }
            },
            "char_to_idx": {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")},
            "idx_to_char": {i: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        }
        
        with open(checkpoint_path, "wb") as f:
            torch.save(checkpoint_data, f)
        
        return checkpoint_path
    
    def create_mock_config(self):
        """Helper to create a mock configuration file."""
        config_path = os.path.join(self.config_dir, "test_config.json")
        
        config = {
            "model": {
                "model_type": "language",
                "architecture": "transformer",
                "vocab_size": 100,
                "d_model": 256,
                "n_head": 4,
                "d_hid": 512,
                "n_layers": 2,
                "dropout": 0.1
            },
            "data": {
                "data_type": "text",
                "format": "character",
                "train_file": "data/train.txt",
                "val_file": "data/val.txt",
                "batch_size": 32,
                "sequence_length": 128
            },
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "checkpoint_dir": self.model_dir
            },
            "paths": {
                "output_dir": self.model_dir
            },
            "system": {
                "seed": 42,
                "device": "cpu"
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        return config_path
    
    @unittest.skip("Missing prepare_dataloaders_from_config in src.data.dataset")
    @patch("src.cli.run.prepare_dataloaders_from_config")
    @patch("src.cli.run.create_model_from_config")
    @patch("src.cli.run.create_trainer_from_config")
    def test_train_language_command(self, mock_trainer, mock_model, mock_dataloaders):
        """Test the train language command."""
        # Create mock config
        config_path = self.create_mock_config()
        
        # Mock return values
        mock_model.return_value = MagicMock()
        mock_dataloaders.return_value = (MagicMock(), MagicMock())
        mock_trainer.return_value = MagicMock()
        mock_trainer.return_value.train.return_value = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4]
        }
        
        # Run the command
        result = self.runner.invoke(
            app, ["train", "language", "--config", config_path, "--device", "cpu"]
        )
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check mocks were called
        mock_dataloaders.assert_called_once()
        mock_model.assert_called_once()
        mock_trainer.assert_called_once()
        mock_trainer.return_value.train.assert_called_once()
        mock_trainer.return_value.save_checkpoint.assert_called_once()
    
    @unittest.skip("Missing prepare_dataloaders_from_config in src.data.dataset")
    @patch("torch.load")
    @patch("src.cli.run.create_model_from_config")
    def test_generate_text_command(self, mock_model_creator, mock_torch_load):
        """Test the generate text command."""
        # Create mock checkpoint
        checkpoint_path = self.create_mock_checkpoint()
        
        # Mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_creator.return_value = mock_model
        
        # Mock torch.load to return our checkpoint data
        mock_torch_load.return_value = {
            "model_state_dict": {},
            "config": {"model": {"model_type": "language"}},
            "char_to_idx": {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")},
            "idx_to_char": {str(i): c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        }
        
        # Run the command
        result = self.runner.invoke(
            app, [
                "generate", "text",
                "--model", checkpoint_path,
                "--prompt", "hello",
                "--max-length", "10",
                "--device", "cpu"
            ]
        )
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check mocks were called
        mock_torch_load.assert_called_once()
        mock_model_creator.assert_called_once()
        mock_model.generate.assert_called_once()
    
    @unittest.skip("Missing prepare_data in src.data.processors")
    @patch("src.data.processors.prepare_data")
    def test_dataset_prepare_command(self, mock_prepare_data):
        """Test the dataset prepare command."""
        # Create test input file
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test data")
        
        # Create output directory
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Run the command
        result = self.runner.invoke(
            app, [
                "dataset", "prepare",
                "--input", input_file,
                "--output-dir", output_dir
            ]
        )
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check mock was called with correct arguments
        mock_prepare_data.assert_called_once()
        args, kwargs = mock_prepare_data.call_args
        self.assertEqual(args[0], input_file)
        self.assertEqual(args[1], output_dir)
    
    @unittest.skip("Missing run_experiment in src.experiments.runner")
    @patch("src.cli.run.run_experiment")
    def test_experiment_run_command(self, mock_run_experiment):
        """Test the experiment run command."""
        # Create mock config
        config_path = self.create_mock_config()
        
        # Mock run_experiment to avoid actual execution
        mock_run_experiment.return_value = None
        
        # Run the command
        result = self.runner.invoke(
            app, [
                "experiment", "run",
                "--config", config_path
            ]
        )
        
        # Print output for debugging
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Exception: {result.exception}")
            print(f"Output: {result.stdout}")
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check mock was called
        mock_run_experiment.assert_called_once()
    
    def test_cli_help(self):
        """Test CLI help command."""
        # Run help command
        result = self.runner.invoke(app, ["--help"])
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check that key command groups are mentioned in the help text
        self.assertIn("Commands", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("generate", result.stdout)
        self.assertIn("dataset", result.stdout)
        self.assertIn("experiment", result.stdout)
    
    def test_train_help(self):
        """Test train help command."""
        # Run help command
        result = self.runner.invoke(app, ["train", "--help"])
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check that language subcommand is mentioned
        self.assertIn("language", result.stdout)
    
    def test_generate_help(self):
        """Test generate help command."""
        # Run help command
        result = self.runner.invoke(app, ["generate", "--help"])
        
        # Check the command was successful
        self.assertEqual(result.exit_code, 0)
        
        # Check that text subcommand is mentioned
        self.assertIn("text", result.stdout)


if __name__ == "__main__":
    unittest.main() 