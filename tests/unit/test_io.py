"""
Unit tests for the io utility module.
"""
import unittest
import os
import tempfile
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path

from src.utils.io import (
    create_output_dir,
    save_args,
    load_json,
    save_json,
    get_file_size,
    format_file_size
)


class TestIO(unittest.TestCase):
    """Tests for io.py functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_create_output_dir(self):
        """Test creating an output directory."""
        # Create an output directory with an experiment name
        experiment_name = "test_experiment"
        output_dir = create_output_dir(self.temp_dir, experiment_name)
        
        # Verify the directory was created
        self.assertTrue(os.path.exists(output_dir))
        
        # Verify the directory name contains the experiment name
        self.assertIn(experiment_name, os.path.basename(output_dir))
        
        # Create an output directory without an experiment name
        output_dir = create_output_dir(self.temp_dir)
        
        # Verify the directory was created
        self.assertTrue(os.path.exists(output_dir))
        
        # Verify the directory name contains a timestamp (should be numeric)
        basename = os.path.basename(output_dir)
        self.assertTrue(any(char.isdigit() for char in basename))
    
    def test_save_args_with_namespace(self):
        """Test saving arguments with an argparse.Namespace object."""
        # Create a sample args object
        args = argparse.Namespace()
        args.learning_rate = 0.001
        args.batch_size = 32
        args.epochs = 10
        args.model_name = "transformer"
        args.device = "cuda"
        
        # Path to save the arguments
        args_path = os.path.join(self.temp_dir, "args.json")
        
        # Save the arguments
        save_args(args, args_path)
        
        # Verify the arguments file exists
        self.assertTrue(os.path.exists(args_path))
        
        # Load the arguments and verify they match
        with open(args_path, "r") as f:
            loaded_args = json.load(f)
        
        self.assertEqual(loaded_args["learning_rate"], 0.001)
        self.assertEqual(loaded_args["batch_size"], 32)
        self.assertEqual(loaded_args["epochs"], 10)
        self.assertEqual(loaded_args["model_name"], "transformer")
        self.assertEqual(loaded_args["device"], "cuda")
    
    def test_save_args_with_dict(self):
        """Test saving arguments with a dictionary."""
        # Create a sample args dictionary
        args = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_name": "transformer",
            "device": "cuda",
            "config": {
                "hidden_dim": 256,
                "num_layers": 4
            }
        }
        
        # Path to save the arguments
        args_path = os.path.join(self.temp_dir, "args.json")
        
        # Save the arguments
        save_args(args, args_path)
        
        # Verify the arguments file exists
        self.assertTrue(os.path.exists(args_path))
        
        # Load the arguments and verify they match
        with open(args_path, "r") as f:
            loaded_args = json.load(f)
        
        self.assertEqual(loaded_args["learning_rate"], 0.001)
        self.assertEqual(loaded_args["batch_size"], 32)
        self.assertEqual(loaded_args["config"]["hidden_dim"], 256)
    
    def test_save_args_with_nonserializable(self):
        """Test saving arguments with non-serializable objects."""
        # Create a sample args dictionary with non-serializable objects
        args = {
            "timestamp": datetime.now(),
            "path": Path(self.temp_dir),
            "function": lambda x: x * 2
        }
        
        # Path to save the arguments
        args_path = os.path.join(self.temp_dir, "args.json")
        
        # Save the arguments
        save_args(args, args_path)
        
        # Verify the arguments file exists
        self.assertTrue(os.path.exists(args_path))
        
        # Load the arguments and verify they were converted to strings
        with open(args_path, "r") as f:
            loaded_args = json.load(f)
        
        self.assertTrue(isinstance(loaded_args["timestamp"], str))
        self.assertTrue(isinstance(loaded_args["path"], str))
        self.assertTrue(isinstance(loaded_args["function"], str))
    
    def test_load_and_save_json(self):
        """Test loading and saving JSON files."""
        # Create a sample JSON data
        data = {
            "name": "test",
            "values": [1, 2, 3, 4, 5],
            "nested": {
                "key1": "value1",
                "key2": "value2"
            }
        }
        
        # Path to save the JSON
        json_path = os.path.join(self.temp_dir, "data.json")
        
        # Save the JSON
        save_json(data, json_path)
        
        # Verify the JSON file exists
        self.assertTrue(os.path.exists(json_path))
        
        # Load the JSON and verify it matches
        loaded_data = load_json(json_path)
        
        self.assertEqual(loaded_data["name"], "test")
        self.assertEqual(loaded_data["values"], [1, 2, 3, 4, 5])
        self.assertEqual(loaded_data["nested"]["key1"], "value1")
        self.assertEqual(loaded_data["nested"]["key2"], "value2")
    
    def test_get_file_size(self):
        """Test getting file size."""
        # Create a sample file with known content
        file_path = os.path.join(self.temp_dir, "sample.txt")
        content = "a" * 1024  # 1 KB of data
        
        with open(file_path, "w") as f:
            f.write(content)
        
        # Get the file size
        size = get_file_size(file_path)
        
        # Verify the size is correct (should be 1024 bytes)
        self.assertEqual(size, 1024)
    
    def test_format_file_size(self):
        """Test formatting file size."""
        # Test different sizes
        self.assertEqual(format_file_size(500), "500.00 B")
        self.assertEqual(format_file_size(1024), "1.00 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.00 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024), "1.00 GB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024 * 1024), "1.00 TB")
        
        # Test non-exact sizes
        self.assertEqual(format_file_size(1500), "1.46 KB")
        self.assertEqual(format_file_size(1024 * 1024 * 2.5), "2.50 MB")


if __name__ == "__main__":
    unittest.main() 