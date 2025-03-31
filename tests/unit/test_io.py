"""
Unit tests for the io utility module.
"""
import sys
import os
import tempfile
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.io import (
    ensure_directory,
    load_json,
    save_json,
    get_file_size,
    format_file_size,
    create_output_dir
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
        
        # Test creating directory without explicit experiment name (should still work)
        # The function should handle default naming if needed, let's use a simple default
        default_experiment_name = "default_run"
        output_dir_default = create_output_dir(self.temp_dir, default_experiment_name)
        self.assertTrue(os.path.exists(output_dir_default))
        self.assertIn(default_experiment_name, os.path.basename(output_dir_default))
    
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