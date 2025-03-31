# tests/unit/test_data_processors.py
import unittest
import os
import tempfile
import pickle
import shutil
import numpy as np

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.processors import prepare_text_data, prepare_data, split_data

class TestDataProcessors(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "processed")

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_prepare_text_data_character(self):
        """Test prepare_text_data with character format and splitting."""
        # Create a dummy input file
        input_content = "abcdefghijklmnopqrstuvwxyz" * 4 # 104 chars
        input_file_path = os.path.join(self.test_dir, "input.txt")
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(input_content)
        
        # Expected results for character processing
        expected_chars = sorted(list(set(input_content)))
        expected_vocab_size = len(expected_chars)
        expected_char_to_idx = {ch: i for i, ch in enumerate(expected_chars)}
        expected_idx_to_char = {i: ch for i, ch in enumerate(expected_chars)}
        expected_token_ids = np.array([expected_char_to_idx.get(c, 0) for c in input_content], dtype=np.uint16)
        
        # Config with explicit split ratios (e.g., 70/15/15 for easier length check)
        config = {
             "data": {"split_ratios": [0.7, 0.15, 0.15]},
             "seed": 42 # Use fixed seed for reproducible split
        }
        total_len = len(expected_token_ids)
        expected_train_len = int(total_len * 0.7)
        expected_val_len = int(total_len * 0.15)
        # Test length adjusts for rounding
        expected_test_len = total_len - expected_train_len - expected_val_len 

        # Run the function
        output_paths = prepare_text_data(input_file_path, self.output_dir, config=config)

        # Assert output files exist
        self.assertIsInstance(output_paths, dict)
        self.assertIn("train", output_paths)
        self.assertIn("val", output_paths)
        self.assertIn("test", output_paths)
        train_path = output_paths["train"]
        val_path = output_paths["val"]
        test_path = output_paths["test"]
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(val_path))
        self.assertTrue(os.path.exists(test_path))

        # Load and verify each split file
        for split_name, path, expected_len in zip(["train", "val", "test"], 
                                                  [train_path, val_path, test_path], 
                                                  [expected_train_len, expected_val_len, expected_test_len]):
            with open(path, "rb") as f:
                loaded_data = pickle.load(f)
            
            self.assertIsInstance(loaded_data, dict, f"Content of {split_name} split is not a dict")
            # Check common char metadata
            self.assertEqual(loaded_data.get("chars"), expected_chars, f"Incorrect chars in {split_name}")
            self.assertEqual(loaded_data.get("char_to_idx"), expected_char_to_idx, f"Incorrect char_to_idx in {split_name}")
            self.assertEqual(loaded_data.get("idx_to_char"), expected_idx_to_char, f"Incorrect idx_to_char in {split_name}")
            self.assertEqual(loaded_data.get("vocab_size"), expected_vocab_size, f"Incorrect vocab_size in {split_name}")
            # Check token IDs
            self.assertIn("token_ids", loaded_data, f"token_ids missing in {split_name}")
            self.assertIsInstance(loaded_data["token_ids"], np.ndarray, f"token_ids in {split_name} is not a numpy array")
            self.assertEqual(len(loaded_data["token_ids"]), expected_len, f"Incorrect number of tokens in {split_name}")
            self.assertEqual(loaded_data["token_ids"].dtype, np.uint16, f"Incorrect dtype for token_ids in {split_name}")
            # Note: Verifying the *exact* tokens requires replicating the shuffle/split logic, 
            # which is tested separately in test_split_data. Here we focus on length and metadata.

    def test_prepare_text_data_unsupported_format(self):
        """Test prepare_text_data raises error for unsupported format."""
        input_file_path = os.path.join(self.test_dir, "input.txt")
        with open(input_file_path, "w") as f:
            f.write("test")
            
        bad_config = {"data": {"format": "token"}} # Assuming "token" is unsupported
        
        with self.assertRaisesRegex(ValueError, "Unsupported text format: token"):
            prepare_text_data(input_file_path, self.output_dir, config=bad_config)

    @unittest.mock.patch('transformers.AutoTokenizer.from_pretrained')
    def test_prepare_text_data_tokenizer(self, mock_from_pretrained):
        """Test prepare_text_data with a standard tokenizer."""
        # Mock the tokenizer object
        mock_tokenizer = unittest.mock.MagicMock()
        mock_tokenizer.vocab_size = 10000 # Example vocab size
        # Configure encode to return a list of integers
        mock_tokenizer.encode.return_value = list(range(500)) # Example token IDs
        mock_from_pretrained.return_value = mock_tokenizer

        # Create dummy input file
        input_content = "This is text to be tokenized." * 5
        input_file_path = os.path.join(self.test_dir, "input.txt")
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(input_content)
        
        tokenizer_name = "mock-tokenizer"
        config = {
             "data": {"tokenizer_name": tokenizer_name, "split_ratios": [0.7, 0.15, 0.15]},
             "seed": 42
        }
        total_len = len(mock_tokenizer.encode.return_value)
        expected_train_len = int(total_len * 0.7)
        expected_val_len = int(total_len * 0.15)
        expected_test_len = total_len - expected_train_len - expected_val_len

        # Run the function
        output_paths = prepare_text_data(input_file_path, self.output_dir, config=config)

        # Assert AutoTokenizer was called
        mock_from_pretrained.assert_called_once_with(tokenizer_name)
        # Assert tokenizer.encode was called
        mock_tokenizer.encode.assert_called_once_with(input_content)

        # Assert output files exist
        self.assertIsInstance(output_paths, dict)
        self.assertIn("train", output_paths)
        self.assertIn("val", output_paths)
        self.assertIn("test", output_paths)

        # Load and verify train split file (as an example)
        with open(output_paths["train"], "rb") as f:
            loaded_data = pickle.load(f)
        
        self.assertIsInstance(loaded_data, dict, "Train split content is not a dict")
        self.assertEqual(loaded_data.get("tokenizer_name"), tokenizer_name)
        self.assertEqual(loaded_data.get("vocab_size"), mock_tokenizer.vocab_size)
        self.assertIn("token_ids", loaded_data)
        self.assertIsInstance(loaded_data["token_ids"], np.ndarray)
        self.assertEqual(len(loaded_data["token_ids"]), expected_train_len)
        self.assertEqual(loaded_data["token_ids"].dtype, np.uint16)
        # Check that original text/char maps are NOT present
        self.assertNotIn("text", loaded_data)
        self.assertNotIn("chars", loaded_data)
        self.assertNotIn("char_to_idx", loaded_data)
        self.assertNotIn("idx_to_char", loaded_data)

    # --- Tests for prepare_data (dispatcher) --- 

    @unittest.mock.patch('src.data.processors.prepare_text_data')
    def test_prepare_data_calls_text(self, mock_prepare_text):
        """Test prepare_data calls prepare_text_data for .txt files."""
        input_file_path = os.path.join(self.test_dir, "input.txt")
        with open(input_file_path, "w") as f: f.write("test")
        
        prepare_data(input_file_path, self.output_dir, config=None)
        mock_prepare_text.assert_called_once_with(input_file_path, self.output_dir, None)

    @unittest.mock.patch('src.data.processors.prepare_json_data')
    def test_prepare_data_calls_json(self, mock_prepare_json):
        """Test prepare_data calls prepare_json_data for .json files."""
        input_file_path = os.path.join(self.test_dir, "input.json")
        with open(input_file_path, "w") as f: f.write("{}") # Minimal valid JSON
        
        prepare_data(input_file_path, self.output_dir, config=None)
        mock_prepare_json.assert_called_once_with(input_file_path, self.output_dir, None)
        
    @unittest.mock.patch('src.data.processors.prepare_json_data')
    def test_prepare_data_calls_jsonl(self, mock_prepare_json):
        """Test prepare_data calls prepare_json_data for .jsonl files."""
        input_file_path = os.path.join(self.test_dir, "input.jsonl")
        with open(input_file_path, "w") as f: f.write("{}\n") # Minimal valid JSONL
        
        prepare_data(input_file_path, self.output_dir, config=None)
        mock_prepare_json.assert_called_once_with(input_file_path, self.output_dir, None)

    def test_prepare_data_raises_not_implemented(self):
        """Test prepare_data raises NotImplementedError for image/audio."""
        img_path = os.path.join(self.test_dir, "input.png")
        with open(img_path, "w") as f: f.write("") # Empty file is enough
        audio_path = os.path.join(self.test_dir, "input.wav")
        with open(audio_path, "w") as f: f.write("")

        with self.assertRaisesRegex(NotImplementedError, "Image data processing not yet implemented"):
            prepare_data(img_path, self.output_dir, config=None)
            
        with self.assertRaisesRegex(NotImplementedError, "Audio data processing not yet implemented"):
            prepare_data(audio_path, self.output_dir, config=None)

    def test_prepare_data_raises_value_error_unknown_type(self):
        """Test prepare_data raises ValueError for unknown file types."""
        unknown_path = os.path.join(self.test_dir, "input.unknown")
        with open(unknown_path, "w") as f: f.write("")
        
        with self.assertRaisesRegex(ValueError, "Cannot infer data type from file extension"):
            prepare_data(unknown_path, self.output_dir, config=None)

    # --- Test for split_data utility --- 
    def test_split_data(self):
        """Test the split_data utility function."""
        data = list(range(100)) # Simple list of 100 items
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

        train_set, val_set, test_set = split_data(
            data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=42
        )

        # Check lengths
        self.assertEqual(len(train_set), int(100 * train_ratio))
        self.assertEqual(len(val_set), int(100 * val_ratio))
        self.assertEqual(len(test_set), int(100 * test_ratio))

        # Check for overlap (should be none due to shuffling)
        self.assertEqual(len(set(train_set).intersection(set(val_set))), 0)
        self.assertEqual(len(set(train_set).intersection(set(test_set))), 0)
        self.assertEqual(len(set(val_set).intersection(set(test_set))), 0)

        # Check that all original elements are present
        combined_set = set(train_set).union(set(val_set)).union(set(test_set))
        self.assertEqual(combined_set, set(data))
        
    def test_split_data_ratios_error(self):
        """Test split_data raises error if ratios don't sum to 1."""
        data = list(range(10))
        with self.assertRaisesRegex(AssertionError, "Ratios must sum to 1"):
            split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)


# Placeholder for other tests (prepare_data dispatch, split_data)

if __name__ == '__main__':
    unittest.main() 