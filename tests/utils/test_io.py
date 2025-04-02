import os
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from craft.utils import io


# === Tests for ensure_directory ===

def test_ensure_directory_creates_new(tmp_path):
    """Test that ensure_directory creates a directory if it doesn't exist."""
    new_dir = tmp_path / "new_folder"
    assert not new_dir.exists()
    io.ensure_directory(str(new_dir))
    assert new_dir.exists()
    assert new_dir.is_dir()

def test_ensure_directory_existing(tmp_path):
    """Test that ensure_directory does nothing if the directory already exists."""
    existing_dir = tmp_path / "existing_folder"
    existing_dir.mkdir()
    assert existing_dir.exists()
    # Should run without error
    io.ensure_directory(str(existing_dir))
    assert existing_dir.exists()
    assert existing_dir.is_dir()

def test_ensure_directory_nested(tmp_path):
    """Test that ensure_directory creates nested directories."""
    nested_dir = tmp_path / "parent" / "child"
    assert not nested_dir.exists()
    io.ensure_directory(str(nested_dir))
    assert nested_dir.exists()
    assert nested_dir.is_dir()

# === Tests for load_json ===

def test_load_json_success(tmp_path):
    """Test loading a valid JSON file."""
    data_to_save = {"key": "value", "number": 123}
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data_to_save), encoding='utf-8')

    loaded_data = io.load_json(str(file_path))
    assert loaded_data == data_to_save

def test_load_json_file_not_found(tmp_path):
    """Test loading a non-existent JSON file."""
    non_existent_path = tmp_path / "not_real.json"
    with pytest.raises(FileNotFoundError):
        io.load_json(str(non_existent_path))

def test_load_json_invalid_json(tmp_path):
    """Test loading a file with invalid JSON content."""
    file_path = tmp_path / "invalid.json"
    file_path.write_text("this is not json", encoding='utf-8')

    with pytest.raises(json.JSONDecodeError):
        io.load_json(str(file_path))

# === Tests for save_json ===

def test_save_json_success(tmp_path):
    """Test saving a dictionary to a JSON file."""
    data_to_save = {"name": "test", "value": [1, 2, 3]}
    file_path = tmp_path / "output.json"

    io.save_json(data_to_save, str(file_path))

    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    assert loaded_data == data_to_save

def test_save_json_creates_parent_dir(tmp_path):
    """Test that save_json creates parent directories if they don't exist."""
    data_to_save = {"nested": True}
    file_path = tmp_path / "new_parent" / "output.json"

    assert not file_path.parent.exists()
    io.save_json(data_to_save, str(file_path))

    assert file_path.parent.exists()
    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    assert loaded_data == data_to_save

def test_save_json_indentation(tmp_path):
    """Test saving JSON with specific indentation."""
    data_to_save = {"a": 1, "b": 2}
    file_path_indent2 = tmp_path / "indent2.json"
    file_path_indent4 = tmp_path / "indent4.json" # Default

    io.save_json(data_to_save, str(file_path_indent2), indent=2)
    io.save_json(data_to_save, str(file_path_indent4)) # Default indent=4

    content_indent2 = file_path_indent2.read_text(encoding='utf-8')
    content_indent4 = file_path_indent4.read_text(encoding='utf-8')

    expected_indent2 = '{\n  "a": 1,\n  "b": 2\n}'
    expected_indent4 = '{\n    "a": 1,\n    "b": 2\n}'
    assert content_indent2 == expected_indent2
    assert content_indent4 == expected_indent4

@patch('craft.utils.io.open')
def test_save_json_exception(mock_open, tmp_path, caplog):
    """Test that save_json handles exceptions during file writing."""
    data_to_save = {"error": "case"}
    file_path = tmp_path / "error_output.json"

    # Simulate an OSError when trying to write
    mock_file = MagicMock()
    mock_file.__enter__.return_value.write.side_effect = OSError("Disk full simulation")
    mock_open.return_value = mock_file

    caplog.set_level(logging.ERROR)
    with pytest.raises(OSError):
        io.save_json(data_to_save, str(file_path))

    # Check that the error was logged
    assert f"Failed to save JSON to {file_path}" in caplog.text
    assert "Disk full simulation" in caplog.text

# === Tests for get_file_size ===

def test_get_file_size_existing(tmp_path):
    """Test getting the size of an existing file."""
    file_path = tmp_path / "file.txt"
    content = "Hello, world!"
    file_path.write_text(content, encoding='utf-8')
    expected_size = len(content.encode('utf-8'))

    assert io.get_file_size(str(file_path)) == expected_size

def test_get_file_size_empty(tmp_path):
    """Test getting the size of an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.touch()

    assert io.get_file_size(str(file_path)) == 0

def test_get_file_size_non_existent(tmp_path, caplog):
    """Test getting the size of a non-existent file."""
    non_existent_path = tmp_path / "not_real.txt"

    caplog.set_level(logging.ERROR)
    size = io.get_file_size(str(non_existent_path))

    assert size == 0
    assert f"Failed to get size of {non_existent_path}" in caplog.text

# === Tests for format_file_size ===

@pytest.mark.parametrize(
    "size_bytes, expected_str",
    [
        (0, "0B"),
        (100, "100.00 B"),
        (1023, "1023.00 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1024 * 1024 - 1, "1024.00 KB"), # Should be slightly less than 1MB
        (1024 * 1024, "1.00 MB"),
        (1.5 * 1024 * 1024, "1.50 MB"),
        (1024 * 1024 * 1024, "1.00 GB"),
        (10 * 1024 * 1024 * 1024, "10.00 GB"),
        (1024 ** 4, "1.00 TB"),
        (1024 ** 5, "1.00 PB"),
        (1024 ** 6, "1024.00 PB"), # Max unit is PB
    ]
)
def test_format_file_size(size_bytes, expected_str):
    """Test formatting file sizes into human-readable strings."""
    assert io.format_file_size(size_bytes) == expected_str

# === Tests for create_output_dir ===

def test_create_output_dir_new(tmp_path):
    """Test creating a new output directory."""
    base_dir = tmp_path / "base"
    exp_name = "my_experiment"
    expected_path = base_dir / exp_name

    assert not expected_path.exists()
    returned_path_str = io.create_output_dir(str(base_dir), exp_name)
    returned_path = Path(returned_path_str)

    assert expected_path.exists()
    assert expected_path.is_dir()
    assert returned_path.is_absolute()
    # Resolve potential symlinks/relative components for comparison
    assert returned_path.resolve() == expected_path.resolve()

def test_create_output_dir_existing_base(tmp_path):
    """Test creating an output directory when the base directory exists."""
    base_dir = tmp_path / "existing_base"
    base_dir.mkdir()
    exp_name = "another_run"
    expected_path = base_dir / exp_name

    assert base_dir.exists()
    assert not expected_path.exists()

    returned_path_str = io.create_output_dir(str(base_dir), exp_name)
    returned_path = Path(returned_path_str)

    assert expected_path.exists()
    assert expected_path.is_dir()
    assert returned_path.is_absolute()
    assert returned_path.resolve() == expected_path.resolve()

def test_create_output_dir_already_exists(tmp_path):
    """Test creating an output directory when the target directory already exists."""
    base_dir = tmp_path / "base_final"
    exp_name = "final_exp"
    target_dir = base_dir / exp_name
    target_dir.mkdir(parents=True)

    assert target_dir.exists()

    returned_path_str = io.create_output_dir(str(base_dir), exp_name)
    returned_path = Path(returned_path_str)

    assert target_dir.exists() # Should still exist
    assert target_dir.is_dir()
    assert returned_path.is_absolute()
    assert returned_path.resolve() == target_dir.resolve()

# (Add tests for create_output_dir here) 