import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarking.runner import BenchmarkRunner

class MockArgs:
    """Mock arguments for testing."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TestBenchmarkRunner(unittest.TestCase):
    """Unit tests for the benchmark runner module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up mock arguments
        self.args = MockArgs(
            output_dir=self.temp_dir,
            benchmarks=None,  # Run all benchmarks
            baseline_file=None,
            log_level="INFO",
            data_dir=os.path.join(self.temp_dir, "data"),
            model_checkpoint=None,
            iterations=1
        )
        
        # Create data directory
        os.makedirs(self.args.data_dir, exist_ok=True)
        
        # Create benchmark directory
        self.benchmark_dir = os.path.join(self.temp_dir, "benchmarks")
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        # Create sample benchmark results
        self.sample_results = {
            "metadata": {
                "timestamp": "2023-01-01T00:00:00",
                "system_info": {
                    "platform": "test",
                    "python_version": "3.8.0"
                }
            },
            "benchmarks": {
                "training_speed": {
                    "status": "passed",
                    "execution_time": 10.5,
                    "results": {
                        "throughput": 1000,
                        "memory_usage": 2.5
                    }
                },
                "inference_performance": {
                    "status": "passed",
                    "execution_time": 5.2,
                    "results": {
                        "latency": 0.01,
                        "tokens_per_second": 500
                    }
                }
            }
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('benchmarking.runner.importlib.import_module')
    @patch('benchmarking.runner.os.listdir')
    def test_discover_benchmarks(self, mock_listdir, mock_import):
        """Test discovering benchmarks."""
        # Create a mock runner
        runner = BenchmarkRunner(self.args)
        
        # Mock listdir to return benchmark files
        mock_listdir.return_value = [
            "training_speed.py",
            "inference_performance.py",
            "__init__.py",  # Should be ignored
            "test_helper.py"  # Should be included
        ]
        
        # Call method to discover benchmarks
        benchmarks = runner._discover_benchmarks()
        
        # Check that the correct benchmarks were discovered
        self.assertEqual(len(benchmarks), 3)
        self.assertIn("training_speed", benchmarks)
        self.assertIn("inference_performance", benchmarks)
        self.assertIn("test_helper", benchmarks)
    
    @patch('benchmarking.runner.BenchmarkRunner._discover_benchmarks')
    def test_init_with_specific_benchmarks(self, mock_discover):
        """Test initialization with specific benchmarks."""
        # Set specific benchmarks
        self.args.benchmarks = ["training_speed", "model_accuracy"]
        
        # Mock discovery to return all available benchmarks
        mock_discover.return_value = ["training_speed", "inference_performance", "model_accuracy"]
        
        # Create runner
        runner = BenchmarkRunner(self.args)
        
        # Check that only the specified benchmarks are included
        self.assertEqual(len(runner.benchmarks), 2)
        self.assertIn("training_speed", runner.benchmarks)
        self.assertIn("model_accuracy", runner.benchmarks)
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_baseline(self, mock_file, mock_exists):
        """Test loading baseline data."""
        # Set baseline file
        baseline_file = os.path.join(self.temp_dir, "baseline.json")
        self.args.baseline_file = baseline_file
        
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock file open to return sample results
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(self.sample_results)
        
        # Create runner
        runner = BenchmarkRunner(self.args)
        
        # Load baseline
        baseline = runner._load_baseline()
        
        # Check that baseline was loaded
        self.assertEqual(baseline, self.sample_results)
    
    @patch('os.path.exists')
    def test_load_baseline_missing(self, mock_exists):
        """Test loading missing baseline data."""
        # Set baseline file
        baseline_file = os.path.join(self.temp_dir, "nonexistent.json")
        self.args.baseline_file = baseline_file
        
        # Mock file does not exist
        mock_exists.return_value = False
        
        # Create runner
        runner = BenchmarkRunner(self.args)
        
        # Load baseline
        baseline = runner._load_baseline()
        
        # Check that baseline is None
        self.assertIsNone(baseline)
    
    def test_compare_with_baseline(self):
        """Test comparing results with baseline."""
        # Create runner
        runner = BenchmarkRunner(self.args)
        
        # Create current results
        current_results = {
            "status": "passed",
            "execution_time": 9.5,  # Faster than baseline
            "results": {
                "throughput": 1200,  # Better than baseline
                "memory_usage": 2.0  # Better than baseline
            }
        }
        
        # Create baseline results
        baseline_results = {
            "status": "passed",
            "execution_time": 10.5,
            "results": {
                "throughput": 1000,
                "memory_usage": 2.5
            }
        }
        
        # Compare with baseline
        comparison = runner._compare_with_baseline(current_results, baseline_results)
        
        # Check comparison values
        self.assertIn("comparison", comparison)
        self.assertIn("execution_time_change_pct", comparison["comparison"])
        self.assertIn("throughput_change_pct", comparison["comparison"])
        self.assertIn("memory_usage_change_pct", comparison["comparison"])
        
        # Check that the comparisons are correct
        # Execution time decreased by ~9.5%
        self.assertAlmostEqual(comparison["comparison"]["execution_time_change_pct"], -9.52, places=1)
        # Throughput increased by 20%
        self.assertEqual(comparison["comparison"]["throughput_change_pct"], 20.0)
        # Memory usage decreased by 20%
        self.assertEqual(comparison["comparison"]["memory_usage_change_pct"], -20.0)
    
    @patch('benchmarking.runner.BenchmarkRunner._discover_benchmarks')
    @patch('benchmarking.runner.importlib.import_module')
    @patch('benchmarking.runner.time.time')
    def test_run_benchmarks(self, mock_time, mock_import, mock_discover):
        """Test running benchmarks."""
        # Mock time to return sequential values
        mock_time.side_effect = [0.0, 10.5, 20.0, 25.2]
        
        # Mock benchmark discovery
        mock_discover.return_value = ["training_speed", "inference_performance"]
        
        # Mock importing benchmark modules
        training_module = MagicMock()
        training_module.run_benchmark.return_value = {
            "throughput": 1000,
            "memory_usage": 2.5
        }
        
        inference_module = MagicMock()
        inference_module.run_benchmark.return_value = {
            "latency": 0.01,
            "tokens_per_second": 500
        }
        
        mock_import.side_effect = [training_module, inference_module]
        
        # Create runner
        runner = BenchmarkRunner(self.args)
        
        # Run benchmarks
        results = runner.run()
        
        # Check that benchmark modules were called
        training_module.run_benchmark.assert_called_once()
        inference_module.run_benchmark.assert_called_once()
        
        # Check results structure
        self.assertIn("metadata", results)
        self.assertIn("benchmarks", results)
        self.assertEqual(len(results["benchmarks"]), 2)
        
        # Check training results
        self.assertIn("training_speed", results["benchmarks"])
        self.assertEqual(results["benchmarks"]["training_speed"]["status"], "passed")
        self.assertEqual(results["benchmarks"]["training_speed"]["execution_time"], 10.5)
        self.assertEqual(results["benchmarks"]["training_speed"]["results"]["throughput"], 1000)
        
        # Check inference results
        self.assertIn("inference_performance", results["benchmarks"])
        self.assertEqual(results["benchmarks"]["inference_performance"]["status"], "passed")
        self.assertEqual(results["benchmarks"]["inference_performance"]["execution_time"], 5.2)
        self.assertEqual(results["benchmarks"]["inference_performance"]["results"]["tokens_per_second"], 500)

if __name__ == '__main__':
    unittest.main() 