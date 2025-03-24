import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarking.main import parse_args, main
from benchmarking.runner import BenchmarkRunner
from benchmarking.utils.visualization import create_benchmark_report

class MockArgs:
    """Mock arguments for testing."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TestBenchmarkingSystem(unittest.TestCase):
    """Integration tests for the benchmarking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directories
        self.results_dir = os.path.join(self.temp_dir, "results")
        self.reports_dir = os.path.join(self.temp_dir, "reports")
        self.data_dir = os.path.join(self.temp_dir, "data")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create a sample benchmark results file
        self.results_file = os.path.join(self.results_dir, "benchmark_results.json")
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
        
        with open(self.results_file, 'w') as f:
            json.dump(self.sample_results, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('sys.argv')
    @patch('benchmarking.main.BenchmarkRunner')
    def test_main_run_command(self, mock_runner_class, mock_argv):
        """Test the 'run' command."""
        # Mock command line arguments
        mock_argv[1:] = ['run', '--output-dir', self.results_dir]
        
        # Mock runner
        mock_runner = mock_runner_class.return_value
        mock_runner.run.return_value = self.sample_results
        
        # Run the main function
        exit_code = main()
        
        # Check that runner was created and called
        mock_runner_class.assert_called_once()
        mock_runner.run.assert_called_once()
        
        # Check exit code
        self.assertEqual(exit_code, 0)
    
    @patch('sys.argv')
    @patch('benchmarking.main.create_benchmark_report')
    def test_main_report_command(self, mock_report, mock_argv):
        """Test the 'report' command."""
        # Mock command line arguments
        mock_argv[1:] = [
            'report', 
            '--results-file', self.results_file,
            '--output-dir', self.reports_dir
        ]
        
        # Mock report generation
        report_file = os.path.join(self.reports_dir, "benchmark_report.html")
        mock_report.return_value = report_file
        
        # Run the main function
        exit_code = main()
        
        # Check that report generation was called
        mock_report.assert_called_once_with(self.results_file, self.reports_dir, None)
        
        # Check exit code
        self.assertEqual(exit_code, 0)
    
    @patch('sys.argv')
    @patch('builtins.print')
    def test_main_list_command(self, mock_print, mock_argv):
        """Test the 'list' command."""
        # Mock command line arguments
        mock_argv[1:] = ['list']
        
        # Run the main function with list command
        with patch('benchmarking.main.os.path.exists', return_value=True):
            with patch('benchmarking.main.os.listdir', return_value=["training_speed.py", "inference_performance.py"]):
                exit_code = main()
        
        # Check that print was called with benchmark listings
        mock_print.assert_any_call("\nAvailable Benchmarks:")
        
        # Check exit code
        self.assertEqual(exit_code, 0)
    
    @patch('sys.argv')
    def test_main_no_command(self, mock_argv):
        """Test main function with no command."""
        # Mock empty command line arguments
        mock_argv[1:] = []
        
        # Run the main function with no args
        with patch('benchmarking.main.parse_args') as mock_parse:
            mock_parse.side_effect = SystemExit(1)
            with self.assertRaises(SystemExit):
                main()
    
    @patch('benchmarking.runner.BenchmarkRunner._discover_benchmarks')
    @patch('benchmarking.runner.importlib.import_module')
    def test_runner_integration(self, mock_import, mock_discover):
        """Test benchmark runner integration."""
        # Create args for runner
        args = MockArgs(
            output_dir=self.results_dir,
            benchmarks=["training_speed"],
            baseline_file=None,
            log_level="INFO",
            data_dir=self.data_dir,
            model_checkpoint=None,
            iterations=1
        )
        
        # Mock benchmark discovery
        mock_discover.return_value = ["training_speed", "inference_performance"]
        
        # Create a simple mock benchmark module
        class MockBenchmark:
            @staticmethod
            def run_benchmark(args):
                return {"throughput": 1000, "memory_usage": 2.5}
        
        # Mock importing the module
        mock_import.return_value = MockBenchmark
        
        # Create runner
        runner = BenchmarkRunner(args)
        
        # Run benchmarks
        results = runner.run()
        
        # Check results
        self.assertIn("metadata", results)
        self.assertIn("benchmarks", results)
        self.assertEqual(len(results["benchmarks"]), 1)
        self.assertIn("training_speed", results["benchmarks"])
        self.assertEqual(results["benchmarks"]["training_speed"]["status"], "passed")
        self.assertIn("throughput", results["benchmarks"]["training_speed"]["results"])
    
    @patch('matplotlib.pyplot.savefig')
    def test_benchmark_report_generation(self, mock_savefig):
        """Test benchmark report generation."""
        # Generate a report
        report_file = create_benchmark_report(
            self.results_file,
            self.reports_dir,
            None  # No comparison file
        )
        
        # Check that the report file exists
        self.assertTrue(os.path.exists(report_file))
        
        # Check that savefig was called (to generate charts)
        mock_savefig.assert_called()

if __name__ == '__main__':
    unittest.main() 