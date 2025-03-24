import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.visualization import (
    create_throughput_plot,
    create_component_breakdown_chart,
    create_memory_usage_chart,
    create_dashboard,
    save_dashboard
)

class TestVisualization(unittest.TestCase):
    """Unit tests for the visualization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample data for visualization tests
        self.throughput_history = [100, 120, 110, 130, 125]
        self.component_breakdown = {
            'data_loading': 0.1,
            'forward': 0.3,
            'backward': 0.4,
            'optimizer': 0.2
        }
        self.memory_stats = {
            'allocated_gb': 1.2,
            'reserved_gb': 2.0,
            'peak_gb': 1.8
        }
        
        # Sample monitor summary
        self.monitor_summary = {
            'throughput': {
                'tokens_per_second': 1000,
                'samples_per_second': 32
            },
            'throughput_history': self.throughput_history,
            'component_breakdown': self.component_breakdown,
            'memory': self.memory_stats
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('matplotlib.pyplot.show')
    def test_create_throughput_plot(self, mock_show):
        """Test creating a throughput plot."""
        # Create plot
        fig = create_throughput_plot(self.throughput_history, window_size=3)
        
        # Check that a figure was created
        self.assertIsNotNone(fig)
        
        # Check that the figure has the expected elements
        self.assertEqual(len(fig.axes), 1)  # One axis
        axis = fig.axes[0]
        
        # Check title and labels
        self.assertIn("Training Throughput", axis.get_title())
        self.assertEqual(axis.get_xlabel(), "Batch")
        self.assertEqual(axis.get_ylabel(), "Tokens/second")
        
        # Check that there are two lines (throughput and moving average)
        self.assertEqual(len(axis.lines), 2)
    
    @patch('matplotlib.pyplot.show')
    def test_create_component_breakdown_chart(self, mock_show):
        """Test creating a component breakdown chart."""
        # Create chart
        fig = create_component_breakdown_chart(self.component_breakdown)
        
        # Check that a figure was created
        self.assertIsNotNone(fig)
        
        # Check that the figure has the expected elements
        self.assertEqual(len(fig.axes), 1)  # One axis
        axis = fig.axes[0]
        
        # Check title
        self.assertIn("Component Time Breakdown", axis.get_title())
        
        # Check that we have a pie chart (wedges)
        self.assertTrue(len(axis.patches) > 0)
    
    @patch('matplotlib.pyplot.show')
    def test_create_memory_usage_chart(self, mock_show):
        """Test creating a memory usage chart."""
        # Create chart
        fig = create_memory_usage_chart(self.memory_stats)
        
        # Check that a figure was created
        self.assertIsNotNone(fig)
        
        # Check that the figure has the expected elements
        self.assertEqual(len(fig.axes), 1)  # One axis
        axis = fig.axes[0]
        
        # Check title and labels
        self.assertIn("Memory Usage", axis.get_title())
        self.assertEqual(axis.get_xlabel(), "Memory Type")
        self.assertEqual(axis.get_ylabel(), "Memory (GB)")
        
        # Check that we have bars
        self.assertEqual(len(axis.patches), 3)  # Three bars for the three memory stats
    
    @patch('matplotlib.pyplot.show')
    def test_create_dashboard(self, mock_show):
        """Test creating a dashboard."""
        # Create dashboard
        fig = create_dashboard(self.monitor_summary)
        
        # Check that a figure was created
        self.assertIsNotNone(fig)
        
        # Check that the figure has the expected elements
        self.assertGreaterEqual(len(fig.axes), 3)  # At least three subplots
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_dashboard(self, mock_savefig):
        """Test saving a dashboard."""
        # Create output path
        output_path = os.path.join(self.temp_dir, "dashboard.png")
        
        # Save dashboard
        save_dashboard(self.monitor_summary, output_path)
        
        # Check that savefig was called with the right path
        mock_savefig.assert_called_once_with(output_path, dpi=100, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.show')
    def test_empty_data_handling(self, mock_show):
        """Test handling of empty data."""
        # Empty throughput history
        fig1 = create_throughput_plot([])
        self.assertIsNotNone(fig1)
        
        # Empty component breakdown
        fig2 = create_component_breakdown_chart({})
        self.assertIsNotNone(fig2)
        
        # Empty memory stats
        fig3 = create_memory_usage_chart({})
        self.assertIsNotNone(fig3)
        
        # Empty monitor summary
        empty_summary = {
            'throughput': {},
            'throughput_history': [],
            'component_breakdown': {},
            'memory': {}
        }
        fig4 = create_dashboard(empty_summary)
        self.assertIsNotNone(fig4)

if __name__ == '__main__':
    unittest.main() 