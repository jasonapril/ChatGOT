import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.performance.visualization import (
    create_throughput_plot,
    create_component_breakdown_chart,
    create_memory_usage_chart,
    create_dashboard,
    save_dashboard
)
from src.performance.throughput_core import ThroughputMonitor

class TestVisualization(unittest.TestCase):
    """Unit tests for the visualization module."""
    
    def setUp(self):
        """Set up common data for tests."""
        self.monitor = MagicMock(spec=ThroughputMonitor)
        # Provide a realistic summary structure - Use 'components' key
        self.monitor_summary = {
            'throughput': 1500.5,
            'components': { # Renamed from component_breakdown
                'data_loading': 10.0,
                'forward': 50.0,
                'backward': 30.0,
                'optimizer': 5.0,
                'other': 5.0
            },
            'memory': {
                'allocated': 512.0, # MB
                'reserved': 1024.0, # MB
                'peak': 800.0 # MB
            },
            'total_samples': 1000,
            'total_tokens': 100000,
            'avg_batch_time': 0.1,
            'std_batch_time': 0.02,
            'throughput_history': [1400.0, 1500.0, 1600.0, 1550.0]
        }
        self.monitor.get_summary.return_value = self.monitor_summary
        
        # Mock matplotlib objects
        self.mock_fig, self.mock_axes = MagicMock(), [MagicMock(), MagicMock(), MagicMock()]
        self.mock_axes[0].twinx.return_value = MagicMock() # For secondary y-axis
    
    @patch('matplotlib.pyplot.show') # Prevent plots from showing
    def test_create_throughput_plot(self, mock_show):
        """Test creating a throughput plot."""
        # Call the function without the 'ax' argument
        fig = create_throughput_plot(self.monitor_summary['throughput_history'])
        # Assert a figure was returned
        self.assertIsInstance(fig, plt.Figure)
        # Minimal check: axes should have a title
        self.assertTrue(len(fig.axes) > 0)
        self.assertIsNotNone(fig.axes[0].get_title())
    
    @patch('matplotlib.pyplot.show') # Prevent plots from showing
    def test_create_component_breakdown_chart(self, mock_show):
        """Test creating a component breakdown chart."""
        # Use the correct key 'components'
        fig = create_component_breakdown_chart(self.monitor_summary['components'])
        # Assert a figure was returned
        self.assertIsInstance(fig, plt.Figure)
        # Check for pie chart elements (patches/wedges)
        self.assertTrue(len(fig.axes) > 0)
        self.assertTrue(len(fig.axes[0].patches) > 0) # Check for wedges
        self.assertIsNotNone(fig.axes[0].get_title())
    
    @patch('matplotlib.pyplot.show') # Prevent plots from showing
    def test_create_memory_usage_chart(self, mock_show):
        """Test creating a memory usage chart."""
        # Call the function without the 'ax' argument
        fig = create_memory_usage_chart(self.monitor_summary['memory'])
        # Assert a figure was returned
        self.assertIsInstance(fig, plt.Figure)
        # Check for bar chart elements (patches)
        self.assertTrue(len(fig.axes) > 0)
        self.assertTrue(len(fig.axes[0].patches) > 0) # Check for bars
        self.assertIsNotNone(fig.axes[0].get_title())
    
    @patch('matplotlib.pyplot.show') # Prevent display
    def test_create_dashboard(self, mock_show):
        """Test creating a dashboard."""
        # Call the function with test data - it should create its own figure
        fig = create_dashboard(self.monitor_summary)
        
        # Assert a real Figure object was returned and it has axes
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes) > 0) # Check that subplots were added
    
    @patch('src.performance.visualization.create_dashboard')
    @patch('matplotlib.pyplot.figure') 
    def test_save_dashboard(self, mock_figure, mock_create_dashboard):
        """Test saving a dashboard."""
        # Configure mocks
        mock_dashboard_fig = mock_figure.return_value # Fig returned by create_dashboard
        mock_create_dashboard.return_value = mock_dashboard_fig
        output_path = "test_dashboard.png"
        
        # Call save_dashboard
        save_dashboard(self.monitor_summary, output_path)
        
        # Verify create_dashboard was called
        mock_create_dashboard.assert_called_once_with(self.monitor_summary)
        # Verify savefig was called on the figure returned by create_dashboard
        mock_dashboard_fig.savefig.assert_called_once_with(output_path, dpi=100, bbox_inches='tight')
    
    @patch('matplotlib.pyplot.show') # Prevent plots from showing
    def test_empty_data_handling(self, mock_show):
        """Test handling of empty data."""
        # Create an empty summary structure
        empty_summary = {
            'throughput': 0.0,
            'components': {},
            'memory': {'allocated': 0.0, 'reserved': 0.0, 'peak': 0.0},
            'throughput_history': [],
            'avg_batch_time': 0.0,
            'std_batch_time': 0.0
        }
        
        # Test create_dashboard with empty summary - should run without error
        try:
            fig = create_dashboard(empty_summary)
            self.assertIsInstance(fig, plt.Figure)
        except Exception as e:
            self.fail(f"create_dashboard raised an exception with empty data: {e}")

if __name__ == '__main__':
    unittest.main() 