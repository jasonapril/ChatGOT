#!/usr/bin/env python
"""
Monitoring Visualization Module
==============================

This module provides visualization utilities for training monitoring:

1. Real-time plotting of throughput metrics
2. Component breakdown visualization
3. Memory usage charts
4. Performance trend analysis

These visualizations help identify bottlenecks and optimize training performance.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Any, Tuple

try:
    # Optional imports for interactive plotting
    import IPython
    from IPython.display import clear_output, display
    INTERACTIVE_MODE_AVAILABLE = True
except ImportError:
    INTERACTIVE_MODE_AVAILABLE = False

def create_throughput_plot(throughput_history: List[float], 
                          window_size: int = 50,
                          title: str = "Training Throughput") -> Figure:
    """
    Create a plot of training throughput over time.
    
    Args:
        throughput_history: List of throughput values (tokens/sec)
        window_size: Window size for moving average
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw throughput
    x = np.arange(len(throughput_history))
    ax.plot(x, throughput_history, 'b-', alpha=0.5, label='Throughput')
    
    # Calculate and plot moving average if enough data points
    if len(throughput_history) >= window_size:
        moving_avg = []
        for i in range(len(throughput_history) - window_size + 1):
            window = throughput_history[i:i+window_size]
            moving_avg.append(np.mean(window))
        
        # Plot moving average
        ma_x = np.arange(window_size-1, len(throughput_history))
        ax.plot(ma_x, moving_avg, 'r-', linewidth=2, label=f'{window_size}-batch Moving Avg')
    
    # Add labels and legend
    ax.set_xlabel('Batch')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_component_breakdown_chart(components: Dict[str, float]) -> Figure:
    """
    Create a pie chart showing the breakdown of time spent in different components.
    
    Args:
        components: Dictionary mapping component names to percentages
        
    Returns:
        Matplotlib Figure object
    """
    # Filter out components with zero or negative values
    filtered_components = {k: v for k, v in components.items() if v > 0}
    
    if not filtered_components:
        # Create empty pie chart with a message if no data
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No component data available", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return fig
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define colors for each component
    colors = {
        'forward': '#4285F4',  # Google Blue
        'backward': '#EA4335',  # Google Red
        'optimizer': '#FBBC05',  # Google Yellow
        'data_loading': '#34A853',  # Google Green
        'other': '#7D7D7D'      # Gray
    }
    
    # Extract labels and sizes
    labels = list(filtered_components.keys())
    sizes = list(filtered_components.values())
    component_colors = [colors.get(label, '#7D7D7D') for label in labels]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=component_colors,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False
    )
    
    # Make text easier to read
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color('white')
    
    ax.set_title('Time Spent in Training Components', fontsize=16)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    
    plt.tight_layout()
    return fig

def create_memory_usage_chart(memory_stats: Dict[str, float]) -> Figure:
    """
    Create a bar chart of memory usage statistics.
    
    Args:
        memory_stats: Dictionary with memory statistics in MB
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract memory statistics
    allocated = memory_stats.get('allocated', 0)
    reserved = memory_stats.get('reserved', 0)
    peak = memory_stats.get('peak', 0)
    
    # Create bar chart
    labels = ['Allocated', 'Reserved', 'Peak']
    values = [allocated, reserved, peak]
    colors = ['#4285F4', '#34A853', '#EA4335']
    
    bars = ax.bar(labels, values, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f} MB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('Memory Type')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('GPU Memory Usage')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_dashboard(monitor_summary: Dict[str, Any]) -> Figure:
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        monitor_summary: Summary dictionary from ThroughputMonitor
        
    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Extract data from summary
    throughput_history = monitor_summary.get('throughput_history', [])
    components = monitor_summary.get('components', {})
    memory_stats = monitor_summary.get('memory', {})
    current_throughput = monitor_summary.get('throughput', 0)
    total_tokens = monitor_summary.get('total_tokens', 0)
    avg_batch_time = monitor_summary.get('avg_batch_time', 0)
    std_batch_time = monitor_summary.get('std_batch_time', 0)
    
    # Create grid for subplots
    gs = fig.add_gridspec(2, 3)
    
    # Key metrics text
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis('off')
    metrics_text = (
        f"Current Throughput: {current_throughput:.2f} tokens/sec\n"
        f"Total Tokens Processed: {total_tokens:,}\n"
        f"Avg Batch Time: {avg_batch_time*1000:.2f} ms\n"
        f"Std Dev Batch Time: {std_batch_time*1000:.2f} ms"
    )
    ax_metrics.text(0.5, 0.5, metrics_text, 
                   ha='center', va='center', 
                   fontsize=14,
                   transform=ax_metrics.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_metrics.set_title('Key Metrics', fontsize=16)
    
    # Throughput history plot
    ax_throughput = fig.add_subplot(gs[0, 1:])
    if throughput_history:
        x = np.arange(len(throughput_history))
        ax_throughput.plot(x, throughput_history, 'b-', linewidth=2)
        ax_throughput.set_xlabel('Batch')
        ax_throughput.set_ylabel('Throughput (tokens/sec)')
        ax_throughput.set_title('Training Throughput')
        ax_throughput.grid(True)
    else:
        ax_throughput.text(0.5, 0.5, "No throughput data available",
                          ha='center', va='center',
                          transform=ax_throughput.transAxes,
                          fontsize=14)
        ax_throughput.set_title('Training Throughput')
        ax_throughput.axis('off')
    
    # Component breakdown
    ax_components = fig.add_subplot(gs[1, 0])
    if components:
        # Filter out components with zero or negative values
        filtered_components = {k: v for k, v in components.items() if v > 0}
        
        if filtered_components:
            labels = list(filtered_components.keys())
            sizes = list(filtered_components.values())
            colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#7D7D7D']
            ax_components.pie(sizes, labels=labels, colors=colors[:len(labels)],
                             autopct='%1.1f%%', startangle=90)
            ax_components.axis('equal')
    else:
        ax_components.text(0.5, 0.5, "No component data available",
                          ha='center', va='center',
                          transform=ax_components.transAxes,
                          fontsize=14)
        ax_components.axis('off')
    ax_components.set_title('Time Breakdown')
    
    # Memory usage
    ax_memory = fig.add_subplot(gs[1, 1])
    if memory_stats:
        # Extract memory statistics
        allocated = memory_stats.get('allocated', 0)
        reserved = memory_stats.get('reserved', 0)
        peak = memory_stats.get('peak', 0)
        
        # Create bar chart
        labels = ['Allocated', 'Reserved', 'Peak']
        values = [allocated, reserved, peak]
        colors = ['#4285F4', '#34A853', '#EA4335']
        
        ax_memory.bar(labels, values, color=colors)
        ax_memory.set_ylabel('Memory (MB)')
        ax_memory.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax_memory.text(0.5, 0.5, "No memory data available",
                      ha='center', va='center',
                      transform=ax_memory.transAxes,
                      fontsize=14)
        ax_memory.axis('off')
    ax_memory.set_title('GPU Memory Usage')
    
    # Batch time histogram
    ax_hist = fig.add_subplot(gs[1, 2])
    batch_times = monitor_summary.get('batch_times', [])
    if batch_times:
        ax_hist.hist(batch_times, bins=min(10, len(batch_times)), color='#4285F4', alpha=0.7)
        ax_hist.axvline(avg_batch_time, color='r', linestyle='dashed', linewidth=2)
        ax_hist.set_xlabel('Batch Time (s)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True, linestyle='--', alpha=0.7)
    else:
        ax_hist.text(0.5, 0.5, "No batch time data available",
                    ha='center', va='center',
                    transform=ax_hist.transAxes,
                    fontsize=14)
        ax_hist.axis('off')
    ax_hist.set_title('Batch Time Distribution')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Training Performance Dashboard', fontsize=20)
    
    return fig

def display_dashboard_interactive(monitor_summary: Dict[str, Any]) -> None:
    """
    Display dashboard interactively in Jupyter/IPython environment.
    
    Args:
        monitor_summary: Summary dictionary from ThroughputMonitor
    """
    if not INTERACTIVE_MODE_AVAILABLE:
        logging.warning("IPython display not available. Can't show interactive dashboard.")
        return
        
    try:
        # Clear previous output
        clear_output(wait=True)
        
        # Create and display dashboard
        fig = create_dashboard(monitor_summary)
        display(fig)
        plt.close(fig)  # Close to prevent memory leak
    except Exception as e:
        logging.warning(f"Error displaying interactive dashboard: {e}")

def save_dashboard(monitor_summary: Dict[str, Any], filepath: str) -> None:
    """
    Save dashboard to a file.
    
    Args:
        monitor_summary: Summary dictionary from ThroughputMonitor
        filepath: Path to save the figure
    """
    try:
        fig = create_dashboard(monitor_summary)
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)  # Close to prevent memory leak
        logging.info(f"Dashboard saved to {filepath}")
    except Exception as e:
        logging.warning(f"Error saving dashboard: {e}") 