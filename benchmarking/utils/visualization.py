"""
Benchmark Visualization Utilities
================================

This module provides functions for visualizing benchmark results,
including plotting performance metrics, generating comparison charts,
and creating reports.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
import logging

def create_training_speed_plot(
    results: Dict[str, Any],
    output_file: Optional[str] = None
) -> None:
    """
    Create a plot of training speed results.
    
    Args:
        results: Dictionary of training speed benchmark results
        output_file: Path to save the plot (None to display only)
    """
    if "batch_sizes" not in results:
        logging.warning("Training speed results missing batch_sizes key")
        return
    
    batch_sizes = []
    throughputs = []
    memory_usages = []
    
    for batch_size, metrics in results["batch_sizes"].items():
        if "error" in metrics:
            continue
            
        batch_sizes.append(int(batch_size))
        throughputs.append(metrics.get("throughput", 0))
        memory_usages.append(metrics.get("memory_allocated_gb", 0))
    
    if not batch_sizes:
        logging.warning("No valid batch size data found in results")
        return
    
    # Sort by batch size
    indices = np.argsort(batch_sizes)
    batch_sizes = [batch_sizes[i] for i in indices]
    throughputs = [throughputs[i] for i in indices]
    memory_usages = [memory_usages[i] for i in indices]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (samples/sec)', color=color)
    ax1.plot(batch_sizes, throughputs, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Memory Usage (GB)', color=color)
    ax2.plot(batch_sizes, memory_usages, 's-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Annotate optimal batch size
    if "optimal_batch_size" in results:
        opt_batch = results["optimal_batch_size"]
        opt_throughput = results["peak_throughput"]
        
        ax1.axvline(x=int(opt_batch), color='green', linestyle='--', alpha=0.7)
        ax1.annotate(f'Optimal: {opt_batch}',
                    xy=(int(opt_batch), opt_throughput),
                    xytext=(int(opt_batch) - 5, opt_throughput * 1.1),
                    arrowprops=dict(arrowstyle="->", color='green'))
    
    plt.title('Training Throughput and Memory Usage vs Batch Size')
    fig.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved training speed plot to {output_file}")
    else:
        plt.show()

def create_inference_speed_plot(
    results: Dict[str, Any],
    output_file: Optional[str] = None
) -> None:
    """
    Create a plot of inference speed results.
    
    Args:
        results: Dictionary of inference speed benchmark results
        output_file: Path to save the plot (None to display only)
    """
    # Plot single token latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First token vs subsequent token latency
    if "single_token_latency" in results:
        metrics = results["single_token_latency"]
        if "first_token_latency_ms" in metrics and "subsequent_token_latency_ms" in metrics:
            latencies = [
                metrics["first_token_latency_ms"],
                metrics["subsequent_token_latency_ms"]
            ]
            ax1.bar(['First Token', 'Subsequent Tokens'], latencies, color=['blue', 'orange'])
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Token Generation Latency')
            
            # Add values on bars
            for i, v in enumerate(latencies):
                ax1.text(i, v + 0.1, f'{v:.2f}ms', ha='center')
    
    # Batch size scaling
    if "batch_scaling" in results:
        batch_sizes = []
        throughputs = []
        memory_usages = []
        
        for batch_size, metrics in results["batch_scaling"].items():
            if "error" in metrics:
                continue
                
            batch_sizes.append(int(batch_size))
            throughputs.append(metrics.get("tokens_per_second", 0))
            memory_usages.append(metrics.get("memory_usage_gb", 0))
        
        if batch_sizes:
            # Sort by batch size
            indices = np.argsort(batch_sizes)
            batch_sizes = [batch_sizes[i] for i in indices]
            throughputs = [throughputs[i] for i in indices]
            memory_usages = [memory_usages[i] for i in indices]
            
            color = 'tab:blue'
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (tokens/sec)', color=color)
            line1 = ax2.plot(batch_sizes, throughputs, 'o-', color=color, label='Throughput')
            ax2.tick_params(axis='y', labelcolor=color)
            
            ax3 = ax2.twinx()
            color = 'tab:red'
            ax3.set_ylabel('Memory Usage (GB)', color=color)
            line2 = ax3.plot(batch_sizes, memory_usages, 's-', color=color, label='Memory')
            ax3.tick_params(axis='y', labelcolor=color)
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
            
            ax2.set_title('Inference Scaling with Batch Size')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved inference speed plot to {output_file}")
    else:
        plt.show()

def create_model_accuracy_plot(
    results: Dict[str, Any],
    output_file: Optional[str] = None
) -> None:
    """
    Create a plot of model accuracy results.
    
    Args:
        results: Dictionary of model accuracy benchmark results
        output_file: Path to save the plot (None to display only)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Perplexity
    if "perplexity" in results and "validation_perplexity" in results["perplexity"]:
        perplexity = results["perplexity"]["validation_perplexity"]
        ax1.bar(['Perplexity'], [perplexity], color='blue')
        ax1.set_ylabel('Perplexity (lower is better)')
        ax1.set_title('Model Perplexity')
        
        # Add value on bar
        ax1.text(0, perplexity + 0.1, f'{perplexity:.2f}', ha='center')
    
    # Token prediction accuracy
    if "token_prediction" in results:
        metrics = results["token_prediction"]
        if "top1_accuracy" in metrics and "top5_accuracy" in metrics:
            accuracies = [
                metrics["top1_accuracy"] * 100,  # Convert to percentage
                metrics["top5_accuracy"] * 100
            ]
            ax2.bar(['Top-1', 'Top-5'], accuracies, color=['blue', 'green'])
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Token Prediction Accuracy')
            ax2.set_ylim(0, 100)
            
            # Add values on bars
            for i, v in enumerate(accuracies):
                ax2.text(i, v + 1, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved model accuracy plot to {output_file}")
    else:
        plt.show()

def create_benchmark_report(
    results_file: str,
    output_dir: str,
    comparison_file: Optional[str] = None
) -> str:
    """
    Create a comprehensive benchmark report with visualizations.
    
    Args:
        results_file: Path to benchmark results JSON file
        output_dir: Directory to save report files
        comparison_file: Optional path to baseline results for comparison
        
    Returns:
        Path to generated report HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    comparison = None
    if comparison_file and os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
    
    # Create individual plots
    benchmark_data = results.get("benchmarks", {})
    
    if "training_speed" in benchmark_data:
        create_training_speed_plot(
            benchmark_data["training_speed"],
            os.path.join(output_dir, "training_speed.png")
        )
    
    if "inference_performance" in benchmark_data:
        create_inference_speed_plot(
            benchmark_data["inference_performance"],
            os.path.join(output_dir, "inference_speed.png")
        )
    
    if "model_accuracy" in benchmark_data:
        create_model_accuracy_plot(
            benchmark_data["model_accuracy"],
            os.path.join(output_dir, "model_accuracy.png")
        )
    
    # Generate HTML report
    report_file = os.path.join(output_dir, "benchmark_report.html")
    
    with open(report_file, 'w') as f:
        f.write(_generate_html_report(results, comparison))
    
    logging.info(f"Generated benchmark report at {report_file}")
    return report_file

def _generate_html_report(results: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> str:
    """Generate HTML report content from benchmark results."""
    
    system_info = results.get("system_info", {})
    benchmarks = results.get("benchmarks", {})
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Craft Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .better {{ color: green; }}
            .worse {{ color: red; }}
            .chart {{ margin: 20px 0; max-width: 800px; }}
        </style>
    </head>
    <body>
        <h1>Craft Benchmark Report</h1>
        
        <h2>System Information</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Device</td><td>{system_info.get("device", "Unknown")}</td></tr>
            <tr><td>CUDA Version</td><td>{system_info.get("cuda_version", "N/A")}</td></tr>
            <tr><td>Python Version</td><td>{system_info.get("python_version", "Unknown")}</td></tr>
            <tr><td>Timestamp</td><td>{system_info.get("timestamp", "Unknown")}</td></tr>
        </table>
        
        <h2>Summary</h2>
        <table>
            <tr><th>Benchmark</th><th>Key Metrics</th></tr>
    """
    
    # Add summary rows
    if "training_speed" in benchmarks:
        data = benchmarks["training_speed"]
        optimal_batch = data.get("optimal_batch_size", "N/A")
        peak_throughput = data.get("peak_throughput", 0)
        
        html += f"""
            <tr>
                <td>Training Speed</td>
                <td>
                    <div>Optimal batch size: <span class="metric">{optimal_batch}</span></div>
                    <div>Peak throughput: <span class="metric">{peak_throughput:.2f}</span> samples/sec</div>
                </td>
            </tr>
        """
    
    if "inference_performance" in benchmarks:
        data = benchmarks["inference_performance"]
        tokens_per_sec = data.get("avg_tokens_per_second", 0)
        first_token = data.get("single_token_latency", {}).get("first_token_latency_ms", 0)
        subsequent_token = data.get("single_token_latency", {}).get("subsequent_token_latency_ms", 0)
        
        html += f"""
            <tr>
                <td>Inference Performance</td>
                <td>
                    <div>Average generation speed: <span class="metric">{tokens_per_sec:.2f}</span> tokens/sec</div>
                    <div>First token latency: <span class="metric">{first_token:.2f}</span> ms</div>
                    <div>Subsequent token latency: <span class="metric">{subsequent_token:.2f}</span> ms</div>
                </td>
            </tr>
        """
    
    if "model_accuracy" in benchmarks:
        data = benchmarks["model_accuracy"]
        perplexity = data.get("perplexity", {}).get("validation_perplexity", "N/A")
        top1_acc = data.get("token_prediction", {}).get("top1_accuracy", 0) * 100
        
        html += f"""
            <tr>
                <td>Model Accuracy</td>
                <td>
                    <div>Validation perplexity: <span class="metric">{perplexity if perplexity == "N/A" else f"{perplexity:.2f}"}</span></div>
                    <div>Token prediction accuracy: <span class="metric">{top1_acc:.2f}%</span></div>
                </td>
            </tr>
        """
    
    html += """
        </table>
    """
    
    # Add comparison if available
    if comparison:
        html += """
        <h2>Comparison with Baseline</h2>
        <table>
            <tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Change</th></tr>
        """
        
        # Add comparison rows for each benchmark
        if "training_speed" in benchmarks and "training_speed" in comparison.get("benchmarks", {}):
            current = benchmarks["training_speed"].get("peak_throughput", 0)
            baseline = comparison["benchmarks"]["training_speed"].get("peak_throughput", 0)
            if current > 0 and baseline > 0:
                change_pct = ((current - baseline) / baseline) * 100
                class_name = "better" if change_pct > 0 else "worse" if change_pct < 0 else ""
                sign = "+" if change_pct > 0 else ""
                
                html += f"""
                <tr>
                    <td>Training Throughput</td>
                    <td>{current:.2f} samples/sec</td>
                    <td>{baseline:.2f} samples/sec</td>
                    <td class="{class_name}">{sign}{change_pct:.2f}%</td>
                </tr>
                """
        
        if "inference_performance" in benchmarks and "inference_performance" in comparison.get("benchmarks", {}):
            current = benchmarks["inference_performance"].get("avg_tokens_per_second", 0)
            baseline = comparison["benchmarks"]["inference_performance"].get("avg_tokens_per_second", 0)
            if current > 0 and baseline > 0:
                change_pct = ((current - baseline) / baseline) * 100
                class_name = "better" if change_pct > 0 else "worse" if change_pct < 0 else ""
                sign = "+" if change_pct > 0 else ""
                
                html += f"""
                <tr>
                    <td>Inference Speed</td>
                    <td>{current:.2f} tokens/sec</td>
                    <td>{baseline:.2f} tokens/sec</td>
                    <td class="{class_name}">{sign}{change_pct:.2f}%</td>
                </tr>
                """
        
        if "model_accuracy" in benchmarks and "model_accuracy" in comparison.get("benchmarks", {}):
            current_perp = benchmarks["model_accuracy"].get("validation_perplexity", 0)
            baseline_perp = comparison["benchmarks"]["model_accuracy"].get("validation_perplexity", 0)
            
            if current_perp > 0 and baseline_perp > 0:
                change_pct = ((baseline_perp - current_perp) / baseline_perp) * 100  # Lower perplexity is better
                class_name = "better" if change_pct > 0 else "worse" if change_pct < 0 else ""
                sign = "+" if change_pct > 0 else ""
                
                html += f"""
                <tr>
                    <td>Perplexity</td>
                    <td>{current_perp:.2f}</td>
                    <td>{baseline_perp:.2f}</td>
                    <td class="{class_name}">{sign}{change_pct:.2f}%</td>
                </tr>
                """
        
        html += """
        </table>
        """
    
    # Add charts
    html += """
        <h2>Visualizations</h2>
    """
    
    if "training_speed" in benchmarks:
        html += """
        <h3>Training Speed</h3>
        <div class="chart">
            <img src="training_speed.png" alt="Training Speed Chart" width="100%">
        </div>
        """
    
    if "inference_performance" in benchmarks:
        html += """
        <h3>Inference Performance</h3>
        <div class="chart">
            <img src="inference_speed.png" alt="Inference Speed Chart" width="100%">
        </div>
        """
    
    if "model_accuracy" in benchmarks:
        html += """
        <h3>Model Accuracy</h3>
        <div class="chart">
            <img src="model_accuracy.png" alt="Model Accuracy Chart" width="100%">
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    return html 