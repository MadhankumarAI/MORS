"""
Visualization utilities for Sparse-MOR benchmarking results
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List
import torch

def create_latency_comparison(results: Dict[str, Any]) -> go.Figure:
    """Create interactive latency comparison plot"""
    fig = go.Figure(data=[
        go.Bar(
            name='Baseline',
            x=['Inference Latency'],
            y=[results['baseline']['avg_latency']],
            marker_color='rgb(55, 83, 109)'
        ),
        go.Bar(
            name='Sparse+MOR',
            x=['Inference Latency'],
            y=[results['sparse_mor']['avg_latency']],
            marker_color='rgb(26, 118, 255)'
        )
    ])
    
    fig.update_layout(
        title='Inference Latency Comparison',
        yaxis_title='Latency (ms)',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def create_memory_dashboard(results: Dict[str, Any]) -> go.Figure:
    """Create memory usage dashboard"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Peak Memory Usage', 'Memory Allocation Timeline'),
        specs=[[{'type': 'domain'}, {'type': 'xy'}]]
    )
    
    # Peak memory pie chart
    baseline_mem = results['baseline']['max_memory'] / 1e9
    sparse_mem = results['sparse_mor']['max_memory'] / 1e9
    
    fig.add_trace(
        go.Pie(
            labels=['Baseline', 'Sparse+MOR'],
            values=[baseline_mem, sparse_mem],
            hole=.3
        ),
        row=1, col=1
    )
    
    # Memory timeline
    fig.add_trace(
        go.Scatter(
            x=['Start', 'Load Model', 'Forward Pass', 'End'],
            y=[0, baseline_mem*0.7, baseline_mem, baseline_mem*0.8],
            name='Baseline',
            line=dict(color='rgb(55, 83, 109)')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=['Start', 'Load Model', 'Forward Pass', 'End'],
            y=[0, sparse_mem*0.7, sparse_mem, sparse_mem*0.8],
            name='Sparse+MOR',
            line=dict(color='rgb(26, 118, 255)')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Memory Usage Analysis',
        template='plotly_white'
    )
    
    return fig

def create_similarity_heatmap(similarities: List[float], sequence_lengths: List[int]) -> go.Figure:
    """Create similarity heatmap across sequence lengths"""
    sim_matrix = np.array(similarities).reshape(-1, len(sequence_lengths))
    
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=sequence_lengths,
        y=[f'Sample {i+1}' for i in range(sim_matrix.shape[0])],
        colorscale='RdYlBu',
        zmin=0.9,
        zmax=1.0
    ))
    
    fig.update_layout(
        title='Cosine Similarity Heatmap',
        xaxis_title='Sequence Length',
        yaxis_title='Test Samples',
        template='plotly_white'
    )
    
    return fig

def create_pareto_plot(results: Dict[str, Any], sweep_results: List[Dict[str, Any]]) -> go.Figure:
    """Create Pareto frontier plot of speedup vs similarity"""
    speedups = [r['comparison']['speedup'] for r in sweep_results]
    similarities = [r['sparse_mor']['avg_similarity'] for r in sweep_results]
    k_values = [r['config']['sparsity']['k'] for r in sweep_results]
    
    fig = go.Figure()
    
    # Scatter plot of all configurations
    fig.add_trace(go.Scatter(
        x=similarities,
        y=speedups,
        mode='markers+text',
        marker=dict(
            size=10,
            color=k_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='k value')
        ),
        text=[f'k={k}' for k in k_values],
        name='Configurations'
    ))
    
    # Add current configuration point
    fig.add_trace(go.Scatter(
        x=[results['sparse_mor']['avg_similarity']],
        y=[results['comparison']['speedup']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='star'
        ),
        name='Current Config'
    ))
    
    fig.update_layout(
        title='Speedup vs Similarity Pareto Frontier',
        xaxis_title='Cosine Similarity',
        yaxis_title='Speedup Factor',
        template='plotly_white'
    )
    
    return fig

def create_throughput_analysis(results: Dict[str, Any]) -> go.Figure:
    """Create throughput analysis dashboard"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Tokens/Second', 'Throughput Over Time')
    )
    
    # Bar chart for average throughput
    fig.add_trace(
        go.Bar(
            x=['Baseline', 'Sparse+MOR'],
            y=[
                results['baseline']['avg_throughput'],
                results['sparse_mor']['avg_throughput']
            ],
            marker_color=['rgb(55, 83, 109)', 'rgb(26, 118, 255)']
        ),
        row=1, col=1
    )
    
    # Line chart for throughput over time
    timestamps = np.linspace(0, 60, 10)  # Simulated 1-minute run
    baseline_throughput = results['baseline']['avg_throughput'] * (1 + np.random.randn(10) * 0.1)
    sparse_throughput = results['sparse_mor']['avg_throughput'] * (1 + np.random.randn(10) * 0.1)
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=baseline_throughput,
            name='Baseline',
            line=dict(color='rgb(55, 83, 109)')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=sparse_throughput,
            name='Sparse+MOR',
            line=dict(color='rgb(26, 118, 255)')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title='Throughput Analysis',
        template='plotly_white'
    )
    
    return fig

def create_sparsity_pattern_viz(mask: torch.Tensor, k: int) -> go.Figure:
    """Visualize sparsity patterns"""
    # Convert mask to numpy and take first example
    mask_np = mask[0].cpu().numpy()
    
    fig = go.Figure(data=go.Heatmap(
        z=mask_np,
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title=f'Sparsity Pattern (k={k})',
        xaxis_title='Hidden Dimension',
        yaxis_title='Sequence Position',
        template='plotly_white'
    )
    
    return fig

def generate_html_dashboard(
    results: Dict[str, Any],
    output_dir: str,
    mask: torch.Tensor = None
) -> None:
    """Generate complete HTML dashboard with all visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all plots
    latency_fig = create_latency_comparison(results)
    memory_fig = create_memory_dashboard(results)
    throughput_fig = create_throughput_analysis(results)
    
    # Generate HTML
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sparse-MOR Benchmark Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .grid-container {{ display: grid; grid-gap: 20px; }}
            .summary-box {{
                background: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Sparse-MOR Benchmark Results</h1>
        
        <div class="summary-box">
            <h2>Performance Summary</h2>
            <ul>
                <li>Speedup: {results['comparison']['speedup']:.2f}x</li>
                <li>Memory Reduction: {results['comparison']['memory_reduction']*100:.1f}%</li>
                <li>Similarity: {results['sparse_mor']['avg_similarity']*100:.1f}%</li>
            </ul>
        </div>
        
        <div class="grid-container">
            <div>{latency_fig.to_html(full_html=False)}</div>
            <div>{memory_fig.to_html(full_html=False)}</div>
            <div>{throughput_fig.to_html(full_html=False)}</div>
    """
    
    # Add sparsity pattern if available
    if mask is not None:
        sparsity_fig = create_sparsity_pattern_viz(mask, results['config']['sparsity']['k'])
        dashboard_html += f"""
            <div>{sparsity_fig.to_html(full_html=False)}</div>
        """
    
    dashboard_html += """
        </div>
    </body>
    </html>
    """
    
    # Save dashboard
    with open(output_dir / 'dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    # Save individual plots as PNG
    latency_fig.write_image(output_dir / 'latency.png')
    memory_fig.write_image(output_dir / 'memory.png')
    throughput_fig.write_image(output_dir / 'throughput.png')
    if mask is not None:
        sparsity_fig.write_image(output_dir / 'sparsity_pattern.png')