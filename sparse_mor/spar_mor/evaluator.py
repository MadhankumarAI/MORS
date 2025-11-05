"""
Evaluation and benchmarking utilities
"""
import time
from typing import Dict, Any, List, Tuple
import json
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np

from .spar_mor import SparMoRPipeline
from .visualization import (
    create_latency_comparison,
    create_memory_dashboard,
    create_throughput_analysis,
    create_pareto_plot,
    create_sparsity_pattern_viz,
    generate_html_dashboard
)

def measure_latency(
    model: PreTrainedModel,
    inputs: Dict[str, torch.Tensor],
    num_runs: int = 100
) -> float:
    """Measure average inference latency"""
    latencies = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(**inputs)
            latencies.append(time.perf_counter() - start)
            
    return sum(latencies) / len(latencies) * 1000  # ms

def measure_gpu_memory(model: PreTrainedModel) -> Tuple[int, int]:
    """Get max and current allocated GPU memory"""
    torch.cuda.reset_peak_memory_stats()
    max_mem = torch.cuda.max_memory_allocated()
    current_mem = torch.cuda.memory_allocated()
    return max_mem, current_mem

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute cosine similarity between tensors"""
    return F.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()

def run_comparison(
    model: PreTrainedModel,
    eval_inputs: List[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run comprehensive comparison between baseline and sparsity+MOR
    
    Args:
        model: Model to evaluate
        eval_inputs: List of input batches
        config: Config with sparsity_k etc
        output_dir: Directory to save results
        
    Returns:
        Dict with metrics and paths to saved plots
    """
    results = {
        "baseline": {},
        "sparse_mor": {},
        "comparison": {}
    }
    
    # Baseline measurements
    baseline_latencies = []
    baseline_throughput = []
    baseline_outputs = []
    
    model.eval()
    with torch.no_grad():
        for inputs in eval_inputs:
            # Measure baseline
            latency = measure_latency(model, inputs)
            baseline_latencies.append(latency)
            
            max_mem, current_mem = measure_gpu_memory(model)
            results["baseline"]["max_memory"] = max_mem
            results["baseline"]["current_memory"] = current_mem
            
            outputs = model(**inputs)
            baseline_outputs.append(outputs.logits)
            
            # Calculate throughput
            num_tokens = inputs["input_ids"].numel()
            throughput = num_tokens / (latency / 1000)  # tokens/sec
            baseline_throughput.append(throughput)
            
    results["baseline"]["avg_latency"] = sum(baseline_latencies) / len(baseline_latencies)
    results["baseline"]["avg_throughput"] = sum(baseline_throughput) / len(baseline_throughput)
    
    # Sparsity + MOR measurements
    sparse_latencies = []
    sparse_throughput = []
    sparse_outputs = []
    similarities = []
    
    pipeline = SparMoRPipeline(
        model=model,
        sparsity_k=config.get("sparsity_k", 200)
    )
    
    with torch.no_grad():
        for inputs, baseline_out in zip(eval_inputs, baseline_outputs):
            # Measure sparse + MOR
            latency = measure_latency(pipeline, inputs)
            sparse_latencies.append(latency)
            
            max_mem, current_mem = measure_gpu_memory(pipeline)
            results["sparse_mor"]["max_memory"] = max_mem
            results["sparse_mor"]["current_memory"] = current_mem
            
            outputs = pipeline.forward(**inputs)
            sparse_outputs.append(outputs.logits)
            
            # Calculate metrics
            sim = cosine_similarity(outputs.logits, baseline_out)
            similarities.append(sim)
            
            num_tokens = inputs["input_ids"].numel()
            throughput = num_tokens / (latency / 1000)
            sparse_throughput.append(throughput)
            
    avg_latency = sum(sparse_latencies) / len(sparse_latencies)
    avg_throughput = sum(sparse_throughput) / len(sparse_throughput)
    avg_similarity = sum(similarities) / len(similarities)
    
    results["sparse_mor"]["avg_latency"] = avg_latency
    results["sparse_mor"]["avg_throughput"] = avg_throughput
    results["sparse_mor"]["avg_similarity"] = avg_similarity
    
    # Calculate comparisons
    speedup = results["baseline"]["avg_latency"] / avg_latency
    mem_reduction = 1 - (results["sparse_mor"]["max_memory"] / results["baseline"]["max_memory"])
    
    results["comparison"]["speedup"] = speedup
    results["comparison"]["memory_reduction"] = mem_reduction
    results["comparison"]["speedup_x_similarity"] = speedup * avg_similarity
    
    # Generate interactive dashboard and plots
    sparse_outputs_sample = sparse_outputs[0] if sparse_outputs else None
    baseline_outputs_sample = baseline_outputs[0] if baseline_outputs else None
    
    if sparse_outputs_sample is not None and baseline_outputs_sample is not None:
        # Get sparsity mask from a sample forward pass
        _, mask = pipeline.apply_sparsity(baseline_outputs_sample)
    else:
        mask = None
    
    # Generate dashboard with all visualizations
    generate_html_dashboard(
        results=results,
        output_dir=output_dir,
        mask=mask
    )
    
    # Save results JSON
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results