"""
Inference-Time Sparsification Experiment Script

Benchmarks transformer models (BERT, DistilBERT, RoBERTa) with various
sparsification strategies from the sparsifier package. Measures latency,
memory usage, and reconstruction quality.

Usage:
    python experiment.py --model bert-base-uncased --mode topk --k 32
    python experiment.py --model distilbert-base-uncased --benchmark-all
    python experiment.py --interactive

Author: Madhan
Date: 2025-10-12
"""

import argparse
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Check for required packages
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error: Missing required package. Please install: {e}")
    print("Run: pip install torch transformers matplotlib seaborn")
    sys.exit(1)

# Import sparsifier modules
try:
    from sparsifier import Sparsifier, compute_sparsity_stats, compute_reconstruction_error
    from sparsifier import visualize_sparsity_pattern, estimate_memory_savings
    
    
except ImportError:
    print("Error: Cannot import sparsifier package.")
    sys.exit(1)


# ============================================================================
# Configuration Presets (Based on demo_op.md results)
# ============================================================================

SPARSITY_PRESETS = {
    "topk_aggressive": {
        "mode": "topk",
        "config": {"k": 12, "renorm": True},
        "description": "Top-K 5% (aggressive sparsity, 90% reduction)"
    },
    "topk_moderate": {
        "mode": "topk", 
        "config": {"k": 32, "renorm": True},
        "description": "Top-K 25% (balanced sparsity/quality)"
    },
    "topk_conservative": {
        "mode": "topk",
        "config": {"k": 64, "renorm": True},
        "description": "Top-K 50% (conservative, high quality)"
    },
    "threshold": {
        "mode": "threshold",
        "config": {"threshold": 0.5, "renorm": True},
        "description": "Absolute threshold 0.5 (37.5% sparse from demo)"
    },
    "adaptive": {
        "mode": "adaptive_threshold",
        "config": {"alpha": 0.3, "renorm": True},
        "description": "Adaptive 30% (59.2% sparse from demo)"
    },
    "block": {
        "mode": "block",
        "config": {"block_size": 8, "keep_ratio": 0.5, "renorm": True},
        "description": "Block sparsity (50% sparse from demo)"
    },
    "hybrid": {
        "mode": "hybrid",
        "config": {"hybrid_static_keep_ratio": 0.5, "k": 16, "renorm": True},
        "description": "Hybrid static+dynamic (75% sparse from demo)"
    }
}


# ============================================================================
# Model Wrapper
# ============================================================================

class SparseModelWrapper:
    """Wraps a transformer model with sparsification applied to embeddings."""
    
    def __init__(self, model_name: str, sparsifier: Optional[Sparsifier] = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.sparsifier = sparsifier
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_texts(self, texts: List[str], apply_sparsity: bool = True) -> Tuple[np.ndarray, Dict]:
        """Encode texts and optionally apply sparsification.
        
        Returns:
            embeddings: (N, D) numpy array
            metrics: dict with timing and memory stats
        """
        metrics = {}
        
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               return_tensors="pt", max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["forward_time"] = time.perf_counter() - start
        
        # Store original for comparison
        embeddings_original = embeddings.copy()
        metrics["original_shape"] = embeddings.shape
        
        # Apply sparsification
        if apply_sparsity and self.sparsifier is not None:
            start = time.perf_counter()
            embeddings = self.sparsifier.apply_to_batch(embeddings)
            metrics["sparsify_time"] = time.perf_counter() - start
            
            # Compute reconstruction error
            metrics["mse"] = compute_reconstruction_error(embeddings_original, embeddings, "mse")
            metrics["mae"] = compute_reconstruction_error(embeddings_original, embeddings, "mae")
            metrics["cosine_dist"] = compute_reconstruction_error(embeddings_original, embeddings, "cosine")
            
            # Sparsity stats
            stats = compute_sparsity_stats(embeddings)
            metrics.update(stats)
            
            # Memory savings
            mem = estimate_memory_savings(embeddings_original, embeddings)
            metrics["memory_savings_ratio"] = mem["savings_ratio"]
            metrics["compression_ratio"] = mem["compression_ratio"]
        else:
            metrics["sparsify_time"] = 0.0
            metrics["mse"] = 0.0
            metrics["nonzero_elements"] = embeddings.size
            metrics["sparsity_fraction"] = 0.0
        
        metrics["total_time"] = metrics["forward_time"] + metrics["sparsify_time"]
        
        return embeddings, metrics
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        mem_info = {}
        if torch.cuda.is_available():
            mem_info["allocated_mb"] = torch.cuda.memory_allocated(self.device) / 1024**2
            mem_info["reserved_mb"] = torch.cuda.memory_reserved(self.device) / 1024**2
            mem_info["max_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024**2
        return mem_info


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Orchestrates experiments across different sparsity configurations."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.results = []
    
    def run_single_experiment(self, texts: List[str], preset_name: str, 
                             num_runs: int = 5) -> Dict:
        """Run experiment with a single sparsity configuration."""
        preset = SPARSITY_PRESETS[preset_name]
        print(f"\n{'='*70}")
        print(f"Experiment: {preset_name}")
        print(f"Description: {preset['description']}")
        print(f"{'='*70}")
        
        # Create model wrapper with sparsifier
        model = SparseModelWrapper(self.model_name, device=self.device)
        D = model.embedding_dim
        
        sparsifier = Sparsifier(D, mode=preset["mode"], config=preset["config"])
        
        # Handle hybrid mode - needs static mask
        if preset["mode"] == "hybrid":
            sparsifier.create_random_static_mask(preset["config"].get("hybrid_static_keep_ratio", 0.5))
        
        model.sparsifier = sparsifier
        
        # Warm-up run
        print("Warming up...")
        model.encode_texts(texts[:2], apply_sparsity=True)
        
        # Multiple runs for averaging
        print(f"Running {num_runs} trials...")
        run_metrics = []
        
        for i in range(num_runs):
            embeddings, metrics = model.encode_texts(texts, apply_sparsity=True)
            run_metrics.append(metrics)
            print(f"  Run {i+1}/{num_runs}: "
                  f"Total={metrics['total_time']*1000:.2f}ms, "
                  f"Sparse={metrics['sparsity_fraction']*100:.1f}%, "
                  f"MSE={metrics['mse']:.6f}")
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(run_metrics)
        aggregated["preset_name"] = preset_name
        aggregated["description"] = preset["description"]
        aggregated["mode"] = preset["mode"]
        aggregated["config"] = preset["config"]
        
        # Get final embeddings for visualization
        embeddings, _ = model.encode_texts(texts[:10], apply_sparsity=True)
        aggregated["sample_embeddings"] = embeddings
        
        # Memory info
        aggregated["memory_info"] = model.get_memory_usage()
        
        self.results.append(aggregated)
        
        self._print_summary(aggregated)
        
        return aggregated
    
    def run_benchmark_sweep(self, texts: List[str], presets: Optional[List[str]] = None,
                           num_runs: int = 5) -> List[Dict]:
        """Run experiments across multiple sparsity configurations."""
        if presets is None:
            presets = list(SPARSITY_PRESETS.keys())
        
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK SWEEP: {len(presets)} configurations")
        print(f"# Model: {self.model_name}")
        print(f"# Samples: {len(texts)}, Runs per config: {num_runs}")
        print(f"{'#'*70}\n")
        
        for preset_name in presets:
            self.run_single_experiment(texts, preset_name, num_runs)
        
        return self.results
    
    def _aggregate_metrics(self, run_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across multiple runs."""
        agg = {}
        
        # Numerical metrics to average
        num_keys = ["forward_time", "sparsify_time", "total_time", "mse", "mae", 
                   "cosine_dist", "sparsity_fraction", "nonzero_elements",
                   "memory_savings_ratio", "compression_ratio"]
        
        for key in num_keys:
            values = [m.get(key, 0.0) for m in run_metrics]
            agg[f"{key}_mean"] = np.mean(values)
            agg[f"{key}_std"] = np.std(values)
            agg[f"{key}_min"] = np.min(values)
            agg[f"{key}_max"] = np.max(values)
        
        return agg
    
    def _print_summary(self, metrics: Dict):
        """Print summary of experiment results."""
        print(f"\n{'-'*70}")
        print("RESULTS SUMMARY")
        print(f"{'-'*70}")
        print(f"Sparsity:           {metrics['sparsity_fraction_mean']*100:.1f}% ± {metrics['sparsity_fraction_std']*100:.1f}%")
        print(f"Total Time:         {metrics['total_time_mean']*1000:.2f}ms ± {metrics['total_time_std']*1000:.2f}ms")
        print(f"  - Forward:        {metrics['forward_time_mean']*1000:.2f}ms")
        print(f"  - Sparsify:       {metrics['sparsify_time_mean']*1000:.2f}ms")
        print(f"Reconstruction MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"Cosine Distance:    {metrics['cosine_dist_mean']:.6f}")
        print(f"Memory Savings:     {metrics['memory_savings_ratio_mean']*100:.1f}%")
        print(f"Compression Ratio:  {metrics['compression_ratio_mean']:.2f}x")
        print(f"{'-'*70}\n")
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        # Remove numpy arrays for JSON serialization
        results_clean = []
        for r in self.results:
            r_clean = {k: v for k, v in r.items() if k != "sample_embeddings"}
            results_clean.append(r_clean)
        
        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    def generate_visualizations(self, output_dir: str = "experiment_outputs"):
        """Generate comparison visualizations."""
        Path(output_dir).mkdir(exist_ok=True)
        
        if len(self.results) < 2:
            print("Need at least 2 experiments for comparison plots.")
            return
        
        print(f"\nGenerating visualizations in: {output_dir}/")
        
        # Extract data
        names = [r["preset_name"] for r in self.results]
        sparsity = [r["sparsity_fraction_mean"] * 100 for r in self.results]
        latency = [r["total_time_mean"] * 1000 for r in self.results]
        mse = [r["mse_mean"] for r in self.results]
        memory_savings = [r["memory_savings_ratio_mean"] * 100 for r in self.results]
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Sparsity vs Latency
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sparsity, latency, s=100, alpha=0.6)
        for i, name in enumerate(names):
            ax.annotate(name, (sparsity[i], latency[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel("Sparsity (%)", fontsize=12)
        ax.set_ylabel("Total Latency (ms)", fontsize=12)
        ax.set_title(f"Sparsity vs Latency - {self.model_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sparsity_vs_latency.png", dpi=150)
        print("sparsity_vs_latency.png")
        plt.close()
        
        # 2. Sparsity vs MSE
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sparsity, mse, s=100, alpha=0.6, c='coral')
        for i, name in enumerate(names):
            ax.annotate(name, (sparsity[i], mse[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel("Sparsity (%)", fontsize=12)
        ax.set_ylabel("Reconstruction MSE", fontsize=12)
        ax.set_title(f"Sparsity vs Quality - {self.model_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sparsity_vs_mse.png", dpi=150)
        print("sparsity_vs_mse.png")
        plt.close()
        
        # 3. Bar chart comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sparsity
        axes[0, 0].barh(names, sparsity, color='steelblue')
        axes[0, 0].set_xlabel("Sparsity (%)")
        axes[0, 0].set_title("Sparsity Level")
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Latency
        axes[0, 1].barh(names, latency, color='coral')
        axes[0, 1].set_xlabel("Latency (ms)")
        axes[0, 1].set_title("Total Inference Time")
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # MSE
        axes[1, 0].barh(names, mse, color='mediumpurple')
        axes[1, 0].set_xlabel("MSE")
        axes[1, 0].set_title("Reconstruction Error")
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Memory Savings
        axes[1, 1].barh(names, memory_savings, color='seagreen')
        axes[1, 1].set_xlabel("Memory Savings (%)")
        axes[1, 1].set_title("Memory Reduction")
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f"Sparsity Benchmark Comparison - {self.model_name}", 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/benchmark_comparison.png", dpi=150)
        print("benchmark_comparison.png")
        plt.close()
        
        # 4. Sparsity pattern visualization for first result
        if "sample_embeddings" in self.results[0]:
            embeddings = self.results[0]["sample_embeddings"]
            pattern_str = visualize_sparsity_pattern(embeddings[:5], max_rows=5, max_cols=80)
            
            with open(f"{output_dir}/sparsity_pattern.txt", 'w') as f:
                f.write(f"Sparsity Pattern - {self.results[0]['preset_name']}\n")
                f.write("="*70 + "\n")
                f.write(pattern_str)
            print("sparsity_pattern.txt")
        
        print(f"\nVisualization complete!")


# ============================================================================
# CLI and Interactive Mode
# ============================================================================

def interactive_mode():
    """Interactive CLI for running experiments."""
    print("\n" + "="*70)
    print("INFERENCE-TIME SPARSIFICATION EXPERIMENT (Interactive Mode)")
    print("="*70 + "\n")
    
    # Select model
    available_models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            choice = input("\nSelect model (1-4): ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(available_models):
                model_name = available_models[model_idx]
                break
            else:
                print("Invalid choice. Please select 1-4.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Select experiment type
    print("\nExperiment types:")
    print("  1. Single sparsity configuration")
    print("  2. Full benchmark sweep (all configurations)")
    
    while True:
        try:
            choice = input("\nSelect experiment type (1-2): ").strip()
            exp_type = int(choice)
            if exp_type in [1, 2]:
                break
            else:
                print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Select sparsity preset (for single config)
    if exp_type == 1:
        print("\nAvailable sparsity presets:")
        preset_list = list(SPARSITY_PRESETS.keys())
        for i, preset in enumerate(preset_list, 1):
            desc = SPARSITY_PRESETS[preset]["description"]
            print(f"  {i}. {preset:20s} - {desc}")
        
        while True:
            try:
                choice = input(f"\nSelect preset (1-{len(preset_list)}): ").strip()
                preset_idx = int(choice) - 1
                if 0 <= preset_idx < len(preset_list):
                    selected_preset = preset_list[preset_idx]
                    break
                else:
                    print(f"Invalid choice. Please select 1-{len(preset_list)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Number of runs
    while True:
        try:
            num_runs = int(input("\nNumber of runs per config (default 5): ").strip() or "5")
            if num_runs > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Sparsification reduces computational costs while maintaining accuracy.",
        "Transformers have revolutionized NLP tasks.",
        "Efficient inference is crucial for production systems.",
        "Model compression techniques include pruning and quantization.",
        "Embeddings capture semantic meaning of words and sentences.",
        "Optimization improves model performance and reduces latency."
    ]
    
    # Run experiment
    runner = ExperimentRunner(model_name, device=device)
    
    if exp_type == 1:
        runner.run_single_experiment(sample_texts, selected_preset, num_runs)
    else:
        runner.run_benchmark_sweep(sample_texts, num_runs=num_runs)
    
    # Save and visualize
    output_dir = f"experiment_outputs_{model_name.replace('/', '_')}"
    runner.save_results(f"{output_dir}/results.json")
    runner.generate_visualizations(output_dir)
    
    print(f"\n{'='*70}")
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inference-time sparsification experiments for transformer models"
    )
    
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model name from HuggingFace (default: bert-base-uncased)")
    parser.add_argument("--preset", type=str, choices=list(SPARSITY_PRESETS.keys()),
                       help="Sparsity preset to use")
    parser.add_argument("--benchmark-all", action="store_true",
                       help="Run benchmark sweep across all presets")
    parser.add_argument("--num-runs", type=int, default=5,
                       help="Number of runs per configuration (default: 5)")
    parser.add_argument("--output-dir", type=str, default="experiment_outputs",
                       help="Output directory for results (default: experiment_outputs)")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                       help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Auto-detect device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample texts
    sample_texts = [
        "this is apple",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Sparsification reduces computational costs while maintaining accuracy.",
        "Transformers have revolutionized NLP tasks.",
        "Efficient inference is crucial for production systems.",
        "Model compression techniques include pruning and quantization.",
        "Embeddings capture semantic meaning of words and sentences.",
        "Optimization improves model performance and reduces latency."
    ]
    
    # Create runner
    runner = ExperimentRunner(args.model, device=device)
    
    # Run experiments
    if args.benchmark_all:
        runner.run_benchmark_sweep(sample_texts, num_runs=args.num_runs)
    elif args.preset:
        runner.run_single_experiment(sample_texts, args.preset, num_runs=args.num_runs)
    else:
        print("Error: Specify either --preset or --benchmark-all")
        parser.print_help()
        return
    
    # Save and visualize
    output_dir = args.output_dir + f"_{args.model.replace('/', '_')}"
    runner.save_results(f"{output_dir}/results.json")
    runner.generate_visualizations(output_dir)
    
    print(f"\n{'='*70}")
    print("Experiment complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()