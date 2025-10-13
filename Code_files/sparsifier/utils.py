"""
Utility functions for sparsifier module.

Provides helper functions for array manipulation, validation,
and common operations used across the sparsifier modules.

Author: Madhan
Date: 2025-10-12
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure input is 2D array.
    
    Converts 1D arrays to (1, D) shape for batch processing.
    
    Args:
        x: input array
    Returns:
        2D array (N, D)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")


def compute_sparsity_stats(X: np.ndarray) -> dict:
    """Compute detailed sparsity statistics.
    
    Args:
        X: input array (any shape)
    Returns:
        Dictionary with sparsity metrics
    """
    X = np.asarray(X)
    total = X.size
    nonzeros = np.count_nonzero(X)
    zeros = total - nonzeros
    
    # Per-row stats if 2D
    if X.ndim == 2:
        row_nonzeros = np.count_nonzero(X, axis=1)
        row_sparsity = 1.0 - (row_nonzeros / X.shape[1])
        per_row_stats = {
            "mean_nonzeros_per_row": float(np.mean(row_nonzeros)),
            "std_nonzeros_per_row": float(np.std(row_nonzeros)),
            "min_nonzeros_per_row": int(np.min(row_nonzeros)),
            "max_nonzeros_per_row": int(np.max(row_nonzeros)),
            "mean_sparsity_per_row": float(np.mean(row_sparsity))
        }
    else:
        per_row_stats = {}
    
    return {
        "total_elements": int(total),
        "nonzero_elements": int(nonzeros),
        "zero_elements": int(zeros),
        "sparsity_fraction": float(zeros / total),
        "density_fraction": float(nonzeros / total),
        **per_row_stats
    }


def compute_reconstruction_error(X_orig: np.ndarray, X_recon: np.ndarray,
                                 metric: str = "mse") -> float:
    """Compute reconstruction error between original and reconstructed.
    
    Args:
        X_orig: original array
        X_recon: reconstructed array
        metric: "mse", "mae", or "cosine"
    Returns:
        Error value
    """
    X_orig = np.asarray(X_orig).flatten()
    X_recon = np.asarray(X_recon).flatten()
    
    if metric == "mse":
        return float(np.mean((X_orig - X_recon) ** 2))
    elif metric == "mae":
        return float(np.mean(np.abs(X_orig - X_recon)))
    elif metric == "cosine":
        # Cosine similarity (1 - similarity = distance)
        norm_orig = np.linalg.norm(X_orig)
        norm_recon = np.linalg.norm(X_recon)
        if norm_orig == 0 or norm_recon == 0:
            return 1.0
        similarity = np.dot(X_orig, X_recon) / (norm_orig * norm_recon)
        return float(1.0 - similarity)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def visualize_sparsity_pattern(X: np.ndarray, max_rows: int = 10,
                               max_cols: int = 50) -> str:
    """Create ASCII visualization of sparsity pattern.
    
    Args:
        X: 2D array to visualize
        max_rows: maximum rows to display
        max_cols: maximum columns to display
    Returns:
        String representation
    """
    X = ensure_2d(X)
    N, D = X.shape
    
    # Truncate if too large
    X_vis = X[:min(N, max_rows), :min(D, max_cols)]
    N_vis, D_vis = X_vis.shape
    
    lines = [f"Sparsity Pattern (showing {N_vis}×{D_vis} of {N}×{D}):"]
    lines.append("  " + "".join([str(i % 10) for i in range(D_vis)]))
    
    for i in range(N_vis):
        row_vis = ""
        for j in range(D_vis):
            if X_vis[i, j] != 0:
                row_vis += "█"
            else:
                row_vis += "·"
        lines.append(f"{i:2d}" + row_vis)
    
    if N > max_rows or D > max_cols:
        lines.append(f"  ... (truncated from {N}×{D})")
    
    return "\n".join(lines)


def compare_sparsity_modes(X: np.ndarray, sparse_arrays: dict,
                          labels: Optional[list] = None) -> str:
    """Compare multiple sparsification results.
    
    Args:
        X: original dense array
        sparse_arrays: dict mapping mode names to sparse arrays
        labels: optional custom labels
    Returns:
        Formatted comparison string
    """
    X = ensure_2d(X)
    lines = ["Sparsity Mode Comparison:", "=" * 60]
    
    for mode_name, X_sparse in sparse_arrays.items():
        stats = compute_sparsity_stats(X_sparse)
        mse = compute_reconstruction_error(X, X_sparse, "mse")
        
        lines.append(f"\n{mode_name}:")
        lines.append(f"  Sparsity: {stats['sparsity_fraction']:.1%}")
        lines.append(f"  Nonzeros: {stats['nonzero_elements']}/{stats['total_elements']}")
        lines.append(f"  MSE: {mse:.6f}")
        
        if "mean_nonzeros_per_row" in stats:
            lines.append(f"  Avg nonzeros/row: {stats['mean_nonzeros_per_row']:.1f}")
    
    return "\n".join(lines)


def validate_config(config: dict, required_keys: list) -> None:
    """Validate configuration dictionary has required keys.
    
    Args:
        config: configuration dict
        required_keys: list of required key names
    Raises:
        ValueError if any required key is missing
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def estimate_memory_savings(X_orig: np.ndarray, X_sparse: np.ndarray) -> dict:
    """Estimate memory savings from sparsification.
    
    Assumes conversion to sparse format (CSR).
    
    Args:
        X_orig: original dense array
        X_sparse: sparsified array
    Returns:
        Dictionary with memory estimates
    """
    dense_bytes = X_orig.nbytes
    
    # For CSR format: data + indices + indptr
    nonzeros = np.count_nonzero(X_sparse)
    if X_sparse.ndim == 2:
        # data (float32) + indices (int32) + indptr (int32 × rows)
        sparse_bytes = nonzeros * 4 + nonzeros * 4 + (X_sparse.shape[0] + 1) * 4
    else:
        sparse_bytes = nonzeros * 4 + nonzeros * 4 + 8
    
    return {
        "dense_bytes": int(dense_bytes),
        "sparse_bytes": int(sparse_bytes),
        "savings_bytes": int(dense_bytes - sparse_bytes),
        "savings_ratio": float((dense_bytes - sparse_bytes) / dense_bytes),
        "compression_ratio": float(dense_bytes / sparse_bytes) if sparse_bytes > 0 else float('inf')
    }


# Example usage
if __name__ == "__main__":
    print("=== Utility Functions Example ===\n")
    
    # Create sample data
    rng = np.random.RandomState(42)
    X = rng.randn(5, 20).astype(np.float32)
    
    # Create sparse version (keep top 30%)
    X_sparse = X.copy()
    k = int(0.3 * 20)
    for i in range(5):
        threshold = np.partition(np.abs(X[i]), -k)[-k]
        X_sparse[i] = np.where(np.abs(X[i]) >= threshold, X[i], 0)
    
    print("1. Ensure 2D:")
    vec_1d = np.array([1, 2, 3])
    vec_2d = ensure_2d(vec_1d)
    print(f"   1D shape {vec_1d.shape} -> 2D shape {vec_2d.shape}\n")
    
    print("2. Sparsity Statistics:")
    stats = compute_sparsity_stats(X_sparse)
    print(f"   Total elements: {stats['total_elements']}")
    print(f"   Nonzeros: {stats['nonzero_elements']}")
    print(f"   Sparsity: {stats['sparsity_fraction']:.1%}")
    print(f"   Mean nonzeros/row: {stats['mean_nonzeros_per_row']:.1f}\n")
    
    print("3. Reconstruction Error:")
    mse = compute_reconstruction_error(X, X_sparse, "mse")
    mae = compute_reconstruction_error(X, X_sparse, "mae")
    cosine = compute_reconstruction_error(X, X_sparse, "cosine")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   Cosine distance: {cosine:.6f}\n")
    
    print("4. Sparsity Pattern Visualization:")
    print(visualize_sparsity_pattern(X_sparse, max_rows=5, max_cols=20))
    print()
    
    print("5. Memory Savings Estimate:")
    mem = estimate_memory_savings(X, X_sparse)
    print(f"   Dense: {mem['dense_bytes']} bytes")
    print(f"   Sparse (CSR): {mem['sparse_bytes']} bytes")
    print(f"   Savings: {mem['savings_ratio']:.1%}")
    print(f"   Compression: {mem['compression_ratio']:.2f}x\n")
    
    print("6. Mode Comparison:")
    # Create different sparse versions
    X_topk = X.copy()
    for i in range(5):
        threshold = np.partition(np.abs(X[i]), -5)[-5]
        X_topk[i] = np.where(np.abs(X[i]) >= threshold, X[i], 0)
    
    X_thresh = np.where(np.abs(X) >= 0.5, X, 0)
    
    comparison = compare_sparsity_modes(X, {
        "Top-5": X_topk,
        "Threshold 0.5": X_thresh,
        "Top-30%": X_sparse
    })
    print(comparison)