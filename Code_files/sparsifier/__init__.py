"""
Sparsifier package - NumPy-only inference-time sparsity for embeddings.

Provides tools for applying various sparsification strategies to encoded
token vectors, with compensation techniques to preserve information.

Main components:
- Sparsifier: main class orchestrating all operations
- MaskManager: static mask creation/management
- SparsityModes: different sparsification algorithms
- CompensationManager: normalization and reconstruction
- utils: helper functions

Author:Madhan
Date: 2025-10-12
"""

from .core import Sparsifier
from .masks import MaskManager
from .modes import SparsityModes
from .compensation import CompensationManager
from .utils import (
    ensure_2d,
    compute_sparsity_stats,
    compute_reconstruction_error,
    visualize_sparsity_pattern,
    compare_sparsity_modes,
    validate_config,
    estimate_memory_savings
)

__version__ = "0.1.0"
__author__ = "Madhan"

__all__ = [
    # Main class
    "Sparsifier",
    
    # Sub-managers
    "MaskManager",
    "SparsityModes", 
    "CompensationManager",
    
    # Utility functions
    "ensure_2d",
    "compute_sparsity_stats",
    "compute_reconstruction_error",
    "visualize_sparsity_pattern",
    "compare_sparsity_modes",
    "validate_config",
    "estimate_memory_savings",
]


# Quick usage example
def quick_start_example():
    """Demonstrates basic sparsifier usage."""
    import numpy as np
    
    print("=== Quick Start Example ===\n")
    
    # Generate sample embeddings
    D = 128  # embedding dimension
    N = 10   # batch size
    X = np.random.randn(N, D).astype(np.float32)
    
    # Create sparsifier with top-k mode
    sparsifier = Sparsifier(
        D=D,
        mode="topk",
        config={"k": 32, "renorm": True}
    )
    
    # Apply sparsification
    X_sparse = sparsifier.apply_to_batch(X)
    
    # Show results
    stats = sparsifier.stats(X)
    print(f"Input: {N} vectors of dimension {D}")
    print(f"Sparsity: {stats['sparsity_fraction']:.1%}")
    print(f"Nonzeros: {stats['nonzeros']}/{stats['total']}")
    
    # Calibrate for better reconstruction
    sparsifier.calibrate(X, calibrate_gamma=True)
    X_calibrated = sparsifier.apply_to_batch(X)
    
    mse = np.mean((X - X_calibrated) ** 2)
    print(f"\nReconstruction MSE: {mse:.6f}")


if __name__ == "__main__":
    quick_start_example()