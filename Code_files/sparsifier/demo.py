"""
Comprehensive demonstration of the sparsifier package.

Shows end-to-end workflows for different use cases:
1. Basic sparsification with different modes
2. Static mask workflows (create, save, load)
3. Calibration and compensation strategies
4. Performance comparison across modes

Author: ChatGPT (for Madhan)
Date: 2025-10-12
"""

import numpy as np
import tempfile
import os

from .utils import compare_sparsity_modes,visualize_sparsity_pattern,estimate_memory_savings
from .core import Sparsifier


def demo_basic_modes():
    """Demonstrate different sparsification modes."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Sparsification Modes")
    print("="*70)
    
    D = 64
    N = 8
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    
    print(f"\nInput: {N} vectors × {D} dimensions")
    print(f"Total elements: {N * D}\n")
    
    # Test different modes
    modes_config = {
        "top-k (k=16)": {"mode": "topk", "config": {"k": 16}},
        "threshold (0.5)": {"mode": "threshold", "config": {"threshold": 0.5}},
        "adaptive (30%)": {"mode": "adaptive_threshold", "config": {"alpha": 0.3}},
        "block (8×4)": {"mode": "block", "config": {"block_size": 8, "blocks_keep": 4}},
    }
    
    results = {}
    for name, spec in modes_config.items():
        sp = Sparsifier(D, mode=spec["mode"], config=spec["config"])
        X_sparse = sp.apply_to_batch(X)
        results[name] = X_sparse
        
        stats = sp.stats(X)
        print(f"{name:20s} → {stats['nonzeros']:4d} nonzeros ({stats['sparsity_fraction']*100:5.1f}% sparse)")
    
    print("\n" + compare_sparsity_modes(X, results))


def demo_static_masks():
    """Demonstrate static mask workflows."""
    print("\n" + "="*70)
    print("DEMO 2: Static Mask Management")
    print("="*70)
    
    D = 32
    N = 4
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    
    sp = Sparsifier(D, mode="static", config={"keep_ratio": 0.5})
    
    # Create random mask
    print("\n1. Creating random mask (50% keep ratio)")
    mask = sp.create_random_static_mask(0.5)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Elements kept: {int(np.sum(mask))}/{D}")
    print(f"   First 16 mask values: {mask[:16].astype(int)}")
    
    # Apply mask
    X_masked = sp.apply_to_batch(X)
    print(f"\n2. Applied mask to {N} vectors")
    print(f"   Nonzeros per row: {[np.count_nonzero(X_masked[i]) for i in range(N)]}")
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        mask_path = os.path.join(tmpdir, "my_mask.npy")
        sp.save_mask(mask_path)
        print(f"\n3. Saved mask to {mask_path}")
        
        # Create new sparsifier and load
        sp2 = Sparsifier(D, mode="static")
        sp2.load_mask(mask_path)
        X_masked2 = sp2.apply_to_batch(X)
        
        match = np.allclose(X_masked, X_masked2)
        print(f"   Loaded mask produces same result: {match}")
    
    # Visualize pattern
    print("\n4. Sparsity pattern:")
    print(visualize_sparsity_pattern(X_masked, max_rows=4, max_cols=32))


def demo_calibration():
    """Demonstrate calibration and compensation."""
    print("\n" + "="*70)
    print("DEMO 3: Calibration & Compensation")
    print("="*70)
    
    D = 128
    N_train = 50
    N_test = 10
    rng = np.random.RandomState(42)
    
    X_train = rng.randn(N_train, D).astype(np.float32)
    X_test = rng.randn(N_test, D).astype(np.float32)
    
    sp = Sparsifier(D, mode="topk", config={"k": 32, "renorm": True})
    
    print("\n1. Baseline (no calibration):")
    X_sparse = sp.apply_to_batch(X_test)
    mse_baseline = np.mean((X_test - X_sparse) ** 2)
    print(f"   MSE: {mse_baseline:.6f}")
    
    print("\n2. With gamma calibration:")
    sp.calibrate(X_train, calibrate_gamma=True)
    X_gamma = sp.apply_to_batch(X_test)
    mse_gamma = np.mean((X_test - X_gamma) ** 2)
    print(f"   Calibrated gamma: {sp.compensation.gamma:.4f}")
    print(f"   MSE: {mse_gamma:.6f}")
    print(f"   Improvement: {(1 - mse_gamma/mse_baseline)*100:.1f}%")
    
    print("\n3. With reconstructor:")
    sp_recon = Sparsifier(D, mode="topk", config={
        "k": 32, 
        "renorm": True,
        "reconstructor_hidden": 64,
        "reconstructor_train_steps": 150
    })
    sp_recon.calibrate(X_train, calibrate_gamma=True, calibrate_reconstructor=True)
    X_recon = sp_recon.apply_to_batch(X_test, reconstruct=True)
    mse_recon = np.mean((X_test - X_recon) ** 2)
    print(f"   MSE: {mse_recon:.6f}")
    print(f"   Improvement: {(1 - mse_recon/mse_baseline)*100:.1f}%")
    
    print("\n4. Summary:")
    print(f"   Baseline MSE:       {mse_baseline:.6f}")
    print(f"   + Gamma:            {mse_gamma:.6f}  ({(mse_gamma/mse_baseline)*100:.1f}% of baseline)")
    print(f"   + Reconstructor:    {mse_recon:.6f}  ({(mse_recon/mse_baseline)*100:.1f}% of baseline)")


def demo_performance_comparison():
    """Compare performance across different sparsity levels."""
    print("\n" + "="*70)
    print("DEMO 4: Performance vs Sparsity Trade-off")
    print("="*70)
    
    D = 256
    N = 20
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    
    print(f"\nTesting with {N} vectors × {D} dimensions")
    print(f"\nSparsity Level    Nonzeros    MSE        Memory Savings")
    print("-" * 65)
    
    keep_ratios = [0.05, 0.10, 0.25, 0.50, 0.75]
    
    for ratio in keep_ratios:
        k = int(D * ratio)
        sp = Sparsifier(D, mode="topk", config={"k": k, "renorm": True})
        X_sparse = sp.apply_to_batch(X)
        
        stats = sp.stats(X)
        mse = np.mean((X - X_sparse) ** 2)
        mem = estimate_memory_savings(X, X_sparse)
        
        print(f"{ratio*100:5.0f}% ({k:3d}/dim)  "
              f"{stats['nonzeros']:6d}      "
              f"{mse:8.6f}   "
              f"{mem['savings_ratio']*100:5.1f}%")


def demo_hybrid_mode():
    """Demonstrate hybrid static + dynamic sparsification."""
    print("\n" + "="*70)
    print("DEMO 5: Hybrid Static + Dynamic Sparsification")
    print("="*70)
    
    D = 64
    N = 6
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    
    print(f"\nInput: {N} vectors × {D} dimensions")
    
    # Create hybrid sparsifier
    sp = Sparsifier(D, mode="hybrid", config={
        "hybrid_static_keep_ratio": 0.5,  # First keep 50% of dimensions
        "k": 16,  # Then keep top-16 within those
        "renorm": True
    })
    
    print("\n1. Creating static coarse mask (50% of dimensions)")
    sp.create_random_static_mask(0.5)
    static_kept = int(np.sum(sp.mask_manager.static_mask))
    print(f"   Static mask keeps: {static_kept}/{D} dimensions")
    
    print(f"\n2. Within those {static_kept} dims, keep top-16")
    X_hybrid = sp.apply_to_batch(X)
    
    stats = sp.stats(X)
    print(f"   Final nonzeros: {stats['nonzeros']}")
    print(f"   Expected: ~{N * 16} (6 rows × 16 per row)")
    print(f"   Sparsity: {stats['sparsity_fraction']:.1%}")
    
    # Compare to pure top-k
    sp_topk = Sparsifier(D, mode="topk", config={"k": 16, "renorm": True})
    X_topk = sp_topk.apply_to_batch(X)
    
    print("\n3. Comparison with pure top-k:")
    mse_hybrid = np.mean((X - X_hybrid) ** 2)
    mse_topk = np.mean((X - X_topk) ** 2)
    print(f"   Hybrid MSE: {mse_hybrid:.6f}")
    print(f"   Top-K MSE:  {mse_topk:.6f}")
    print(f"   Difference: {abs(mse_hybrid - mse_topk):.6f}")
    
    print("\n4. Visualize first 2 rows:")
    print("\n   Hybrid mode:")
    print(visualize_sparsity_pattern(X_hybrid, max_rows=2, max_cols=64))
    print("\n   Pure top-k mode:")
    print(visualize_sparsity_pattern(X_topk, max_rows=2, max_cols=64))


def demo_explain():
    """Demonstrate explainability features."""
    print("\n" + "="*70)
    print("DEMO 6: Explainability - Understanding Individual Sparsification")
    print("="*70)
    
    D = 32
    rng = np.random.RandomState(42)
    
    # Create a single vector with known structure
    x = np.zeros(D, dtype=np.float32)
    x[0] = 2.5   # Large positive
    x[5] = -2.0  # Large negative
    x[10] = 1.5
    x[15] = -1.0
    x[20] = 0.8
    x[25] = 0.5
    x[30] = 0.3
    
    print(f"\nInput vector: {D} dimensions")
    print(f"Notable values at indices: 0, 5, 10, 15, 20, 25, 30")
    
    # Test with top-k
    print("\n1. Top-K mode (k=5):")
    sp_k = Sparsifier(D, mode="topk", config={"k": 5})
    explain_k = sp_k.explain_row(x)
    print(f"   Kept {explain_k['kept_count']} elements")
    print(f"   Kept indices: {explain_k['kept_indices']}")
    print(f"   Original values: {[f'{x[i]:.2f}' for i in explain_k['kept_indices']]}")
    
    # Test with threshold
    print("\n2. Threshold mode (tau=1.0):")
    sp_t = Sparsifier(D, mode="threshold", config={"threshold": 1.0})
    explain_t = sp_t.explain_row(x)
    print(f"   Kept {explain_t['kept_count']} elements")
    print(f"   Kept indices: {explain_t['kept_indices']}")
    print(f"   Original values: {[f'{x[i]:.2f}' for i in explain_t['kept_indices']]}")
    
    # Test with adaptive
    print("\n3. Adaptive threshold (alpha=0.4, i.e., keep |x| >= 0.4*max):")
    sp_a = Sparsifier(D, mode="adaptive_threshold", config={"alpha": 0.4})
    explain_a = sp_a.explain_row(x)
    max_val = np.max(np.abs(x))
    threshold = 0.4 * max_val
    print(f"   Max |value|: {max_val:.2f}")
    print(f"   Computed threshold: {threshold:.2f}")
    print(f"   Kept {explain_a['kept_count']} elements")
    print(f"   Kept indices: {explain_a['kept_indices']}")


def run_all_demos():
    """Run all demonstration functions."""
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  SPARSIFIER PACKAGE - COMPREHENSIVE DEMONSTRATIONS".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    demo_basic_modes()
    demo_static_masks()
    demo_calibration()
    demo_performance_comparison()
    demo_hybrid_mode()
    demo_explain()
    
    print("\n" + "="*70)
    print("All demonstrations completed!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Multiple sparsification modes available for different use cases")
    print("  • Static masks enable consistent sparsity patterns")
    print("  • Calibration improves reconstruction quality")
    print("  • Trade-off between sparsity level and accuracy")
    print("  • Hybrid mode combines static and dynamic strategies")
    print("  • Explainability features help understand decisions")
    print()


if __name__ == "__main__":
    run_all_demos()