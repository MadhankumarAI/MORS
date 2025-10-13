"""
Core Sparsifier class - Main orchestrator for sparsity operations.

This module provides the main Sparsifier class that coordinates different
sparsity modes, normalization, and calibration strategies.

Author: ChatGPT (for Madhan)
Date: 2025-10-12
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any

from .masks import MaskManager
from .modes import SparsityModes
from .compensation import CompensationManager
from .utils import ensure_2d


class Sparsifier:
    """Main sparsifier class for applying inference-time sparsity."""
    
    def __init__(self, D: int, mode: str = "topk", 
                 config: Optional[Dict[str, Any]] = None, 
                 rng_seed: Optional[int] = None):
        """Initialize Sparsifier.

        Args:
            D: embedding dimension (number of features)
            mode: default sparsity mode
            config: dict of mode-specific parameters
            rng_seed: optional seed for deterministic masks
        """
        self.D = int(D)
        self.mode = mode
        self.config = {} if config is None else dict(config)
        self.rng = np.random.RandomState(rng_seed)

        # Initialize sub-managers
        self.mask_manager = MaskManager(D, self.rng)
        self.sparsity_modes = SparsityModes(D, self.config)
        self.compensation = CompensationManager(D, self.rng, self.config)

        # Set defaults
        self._set_default_config()

    def _set_default_config(self):
        """Set default configuration parameters."""
        c = self.config
        c.setdefault("keep_ratio", 0.5)
        c.setdefault("k", max(1, int(self.D * 0.25)))
        c.setdefault("threshold", 0.0)
        c.setdefault("alpha", 0.2)
        c.setdefault("block_size", 16)
        c.setdefault("blocks_keep", None)
        c.setdefault("global_budget", None)
        c.setdefault("hybrid_static_keep_ratio", 0.5)
        c.setdefault("renorm", True)
        c.setdefault("epsilon", 1e-12)
        c.setdefault("reconstructor_hidden", 64)
        c.setdefault("reconstructor_train_steps", 200)
        c.setdefault("reconstructor_lr", 1e-3)

    def set_mode(self, mode: str, **params) -> None:
        """Change sparsity mode and update parameters."""
        self.mode = mode
        for k, v in params.items():
            self.config[k] = v
        self._set_default_config()

    def _apply_mode(self, X: np.ndarray) -> np.ndarray:
        """Apply configured sparsity mode to X."""
        X = ensure_2d(X)
        return self.sparsity_modes.apply_mode(
            X, self.mode, self.mask_manager.static_mask
        )

    def apply_to_batch(self, X: np.ndarray, renorm: Optional[bool] = None, 
                      apply_gamma: bool = True, reconstruct: bool = False) -> np.ndarray:
        """Main entry: apply sparsity with optional compensation.

        Args:
            X: (N,D) original dense encoded vectors
            renorm: override config['renorm'] if provided
            apply_gamma: whether to multiply by gamma scalar
            reconstruct: whether to use reconstructor MLP
        Returns:
            Sparse (N,D) array
        """
        X = ensure_2d(X).astype(np.float32)
        renorm = self.config.get("renorm") if renorm is None else bool(renorm)
        
        # Apply sparsity
        Xs = self._apply_mode(X)
        
        # Apply compensation strategies
        if renorm:
            Xs = self.compensation.l2_renorm(X, Xs, self.config.get("epsilon", 1e-12))
        if apply_gamma:
            Xs = self.compensation.apply_gamma(Xs)
        if reconstruct:
            Xs = self.compensation.apply_reconstructor(Xs)
            
        return Xs

    def calibrate(self, X_val: np.ndarray, calibrate_gamma: bool = True, 
                 calibrate_reconstructor: bool = False) -> Dict[str, Any]:
        """Calibrate sparsifier using held-out data."""
        X_val = ensure_2d(X_val).astype(np.float32)
        result = {}
        
        if calibrate_gamma:
            Xs = self._apply_mode(X_val)
            best_g = self.compensation.calibrate_gamma(X_val, Xs)
            result["gamma"] = best_g
            
        if calibrate_reconstructor:
            self.compensation.init_reconstructor()
            steps = int(self.config.get("reconstructor_train_steps", 200))
            lr = float(self.config.get("reconstructor_lr", 1e-3))
            self.compensation.train_reconstructor(
                self._apply_mode(X_val), X_val, steps, lr
            )
            result["reconstructor_trained"] = True
            
        return result

    def stats(self, X: np.ndarray) -> Dict[str, Any]:
        """Compute sparsity statistics."""
        X = ensure_2d(X)
        Xs = self._apply_mode(X)
        nonzeros = np.count_nonzero(Xs)
        total = Xs.size
        return {
            "nonzeros": int(nonzeros),
            "total": int(total),
            "sparsity_fraction": 1.0 - float(nonzeros) / float(total)
        }

    def explain_row(self, X_row: np.ndarray) -> Dict[str, Any]:
        """Explain sparsification for a single row."""
        v = np.asarray(X_row).reshape(-1)
        sparse = self._apply_mode(v.reshape(1, -1)).reshape(-1)
        kept_idx = np.where(np.abs(sparse) > 0)[0].tolist()
        return {
            "kept_indices": kept_idx,
            "kept_count": len(kept_idx),
            "kept_norm": float(np.linalg.norm(sparse))
        }

    # Delegate mask operations
    def set_static_mask(self, mask: np.ndarray) -> None:
        self.mask_manager.set_static_mask(mask)
    
    def create_random_static_mask(self, keep_ratio: Optional[float] = None) -> np.ndarray:
        kr = self.config.get("keep_ratio") if keep_ratio is None else float(keep_ratio)
        return self.mask_manager.create_random_static_mask(kr)
    
    def save_mask(self, path: str) -> None:
        self.mask_manager.save_mask(path)
    
    def load_mask(self, path: str) -> None:
        self.mask_manager.load_mask(path)


# Example usage
if __name__ == "__main__":
    print("=== Core Sparsifier Example ===\n")
    
    # Create sample data
    D = 128
    N = 4
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    
    # Initialize sparsifier
    sp = Sparsifier(D, mode="topk", config={"k": 32}, rng_seed=0)
    
    # Apply sparsification
    Xs = sp.apply_to_batch(X)
    print(f"Original shape: {X.shape}")
    print(f"Sparsified nonzeros: {np.count_nonzero(Xs)}")
    print(f"Sparsity: {sp.stats(X)['sparsity_fraction']:.2%}\n")
    
    # Change mode
    sp.set_mode("threshold", threshold=0.5)
    Xs2 = sp.apply_to_batch(X)
    print(f"Threshold mode nonzeros: {np.count_nonzero(Xs2)}")
    
    # Explain single row
    explain = sp.explain_row(X[0])
    print(f"\nRow 0 kept {explain['kept_count']} elements")
    print(f"Kept indices (first 10): {explain['kept_indices'][:10]}")