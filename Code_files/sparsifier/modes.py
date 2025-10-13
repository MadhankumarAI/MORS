"""
Sparsity mode implementations.

Implements various sparsification strategies:
- static: use fixed binary mask
- topk: keep top-k largest elements per row
- threshold: keep elements above absolute threshold
- adaptive_threshold: keep elements above relative threshold
- block: block/channel sparsity
- global_budget: global top-k across entire batch
- hybrid: combine static mask + dynamic top-k

Author: Madhan
Date: 2025-10-12
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from .utils import ensure_2d


class SparsityModes:
    """Collection of sparsification strategies."""
    
    def __init__(self, D: int, config: Dict[str, Any]):
        """Initialize with dimension and config dict."""
        self.D = D
        self.config = config
    
    def apply_mode(self, X: np.ndarray, mode: str, 
                   static_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply specified sparsity mode to X.
        
        Args:
            X: (N, D) input array
            mode: sparsity mode name
            static_mask: optional static mask for 'static' and 'hybrid' modes
        Returns:
            Sparsified array (N, D)
        """
        X = ensure_2d(X)
        
        if mode == "static":
            return self._apply_static(X, static_mask)
        elif mode == "topk":
            k = int(self.config.get("k", self.D // 4))
            return self._batch_topk(X, k)
        elif mode == "threshold":
            tau = float(self.config.get("threshold", 0.0))
            return self._batch_threshold(X, tau)
        elif mode == "adaptive_threshold":
            alpha = float(self.config.get("alpha", 0.2))
            return self._batch_adaptive_threshold(X, alpha)
        elif mode == "block":
            block_size = int(self.config.get("block_size", 16))
            blocks_keep = self.config.get("blocks_keep")
            return self._block_sparsify(X, block_size, blocks_keep)
        elif mode == "global_budget":
            T = int(self.config.get("global_budget"))
            return self._global_budget_sparsify(X, T)
        elif mode == "hybrid":
            return self._hybrid_sparsify(X, static_mask)
        else:
            raise ValueError(f"Unknown sparsity mode: {mode}")
    
    # ========== Static Mode ==========
    def _apply_static(self, X: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Apply fixed binary mask to all rows."""
        if mask is None:
            raise ValueError("Static mask not set")
        return X * mask[np.newaxis, :]
    
    # ========== Top-K Mode ==========
    def _topk_row(self, v: np.ndarray, k: int) -> np.ndarray:
        """Keep top-k elements by absolute value in 1D array."""
        if k >= v.size:
            return v.copy()
        absvals = np.abs(v)
        thresh = np.partition(absvals, -k)[-k]
        mask = absvals >= thresh
        return v * mask
    
    def _batch_topk(self, X: np.ndarray, k: int) -> np.ndarray:
        """Apply top-k per row across batch."""
        N = X.shape[0]
        out = np.zeros_like(X)
        for i in range(N):
            out[i] = self._topk_row(X[i], k)
        return out
    
    # ========== Threshold Modes ==========
    def _batch_threshold(self, X: np.ndarray, tau: float) -> np.ndarray:
        """Keep elements where |x| >= tau."""
        return np.where(np.abs(X) >= tau, X, 0.0)
    
    def _threshold_row(self, v: np.ndarray, tau: float) -> np.ndarray:
        """Threshold single row."""
        return v * (np.abs(v) >= tau)
    
    def _adaptive_threshold_row(self, v: np.ndarray, alpha: float) -> np.ndarray:
        """Keep elements >= alpha * max(|v|)."""
        m = np.max(np.abs(v))
        tau = alpha * m
        return self._threshold_row(v, tau)
    
    def _batch_adaptive_threshold(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """Apply adaptive threshold per row."""
        out = np.zeros_like(X)
        for i in range(X.shape[0]):
            out[i] = self._adaptive_threshold_row(X[i], alpha)
        return out
    
    # ========== Block Sparsity ==========
    def _block_sparsify(self, X: np.ndarray, block_size: int, 
                       blocks_keep: Optional[int]) -> np.ndarray:
        """Keep top blocks (groups of consecutive elements) per row."""
        N, D = X.shape
        
        # Pad if necessary
        if D % block_size != 0:
            pad = block_size - (D % block_size)
            X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
            Dp = X.shape[1]
        else:
            pad = 0
            Dp = D
        
        B = Dp // block_size
        
        # Determine blocks to keep
        if blocks_keep is None:
            kr = self.config.get("keep_ratio", 0.5)
            blocks_keep = max(1, int(round(B * kr)))
        
        # Reshape into blocks
        blocks = X.reshape(N, B, block_size)
        norms = np.linalg.norm(blocks, axis=2)  # (N, B)
        
        # Keep top blocks by norm
        out = np.zeros_like(blocks)
        for i in range(N):
            idx = np.argpartition(norms[i], -blocks_keep)[-blocks_keep:]
            out[i, idx, :] = blocks[i, idx, :]
        
        out = out.reshape(N, Dp)
        
        # Remove padding
        if pad:
            out = out[:, :D]
        
        return out
    
    # ========== Global Budget ==========
    def _global_budget_sparsify(self, X: np.ndarray, T: int) -> np.ndarray:
        """Keep top-T activations across entire batch."""
        N, D = X.shape
        if T >= N * D:
            return X.copy()
        
        flat = X.flatten()
        abs_flat = np.abs(flat)
        thresh = np.partition(abs_flat, -T)[-T]
        mask_flat = abs_flat >= thresh
        out_flat = flat * mask_flat
        
        return out_flat.reshape(N, D)
    
    # ========== Hybrid Mode ==========
    def _hybrid_sparsify(self, X: np.ndarray, 
                        static_mask: Optional[np.ndarray]) -> np.ndarray:
        """Apply static mask then top-k within surviving dimensions."""
        if static_mask is None:
            raise ValueError("Hybrid mode requires static mask")
        
        # Apply static mask
        X_static = X * static_mask[np.newaxis, :]
        
        # Get surviving columns
        cols = np.where(static_mask == 1.0)[0]
        if cols.size == 0:
            return np.zeros_like(X)
        
        # Apply top-k on surviving columns
        k = int(self.config.get("k", cols.size // 2))
        sub = X_static[:, cols]
        sub_sparse = np.zeros_like(sub)
        
        for i in range(sub.shape[0]):
            sub_sparse[i] = self._topk_row(sub[i], min(k, sub.shape[1]))
        
        out = np.zeros_like(X)
        out[:, cols] = sub_sparse
        return out


# Example usage
if __name__ == "__main__":
    print("=== Sparsity Modes Example ===\n")
    
    D = 32
    N = 3
    rng = np.random.RandomState(42)
    X = rng.randn(N, D).astype(np.float32)
    
    config = {"k": 8, "threshold": 0.5, "alpha": 0.3, 
              "block_size": 8, "keep_ratio": 0.5}
    modes = SparsityModes(D, config)
    
    print(f"Input shape: {X.shape}")
    print(f"Sample row [0, :8]: {X[0, :8]}\n")
    
    # Test different modes
    print("1. Top-K (k=8):")
    Xk = modes.apply_mode(X, "topk")
    print(f"   Nonzeros: {np.count_nonzero(Xk)} (expected: ~24 = 3 rows × 8)")
    print(f"   Row 0 [0:8]: {Xk[0, :8]}\n")
    
    print("2. Absolute Threshold (tau=0.5):")
    Xt = modes.apply_mode(X, "threshold")
    print(f"   Nonzeros: {np.count_nonzero(Xt)}")
    print(f"   Sparsity: {1 - np.count_nonzero(Xt) / Xt.size:.1%}\n")
    
    print("3. Adaptive Threshold (alpha=0.3):")
    Xa = modes.apply_mode(X, "adaptive_threshold")
    print(f"   Nonzeros: {np.count_nonzero(Xa)}")
    for i in range(N):
        max_val = np.max(np.abs(X[i]))
        kept = np.count_nonzero(Xa[i])
        print(f"   Row {i}: max={max_val:.2f}, kept={kept} elements\n")
    
    print("4. Block Sparsity (block_size=8, keep_ratio=0.5):")
    Xb = modes.apply_mode(X, "block")
    print(f"   Nonzeros: {np.count_nonzero(Xb)}")
    print(f"   Expected: ~{N * (D // 2)} (half the elements)\n")
    
    print("5. Global Budget (T=30):")
    config["global_budget"] = 30
    Xg = modes.apply_mode(X, "global_budget")
    print(f"   Nonzeros: {np.count_nonzero(Xg)} (should be ≤30)")