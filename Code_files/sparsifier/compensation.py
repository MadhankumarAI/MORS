"""
Compensation strategies for sparsified representations.

Provides methods to mitigate information loss from sparsification:
- L2 renormalization: preserve row-wise L2 norms
- Learned gamma scaling: calibrated global/per-dim multipliers
- Reconstructor MLP: learn to predict residual (original - sparse)

Author: Madhan
Date: 2025-10-12
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from .utils import ensure_2d


class CompensationManager:
    """Manages compensation strategies for sparsified data."""
    
    def __init__(self, D: int, rng: np.random.RandomState, config: Dict[str, Any]):
        """Initialize compensation manager.
        
        Args:
            D: embedding dimension
            rng: random state for weight initialization
            config: configuration dict
        """
        self.D = D
        self.rng = rng
        self.config = config
        
        # Learned parameters
        self.gamma: Optional[float] = None
        self.reconstructor: Optional[Dict[str, np.ndarray]] = None
    
    # ========== L2 Renormalization ==========
    def l2_renorm(self, X_orig: np.ndarray, X_sparse: np.ndarray, 
                  eps: float = 1e-12) -> np.ndarray:
        """Scale sparse vectors to match original L2 norms.
        
        Args:
            X_orig: original dense vectors (N, D)
            X_sparse: sparsified vectors (N, D)
            eps: small constant for numerical stability
        Returns:
            Renormalized sparse vectors (N, D)
        """
        X_orig = ensure_2d(X_orig)
        X_sparse = ensure_2d(X_sparse)
        
        orig_norm = np.linalg.norm(X_orig, axis=1) + eps
        sparse_norm = np.linalg.norm(X_sparse, axis=1) + eps
        scale = (orig_norm / sparse_norm).reshape(-1, 1)
        
        return X_sparse * scale
    
    # ========== Gamma Scaling ==========
    def apply_gamma(self, X: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """Apply learned scalar gamma to compensate for sparsity.
        
        Args:
            X: input array
            gamma: override stored gamma if provided
        Returns:
            Scaled array
        """
        g = self.gamma if gamma is None else gamma
        if g is None:
            return X
        return X * float(g)
    
    def calibrate_gamma(self, X_orig: np.ndarray, X_sparse: np.ndarray) -> float:
        """Find optimal gamma via grid search to minimize MSE.
        
        Args:
            X_orig: original vectors (N, D)
            X_sparse: sparsified vectors (N, D)
        Returns:
            Best gamma value
        """
        X_orig = ensure_2d(X_orig)
        X_sparse = ensure_2d(X_sparse)
        
        # Search grid
        candidates = np.concatenate((
            [0.01, 0.1, 0.2, 0.5],
            np.linspace(0.8, 1.2, 21),
            [1.5, 2.0]
        ))
        
        best_g = 1.0
        best_loss = float("inf")
        
        for g in candidates:
            pred = X_sparse * g
            loss = np.mean((X_orig - pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_g = float(g)
        
        self.gamma = best_g
        return best_g
    
    # ========== Reconstructor MLP ===
    def init_reconstructor(self, hidden: Optional[int] = None) -> None:
        """Initialize small MLP to predict residual.
        
        Architecture: Linear -> ReLU -> Linear
        Learns: reconstructor(X_sparse) ≈ (X_orig - X_sparse)
        
        Args:
            hidden: hidden layer size (default from config)
        """
        h = self.config.get("reconstructor_hidden", 64) if hidden is None else int(hidden)
        
        # Xavier-like initialization
        W1 = self.rng.randn(h, self.D) * (1.0 / np.sqrt(self.D))
        b1 = np.zeros(h, dtype=np.float32)
        W2 = self.rng.randn(self.D, h) * (1.0 / np.sqrt(h))
        b2 = np.zeros(self.D, dtype=np.float32)
        
        self.reconstructor = {
            "W1": W1.astype(np.float32),
            "b1": b1,
            "W2": W2.astype(np.float32),
            "b2": b2
        }
    
    def reconstructor_forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through reconstructor MLP.
        
        Args:
            X: input vectors (N, D)
        Returns:
            Predicted residual (N, D)
        """
        if self.reconstructor is None:
            raise ValueError("Reconstructor not initialized")
        
        W1 = self.reconstructor["W1"]
        b1 = self.reconstructor["b1"]
        W2 = self.reconstructor["W2"]
        b2 = self.reconstructor["b2"]
        
        # Forward: X @ W1.T + b1 -> ReLU -> @ W2.T + b2
        hidden = np.maximum(0.0, X.dot(W1.T) + b1)
        out = hidden.dot(W2.T) + b2
        return out
    
    def apply_reconstructor(self, X_sparse: np.ndarray) -> np.ndarray:
        """Apply reconstructor and add correction.
        
        Args:
            X_sparse: sparsified vectors
        Returns:
            X_sparse + correction
        """
        if self.reconstructor is None:
            return X_sparse
        correction = self.reconstructor_forward(X_sparse)
        return X_sparse + correction
    
    def train_reconstructor(self, X_sparse: np.ndarray, X_orig: np.ndarray,
                          steps: int = 200, lr: float = 1e-3) -> None:
        """Train reconstructor via SGD to predict residual.
        
        Loss: || reconstructor(X_sparse) - (X_orig - X_sparse) ||^2
        
        Args:
            X_sparse: sparsified vectors (N, D)
            X_orig: original dense vectors (N, D)
            steps: training iterations
            lr: learning rate
        """
        Xs = ensure_2d(X_sparse).astype(np.float32)
        Xo = ensure_2d(X_orig).astype(np.float32)
        N = Xs.shape[0]
        
        if self.reconstructor is None:
            self.init_reconstructor()
        
        W1 = self.reconstructor["W1"]
        b1 = self.reconstructor["b1"]
        W2 = self.reconstructor["W2"]
        b2 = self.reconstructor["b2"]
        
        # Training loop (full-batch SGD)
        for step in range(steps):
            # Forward pass
            H = np.maximum(0.0, Xs.dot(W1.T) + b1)
            pred = H.dot(W2.T) + b2
            
            # Target is residual
            residual = Xo - Xs
            error = pred - residual  # (N, D)
            loss = 0.5 * np.mean(error ** 2)
            
            # Log progress
            if step % max(1, steps // 5) == 0:
                print(f"  Reconstructor step {step}/{steps} loss={loss:.6e}")
            
            # Backpropagation
            grad_pred = error / N  # dL/dpred
            
            # Output layer gradients
            grad_W2 = grad_pred.T.dot(H)  # (D, hidden)
            grad_b2 = np.sum(grad_pred, axis=0)
            
            # Hidden layer gradients
            grad_H = grad_pred.dot(W2)  # (N, hidden)
            grad_preH = grad_H * (H > 0).astype(np.float32)  # ReLU derivative
            grad_W1 = grad_preH.T.dot(Xs)  # (hidden, D)
            grad_b1 = np.sum(grad_preH, axis=0)
            
            # SGD updates
            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1
        
        # Store updated weights
        self.reconstructor["W1"] = W1
        self.reconstructor["b1"] = b1
        self.reconstructor["W2"] = W2
        self.reconstructor["b2"] = b2


if __name__ == "__main__":
    print("=== Compensation Manager Example ===\n")
    
    D = 64
    N = 8
    rng = np.random.RandomState(42)
    config = {"reconstructor_hidden": 32}
    
    # Create sample data
    X_orig = rng.randn(N, D).astype(np.float32)
    
    # Simulate sparsification (keep top 25%)
    k = D // 4
    X_sparse = np.zeros_like(X_orig)
    for i in range(N):
        idx = np.argpartition(np.abs(X_orig[i]), -k)[-k:]
        X_sparse[i, idx] = X_orig[i, idx]
    
    comp = CompensationManager(D, rng, config)
    
    # 1. L2 Renormalization
    print("1. L2 Renormalization:")
    orig_norms = np.linalg.norm(X_orig, axis=1)
    sparse_norms = np.linalg.norm(X_sparse, axis=1)
    X_renorm = comp.l2_renorm(X_orig, X_sparse)
    renorm_norms = np.linalg.norm(X_renorm, axis=1)
    
    print(f"   Original norms [0:3]: {orig_norms[:3]}")
    print(f"   Sparse norms [0:3]:   {sparse_norms[:3]}")
    print(f"   Renorm norms [0:3]:   {renorm_norms[:3]}")
    print(f"   Norms match: {np.allclose(orig_norms, renorm_norms)}\n")
    
    # 2. Gamma Calibration
    print("2. Gamma Calibration:")
    best_gamma = comp.calibrate_gamma(X_orig, X_sparse)
    X_scaled = comp.apply_gamma(X_sparse)
    
    mse_before = np.mean((X_orig - X_sparse) ** 2)
    mse_after = np.mean((X_orig - X_scaled) ** 2)
    print(f"   Best gamma: {best_gamma:.4f}")
    print(f"   MSE before: {mse_before:.6f}")
    print(f"   MSE after:  {mse_after:.6f}")
    print(f"   Improvement: {(1 - mse_after/mse_before)*100:.1f}%\n")
    
    # 3. Reconstructor Training
    print("3. Reconstructor MLP:")
    comp.init_reconstructor(hidden=32)
    print("Training reconstructor...")
    comp.train_reconstructor(X_sparse, X_orig, steps=100, lr=1e-3)
    
    X_reconstructed = comp.apply_reconstructor(X_sparse)
    mse_reconstructed = np.mean((X_orig - X_reconstructed) ** 2)
    print(f"\n   MSE with reconstructor: {mse_reconstructed:.6f}")
    print(f"   Improvement over sparse: {(1 - mse_reconstructed/mse_before)*100:.1f}%")
    
    # 4. Combined approach
    print("\n4. Combined Compensation:")
    X_combined = comp.l2_renorm(X_orig, X_sparse)
    X_combined = comp.apply_gamma(X_combined)
    X_combined = comp.apply_reconstructor(X_combined)
    mse_combined = np.mean((X_orig - X_combined) ** 2)
    print(f"   MSE with all compensation: {mse_combined:.6f}")