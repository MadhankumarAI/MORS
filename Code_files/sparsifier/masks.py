"""
Mask management for static sparsity patterns.

Handles creation, loading, saving of binary masks for static sparsification.
Useful for fixed sparsity patterns determined offline.

Author:  Madhan
Date: 2025-10-12
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class MaskManager:
    """Manages static binary masks for sparsification."""
    
    def __init__(self, D: int, rng: np.random.RandomState):
        """Initialize mask manager.
        
        Args:
            D: dimension of the mask
            rng: random number generator for creating random masks
        """
        self.D = D
        self.rng = rng
        self.static_mask: Optional[np.ndarray] = None
    
    def set_static_mask(self, mask: np.ndarray) -> None:
        """Set a pre-computed static mask.
        
        Args:
            mask: (D,) binary mask array (0s and 1s)
        """
        mask = np.asarray(mask).astype(np.float32)
        assert mask.shape == (self.D,), f"Mask shape {mask.shape} != ({self.D},)"
        self.static_mask = mask
    
    def create_random_static_mask(self, keep_ratio: float) -> np.ndarray:
        """Create a random binary mask with specified keep ratio.
        
        Args:
            keep_ratio: fraction of elements to keep (0-1)
        Returns:
            Binary mask array of shape (D,)
        """
        keep_ratio = float(keep_ratio)
        assert 0.0 < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
        
        num_keep = max(1, int(round(self.D * keep_ratio)))
        mask = np.zeros(self.D, dtype=np.float32)
        idx = self.rng.choice(self.D, size=num_keep, replace=False)
        mask[idx] = 1.0
        
        self.static_mask = mask
        return mask
    
    def create_structured_mask(self, pattern: str = "alternate") -> np.ndarray:
        """Create structured masks (e.g., alternating, block patterns).
        
        Args:
            pattern: type of structure ("alternate", "first_half", "blocks")
        Returns:
            Binary mask array
        """
        mask = np.zeros(self.D, dtype=np.float32)
        
        if pattern == "alternate":
            mask[::2] = 1.0  # Keep every other element
        elif pattern == "first_half":
            mask[:self.D // 2] = 1.0
        elif pattern == "blocks":
            # Keep first quarter of each half
            block_size = self.D // 4
            mask[:block_size] = 1.0
            mask[self.D // 2:self.D // 2 + block_size] = 1.0
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        self.static_mask = mask
        return mask
    
    def save_mask(self, path: str) -> None:
        """Save current static mask to file.
        
        Args:
            path: file path (will save as .npy format)
        """
        if self.static_mask is None:
            raise ValueError("No static mask to save")
        np.save(path, self.static_mask)
        print(f"Mask saved to {path}")
    
    def load_mask(self, path: str) -> None:
        """Load static mask from file.
        
        Args:
            path: path to .npy file
        """
        m = np.load(path)
        if m.shape != (self.D,):
            raise ValueError(f"Loaded mask shape {m.shape} != ({self.D},)")
        self.static_mask = m.astype(np.float32)
        print(f"Mask loaded from {path}")
    
    def get_mask_stats(self) -> dict:
        """Get statistics about current mask."""
        if self.static_mask is None:
            return {"exists": False}
        
        kept = int(np.sum(self.static_mask))
        return {
            "exists": True,
            "kept_elements": kept,
            "total_elements": self.D,
            "keep_ratio": kept / self.D,
            "sparsity": 1.0 - (kept / self.D)
        }


# Example usage
if __name__ == "__main__":
    print("=== Mask Manager Example ===\n")
    
    D = 64
    rng = np.random.RandomState(42)
    manager = MaskManager(D, rng)
    
    # Create random mask
    print("1. Random mask with 50% keep ratio:")
    mask1 = manager.create_random_static_mask(keep_ratio=0.5)
    stats1 = manager.get_mask_stats()
    print(f"   Kept: {stats1['kept_elements']}/{stats1['total_elements']}")
    print(f"   Sparsity: {stats1['sparsity']:.1%}\n")
    
    # Create structured mask
    print("2. Alternating pattern mask:")
    mask2 = manager.create_structured_mask("alternate")
    print(f"   First 10 elements: {mask2[:10]}")
    print(f"   Stats: {manager.get_mask_stats()}\n")
    
    # Save and load
    print("3. Save/Load mask:")
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_mask.npy")
        manager.save_mask(path)
        
        # Create new manager and load
        manager2 = MaskManager(D, rng)
        manager2.load_mask(path)
        print(f"   Masks match: {np.allclose(mask2, manager2.static_mask)}\n")
    
    # Apply mask to data
    print("4. Apply mask to data:")
    X = rng.randn(3, D).astype(np.float32)
    X_masked = X * mask2[np.newaxis, :]
    print(f"   Original nonzeros: {np.count_nonzero(X)}")
    print(f"   Masked nonzeros: {np.count_nonzero(X_masked)}")
    print(f"   Expected: ~{D // 2 * 3} (3 rows × 32 kept)")