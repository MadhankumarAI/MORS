"""
Sparse-MOR: Top-K Sparsity with Mixture-of-Recursions for Efficient LLM Inference
"""

from .sparsity import apply_sparsity
from .mor import apply_mor
from .spar_mor import sparse_mor_forward, SparMoRPipeline
from .trainer import train_with_spar_mor
from .evaluator import run_comparison

__version__ = "0.1.0"
__all__ = [
    "apply_sparsity",
    "apply_mor", 
    "sparse_mor_forward",
    "SparMoRPipeline",
    "train_with_spar_mor",
    "run_comparison"
]