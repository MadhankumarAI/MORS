"""
Mixture-of-Recursions (MOR) implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MoRLayer(nn.Module):
    """
    Mixture-of-Recursions layer with softmax gating
    """
    def __init__(
        self,
        hidden_dim: int,
        depths: List[int] = [1,2,3],
        gate_hidden: int = 64
    ):
        super().__init__()
        
        self.depths = depths
        self.num_branches = len(depths)
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, self.num_branches)
        )
        
        # Recursive projection for each branch
        self.branch_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(self.num_branches)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MoR refinement
        
        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, hidden_dim)
            
        Returns:
            torch.Tensor: Refined output of same shape
        """
        # Calculate branch weights
        gate_logits = self.gate_net(x)
        branch_weights = F.softmax(gate_logits, dim=-1)
        
        outputs = []
        for i, depth in enumerate(self.depths):
            # Apply recursive projection
            branch_out = x
            for _ in range(depth):
                branch_out = self.branch_projs[i](branch_out)
            outputs.append(branch_out * branch_weights[...,i:i+1])
            
        # Combine weighted branches
        return torch.sum(torch.stack(outputs), dim=0)

def apply_mor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply MoR refinement to input tensor
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_dim)
    
    Returns:
        torch.Tensor: Refined tensor of same shape
    """
    mor = MoRLayer(
        hidden_dim=tensor.size(-1)
    ).to(tensor.device)
    
    return mor(tensor)