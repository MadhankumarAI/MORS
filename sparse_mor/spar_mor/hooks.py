"""
Optional hooks for post-sparsification compensation
"""
import torch
import torch.nn as nn
from typing import Optional

class GammaRenormHook:
    """Hook for gamma renormalization after sparsification"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Setup forward hooks on all normalization layers"""
        for module in self.model.modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                handle = module.register_forward_hook(self._gamma_renorm_hook)
                self.handles.append(handle)
                
    def _gamma_renorm_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor
    ) -> torch.Tensor:
        """Apply gamma renormalization"""
        if not hasattr(module, "weight") or module.weight is None:
            return output
            
        gamma = module.weight.view(1, 1, -1)
        renorm_factor = torch.mean(torch.abs(output), dim=-1, keepdim=True)
        output = output * (gamma / (renorm_factor + 1e-6))
        
        return output
        
    def remove(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

def add_compensation_hooks(
    model: nn.Module,
    gamma_renorm: bool = True
) -> Optional[GammaRenormHook]:
    """
    Add compensation hooks to model
    
    Args:
        model: The model to add hooks to
        gamma_renorm: Whether to add gamma renormalization
        
    Returns:
        Hook object if hooks were added, else None
    """
    if gamma_renorm:
        return GammaRenormHook(model)
    return None