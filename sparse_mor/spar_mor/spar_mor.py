"""
Main pipeline combining sparsity and MoR
"""
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .sparsity import apply_sparsity 
from .mor import apply_mor

class SparMoRPipeline:
    """
    Pipeline for applying sparsity + MOR to transformer models
    """
    def __init__(
        self,
        model: PreTrainedModel,
        sparsity_k: int = 200,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.sparsity_k = sparsity_k
        self.device = device
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with sparsity + MOR
        """
        # Get embeddings
        embeds = self.model.get_input_embeddings()(input_ids)
        
        # Apply sparsity
        sparse_embeds, mask = apply_sparsity(
            embeds,
            k=self.sparsity_k
        )
        
        # Apply MOR refinement
        refined_embeds = apply_mor(sparse_embeds)
        
        # Forward through model
        outputs = self.model(
            inputs_embeds=refined_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs

def sparse_mor_forward(
    inputs: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Helper function for pipeline forward pass
    """
    pipeline = SparMoRPipeline(
        model=model,
        sparsity_k=config.get("sparsity_k", 200),
        device=inputs["input_ids"].device
    )
    
    return pipeline.forward(**inputs)