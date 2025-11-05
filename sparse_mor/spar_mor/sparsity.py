"""
Top-K sparsity implementation with CUDA acceleration
"""
import torch
import torch.nn.functional as F

def apply_sparsity(tensor: torch.Tensor, k: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    Apply top-k sparsification to input tensor on CUDA
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_dim)
        k (int): Number of values to keep per token (default: 200)
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (sparse_tensor, mask)
            - sparse_tensor has same shape as input with zeros for masked values
            - mask has same shape as input with 1.0 for kept values, 0.0 for masked
    """
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
        
    batch_size, seq_len, hidden_dim = tensor.shape
    
    # Get top k values and indices per sequence position
    values, indices = torch.topk(tensor.abs(), k=k, dim=-1)
    threshold = values[...,-1].unsqueeze(-1)
    
    # Create binary mask from threshold
    mask = (tensor.abs() >= threshold).float()
    
    # Apply mask to create sparse tensor
    sparse_tensor = tensor * mask
    
    return sparse_tensor, mask