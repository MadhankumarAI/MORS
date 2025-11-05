# API Reference

## Core Functions

### apply_sparsity

```python
def apply_sparsity(tensor: torch.Tensor, k: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply top-k sparsification to input tensor
    
    Args:
        tensor: Input of shape (batch, seq_len, hidden_dim)
        k: Number of values to keep per token (default: 200)
        
    Returns:
        (sparse_tensor, mask):
            sparse_tensor: Same shape as input with zeros for masked values
            mask: Binary mask showing kept values (1.0) vs masked (0.0)
    """
```

### apply_mor

```python
def apply_mor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply MoR refinement
    
    Args:
        tensor: Input of shape (batch, seq_len, hidden_dim)
        
    Returns:
        Refined tensor of same shape
    """
```

### sparse_mor_forward

```python
def sparse_mor_forward(
    inputs: Dict[str, torch.Tensor],
    model: PreTrainedModel, 
    config: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Full pipeline forward pass
    
    Args:
        inputs: Model input tensors
        model: Hugging Face model
        config: Configuration dict
        
    Returns:
        Model outputs with sparsity + MOR applied
    """
```

## Classes

### SparMoRPipeline

```python
class SparMoRPipeline:
    """
    Pipeline for applying sparsity + MOR to transformer models
    
    Args:
        model: The model to wrap
        sparsity_k: Top-k value (default: 200)
        device: Device to run on (default: "cuda")
    """
```

### MoRLayer

```python
class MoRLayer(nn.Module):
    """
    Mixture-of-Recursions layer
    
    Args:
        hidden_dim: Size of hidden dimension
        depths: List of recursion depths (default: [1,2,3])
        gate_hidden: Hidden dim for gating network (default: 64)
    """
```

## Training

```python
def train_with_spar_mor(
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Dict[str, Any] = None,
    **train_kwargs
) -> PreTrainedModel
```

## Evaluation

```python
def run_comparison(
    model: PreTrainedModel,
    eval_inputs: List[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    output_dir: str = "outputs"
) -> Dict[str, Any]
```