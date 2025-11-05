"""
Unit tests for MoR module
"""
import torch
import pytest
from spar_mor.mor import MoRLayer

@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128).cuda()

def test_mor_forward(sample_input):
    mor = MoRLayer(
        hidden_dim=128,
        depths=[1, 2, 3],
        gate_hidden=64
    ).cuda()
    
    output = mor(sample_input)
    
    # Check output shape
    assert output.shape == sample_input.shape
    
    # Check no NaNs
    assert not torch.isnan(output).any()
    
    # Check reasonable magnitude
    mag_ratio = output.abs().mean() / sample_input.abs().mean()
    assert 0.1 <= mag_ratio <= 10.0

def test_mor_gates(sample_input):
    mor = MoRLayer(
        hidden_dim=128,
        depths=[1, 2],
        gate_hidden=64
    ).cuda()
    
    # Get gate values
    gate_logits = mor.gate_net(sample_input)
    gate_probs = torch.softmax(gate_logits, dim=-1)
    
    # Check gate shape
    assert gate_probs.shape == (2, 16, 2)
    
    # Check probabilities sum to 1
    assert torch.allclose(gate_probs.sum(-1), torch.ones_like(gate_probs.sum(-1)))
    
    # Check values in [0,1]
    assert (gate_probs >= 0).all() and (gate_probs <= 1).all()

def test_mor_recursion(sample_input):
    mor = MoRLayer(
        hidden_dim=128,
        depths=[1, 2],
        gate_hidden=64
    ).cuda()
    
    # Get outputs for depth 1 vs 2
    with torch.no_grad():
        d1_out = mor.branch_projs[0](sample_input)
        d2_out = mor.branch_projs[1](mor.branch_projs[1](sample_input))
    
    # Check different outputs
    assert not torch.allclose(d1_out, d2_out)
    
    # But similar magnitude
    d1_mag = d1_out.abs().mean()
    d2_mag = d2_out.abs().mean()
    assert 0.1 <= d1_mag / d2_mag <= 10.0

def test_mor_device_transfer():
    mor = MoRLayer(hidden_dim=128)
    assert next(mor.parameters()).device.type == "cpu"
    
    mor = mor.cuda()
    assert next(mor.parameters()).device.type == "cuda"
    
    mor = mor.cpu()
    assert next(mor.parameters()).device.type == "cpu"