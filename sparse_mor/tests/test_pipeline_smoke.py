"""
Integration tests for Sparse-MOR pipeline
"""
import torch
import pytest
from transformers import BertModel, GPT2LMHeadModel
from spar_mor import SparMoRPipeline

@pytest.fixture
def sample_inputs():
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
        "attention_mask": torch.ones(batch_size, seq_len).cuda()
    }

def test_pipeline_bert(sample_inputs):
    model = BertModel.from_pretrained("bert-base-uncased").cuda()
    pipeline = SparMoRPipeline(model, sparsity_k=200)
    
    with torch.no_grad():
        outputs = pipeline.forward(**sample_inputs)
    
    assert outputs.last_hidden_state.shape == (2, 16, 768)
    assert not torch.isnan(outputs.last_hidden_state).any()

def test_pipeline_gpt2(sample_inputs):
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    pipeline = SparMoRPipeline(model, sparsity_k=200)
    
    with torch.no_grad():
        outputs = pipeline.forward(**sample_inputs)
    
    assert outputs.logits.shape == (2, 16, 50257)
    assert not torch.isnan(outputs.logits).any()

def test_sparsity_ratio(sample_inputs):
    model = BertModel.from_pretrained("bert-base-uncased").cuda()
    pipeline = SparMoRPipeline(model, sparsity_k=200)
    
    with torch.no_grad():
        # Get embeddings
        embeds = model.get_input_embeddings()(sample_inputs["input_ids"])
        
        # Apply sparsity
        sparse_embeds, mask = pipeline.apply_sparsity(embeds)
        
    sparsity = (mask == 0).float().mean()
    assert 0.70 <= sparsity <= 0.80  # ~74% sparse for k=200

def test_memory_reduction(sample_inputs):
    model = BertModel.from_pretrained("bert-base-uncased").cuda()
    pipeline = SparMoRPipeline(model, sparsity_k=200)
    
    # Baseline memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(**sample_inputs)
    baseline_mem = torch.cuda.max_memory_allocated()
    
    # Sparse memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = pipeline.forward(**sample_inputs)
    sparse_mem = torch.cuda.max_memory_allocated()
    
    reduction = (baseline_mem - sparse_mem) / baseline_mem
    assert reduction > 0.15  # At least 15% reduction