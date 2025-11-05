"""
Example using Sparse-MOR with BERT
"""
import torch
from transformers import BertModel, BertTokenizer
from spar_mor import SparMoRPipeline

def main():
    # Load model and tokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create pipeline
    pipeline = SparMoRPipeline(
        model=model,
        sparsity_k=200,
        device="cuda"
    )
    
    # Example text
    text = ["This is an example sentence.", "Another example here."]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    # Get embeddings with sparsity + MOR
    with torch.no_grad():
        outputs = pipeline.forward(**inputs)
    
    # Print pooled output shape
    print(f"Pooled output shape: {outputs.pooler_output.shape}")
    
    # Print mean activation
    print(f"Mean activation: {outputs.last_hidden_state.abs().mean().item():.3f}")

if __name__ == "__main__":
    main()