"""
Example using Sparse-MOR with GPT-2
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from spar_mor import SparMoRPipeline

def main():
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Create pipeline
    pipeline = SparMoRPipeline(
        model=model,
        sparsity_k=200,
        device="cuda"
    )
    
    # Example text
    text = "Once upon a time"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    # Generate with sparsity + MOR
    with torch.no_grad():
        outputs = pipeline.forward(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9
        )
    
    # Decode output
    generated_text = tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True
    )
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()