"""
Benchmark script comparing baseline vs sparsity+MOR
"""
import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoModel, AutoModelForCausalLM
from spar_mor import run_comparison

def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_prompts(prompts_file: str) -> list[str]:
    """Load benchmark prompts"""
    with open(prompts_file) as f:
        return [line.strip() for line in f]

def prepare_inputs(tokenizer, prompts: list[str], device: str) -> list[dict]:
    """Prepare model inputs from prompts"""
    return [
        tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        for prompt in prompts
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--config-with", required=True, help="Config with sparsity+MOR")
    parser.add_argument("--config-without", help="Optional baseline config")
    parser.add_argument("--prompts-file", required=True, help="File with test prompts")
    parser.add_argument("--save-dir", default="outputs", help="Directory to save results")
    args = parser.parse_args()
    
    # Load configs
    config = load_config(args.config_with)
    
    # Create output dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if "gpt" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoModel.from_pretrained(args.model)
    else:
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoModel.from_pretrained(args.model)
    model = model.to(args.device)
    
    # Load prompts and prepare inputs
    prompts = load_prompts(args.prompts_file)
    inputs = prepare_inputs(tokenizer, prompts, args.device)
    
    # Run comparison
    results = run_comparison(
        model=model,
        eval_inputs=inputs,
        config=config,
        output_dir=str(save_dir)
    )
    
    # Print summary
    print("\nResults Summary:")
    print(f"Speedup: {results['comparison']['speedup']:.2f}x")
    print(f"Memory reduction: {results['comparison']['memory_reduction']*100:.1f}%")
    print(f"Similarity: {results['sparse_mor']['avg_similarity']*100:.1f}%")
    print(f"\nFull results saved to {save_dir}/results.json")
    
if __name__ == "__main__":
    main()