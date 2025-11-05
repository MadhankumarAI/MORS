"""
CLI wrapper for running and training with sparsity+MOR
"""
import argparse
import yaml
import torch
from pathlib import Path
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from datasets import load_dataset
from spar_mor import train_with_spar_mor

def train(args):
    """Fine-tune model with sparsity+MOR"""
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    # Load model
    if "gpt" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model)
    else:
        model = AutoModel.from_pretrained(args.model)
    model = model.to(args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load dataset (example using wikitext)
    train_data = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )
    eval_data = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1", 
        split="validation"
    )
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["training"]["sequence_length"],
            return_tensors="pt"
        )
    
    train_dataset = train_data.map(
        tokenize,
        batched=True,
        remove_columns=train_data.column_names
    )
    eval_dataset = eval_data.map(
        tokenize,
        batched=True,
        remove_columns=eval_data.column_names
    )
    
    # Train
    trained_model = train_with_spar_mor(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )
    
    # Save
    output_dir = Path("trained_model")
    output_dir.mkdir(exist_ok=True)
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    # Train command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True, help="Model name/path")
    train_parser.add_argument("--config", required=True, help="Config file")
    train_parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()