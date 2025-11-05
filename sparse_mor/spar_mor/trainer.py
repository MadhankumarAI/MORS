"""
Training utilities for fine-tuning models with sparsity + MOR
"""
import torch
from transformers import PreTrainedModel, Trainer, TrainingArguments
from typing import Dict, Any, Optional

from .spar_mor import SparMoRPipeline

def train_with_spar_mor(
    model: PreTrainedModel,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: Optional[torch.utils.data.Dataset] = None,
    config: Dict[str, Any] = None,
    **train_kwargs
) -> PreTrainedModel:
    """
    Fine-tune a model with sparsity + MOR pipeline
    
    Args:
        model: The model to fine-tune
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        config: Configuration dict with sparsity_k etc.
        **train_kwargs: Additional kwargs for Trainer
        
    Returns:
        The fine-tuned model
    """
    # Create pipeline
    pipeline = SparMoRPipeline(
        model=model,
        sparsity_k=config.get("sparsity_k", 200)
    )
    
    # Setup trainer
    training_args = TrainingArguments(
        output_dir="./sparse_mor_checkpoints",
        evaluation_strategy="epoch" if eval_dataset else "no",
        learning_rate=config.get("learning_rate", 5e-5),
        num_train_epochs=config.get("num_epochs", 3),
        **train_kwargs
    )
    
    trainer = Trainer(
        model=pipeline,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train and return model
    trainer.train()
    return model