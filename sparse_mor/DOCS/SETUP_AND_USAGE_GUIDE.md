# Sparse-MOR: Detailed Setup and Usage Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Project Structure](#project-structure)
4. [Running Examples](#running-examples)
5. [Benchmarking](#benchmarking)
6. [Training](#training)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## System Requirements

### Hardware Requirements
- CUDA-capable GPU (minimum 6GB VRAM recommended)
- CPU: 4+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 10GB free space for models and data

### Software Requirements
- Python 3.8 or higher
- CUDA Toolkit 11.7 or higher
- Git

## Installation Steps

1. **Create a Python Virtual Environment**
```powershell
# Create and activate virtual environment
python -m venv sparmor_env
.\sparmor_env\Scripts\activate  # On Windows
source sparmor_env/bin/activate  # On Linux/Mac
```

2. **Clone the Repository**
```powershell
git clone https://github.com/MadhankumarAI/MORS.git
cd MORS
```

3. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

4. **Verify Installation**
```powershell
# Run tests to verify setup
pytest tests/
```

## Project Structure

```
sparse_mor/
├── spar_mor/         # Core implementation
├── configs/          # Configuration files
├── examples/         # Example scripts
├── tests/           # Test suite
├── DOCS/            # Documentation
└── outputs/         # Benchmark results
```

## Running Examples

### 1. Basic GPT-2 Example
```powershell
# Run the GPT-2 example
python examples/example_gpt_run.py
```

### 2. BERT Example
```powershell
# Run the BERT example
python examples/example_bert_run.py
```

### Expected Outputs
- Model predictions
- Performance metrics
- Memory usage statistics

## Benchmarking

### 1. Prepare Test Data
```powershell
# Create a prompts file
echo "This is a test sentence." > prompts.txt
echo "Another test sentence." >> prompts.txt
```

### 2. Run Basic Benchmark
```powershell
python run_benchmark.py \
  --model distilgpt2 \
  --device cuda:0 \
  --config-with configs/spar_mor_default.yaml \
  --config-without baseline \
  --prompts-file prompts.txt \
  --save-dir outputs/default
```

### 3. Run High-Similarity Benchmark
```powershell
python run_benchmark.py \
  --model distilgpt2 \
  --device cuda:0 \
  --config-with configs/spar_mor_strict_similarity.yaml \
  --save-dir outputs/high_sim
```

### 4. Analyzing Results
- Check `outputs/results.json` for metrics
- View generated plots in `outputs/*.png`
- Compare latency, memory, and similarity scores

## Training

### 1. Prepare Training Data
- Use your own dataset or
- Use built-in datasets from Hugging Face

### 2. Basic Training
```powershell
python spar_mor.py train \
  --model gpt2 \
  --config configs/spar_mor_default.yaml \
  --device cuda:0
```

### 3. Custom Training
1. Create a custom config file:
```yaml
# custom_config.yaml
sparsity:
  k: 250
  gamma_renorm: true

mor:
  depths: [1, 2, 3]
  gate_hidden: 96

training:
  batch_size: 16
  learning_rate: 3e-5
```

2. Run training:
```powershell
python spar_mor.py train \
  --model gpt2 \
  --config custom_config.yaml \
  --device cuda:0
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use smaller model variant
   - Increase sparsity (lower k value)

2. **Slow Performance**
   - Enable torch.compile in config
   - Check GPU utilization
   - Optimize batch size

3. **Low Similarity Scores**
   - Increase k value
   - Use strict similarity config
   - Enable gamma renormalization

4. **Installation Issues**
   - Check CUDA compatibility
   - Update GPU drivers
   - Verify PyTorch installation

## Advanced Usage

### 1. Custom Model Integration

```python
from spar_mor import SparMoRPipeline

# Create pipeline
pipeline = SparMoRPipeline(
    model=your_model,
    sparsity_k=200,
    device="cuda"
)

# Forward pass
outputs = pipeline.forward(
    input_ids=input_ids,
    attention_mask=attention_mask
)
```

### 2. Configuration Tuning

Adjust these parameters for different tradeoffs:

1. Sparsity vs Quality:
```yaml
sparsity:
  k: 200-300  # Higher = better quality
  gamma_renorm: true/false
```

2. Speed vs Memory:
```yaml
inference:
  batch_size: 8-32
  compile: true
  use_flash_attention: true
```

3. MOR Complexity:
```yaml
mor:
  depths: [1,2,3]
  gate_hidden: 64-128
```

### 3. Monitoring and Logging

1. Enable detailed logging:
```python
from spar_mor.utils import setup_logging

logger = setup_logging(
    level="DEBUG",
    log_file="sparse_mor.log"
)
```

2. Monitor GPU usage:
```python
from spar_mor.utils import profile_memory, clear_gpu_cache

# Check memory usage
mem_stats = profile_memory("before_inference_")
print(mem_stats)

# Clear cache if needed
clear_gpu_cache()
```

### 4. Pipeline Customization

1. Add custom hooks:
```python
from spar_mor.hooks import add_compensation_hooks

# Add hooks
hooks = add_compensation_hooks(
    model,
    gamma_renorm=True
)

# Remove when done
hooks.remove()
```

2. Custom evaluation metrics:
```python
from spar_mor.evaluator import run_comparison

results = run_comparison(
    model=model,
    eval_inputs=inputs,
    config=config,
    output_dir="custom_eval"
)
```

## Performance Optimization Tips

1. **Memory Optimization**
   - Use gradient checkpointing for training
   - Clear cache between runs
   - Monitor memory usage

2. **Speed Optimization**
   - Enable torch.compile
   - Use optimal batch size
   - Enable flash attention

3. **Quality Optimization**
   - Fine-tune sparsity k value
   - Adjust MOR gate network size
   - Enable all compensation mechanisms

## Recommended Workflows

1. **Development Flow**
   ```
   Test → Benchmark → Tune → Deploy
   ```

2. **Training Flow**
   ```
   Prepare Data → Train → Evaluate → Fine-tune
   ```

3. **Deployment Flow**
   ```
   Benchmark → Optimize → Export → Monitor
   ```
To use the new visualizations, simply run the benchmark as before:

   python run_benchmark.py \
  --model distilgpt2 \
  --device cuda:0 \
  --config-with configs/spar_mor_default.yaml \
  --prompts-file prompts.txt \
  --save-dir outputs/default


  Then open outputs/default/dashboard.html in a web browser to access the interactive dashboard, or check the individual PNG plots in the same directory.

The visualizations make it much easier to:

Compare baseline vs Sparse+MOR performance
Analyze memory usage patterns
Understand sparsity distributions
Track throughput variations
Identify optimization opportunities
All visualizations are automatically generated and saved during the benchmark process.

Remember to always monitor system resources and model performance metrics when running experiments or training sessions.