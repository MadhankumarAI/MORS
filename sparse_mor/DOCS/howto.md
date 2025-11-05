# How to Use Sparse-MOR

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/sparse-mor.git
cd sparse-mor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_pipeline_smoke.py
pytest tests/test_mor_unit.py
```

## Benchmarking

1. Basic benchmark:
```bash
python run_benchmark.py \
  --model distilgpt2 \
  --device cuda:0 \
  --config-with configs/spar_mor_default.yaml \
  --config-without baseline \
  --prompts-file prompts.txt \
  --save-dir outputs/default
```

2. High similarity mode:
```bash
python run_benchmark.py \
  --model distilgpt2 \
  --device cuda:0 \
  --config-with configs/spar_mor_strict_similarity.yaml \
  --save-dir outputs/high_sim
```

## Training

Fine-tune with sparsity + MOR:
```bash
python spar_mor.py train \
  --model gpt2 \
  --config configs/spar_mor_default.yaml \
  --device cuda:0
```

## Examples

See the `examples/` directory for:
- `example_gpt_run.py`: Using with GPT-2
- `example_bert_run.py`: Using with BERT

## Configuration

Two config files are provided:

1. `configs/spar_mor_default.yaml`
   - k=200 sparsity
   - Standard settings for production

2. `configs/spar_mor_strict_similarity.yaml`
   - k=300 sparsity
   - Optimized for higher similarity (≥96%)

## Results Analysis

After benchmarking, check:
- `outputs/results.json`: Detailed metrics
- `outputs/*.png`: Visualization plots
  - Latency comparison
  - Memory usage
  - Similarity analysis
  - Throughput
  - Pareto frontier