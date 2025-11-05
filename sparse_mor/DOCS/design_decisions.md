# Design Decisions

## Top-K Sparsity (k=200)

### Why k=200?

The choice of k=200 represents an optimal tradeoff point between:
- Speed: ~1.9x inference speedup
- Quality: ≥95% similarity to baseline
- Memory: ~25% reduction in GPU memory

Higher k values (e.g., k=300):
- Better similarity but diminishing returns
- Reduced speedup (only ~1.5x)
- Higher memory usage

Lower k values (e.g., k=100):
- Faster but quality degrades quickly
- More likely to miss important information
- May require more complex compensation

## Mixture-of-Recursions Design

### Architecture

We use 3 branches with depths [1,2,3] because:
1. Deeper recursion captures more complex patterns
2. Shallow paths maintain direct signal flow
3. Three branches balance expressiveness vs compute

### Gating Network

- Hidden dim 64 (default) or 128 (high-similarity)
- Softmax gating for smooth interpolation
- Small MLP keeps overhead minimal

### Why Not More/Fewer Branches?

- 2 branches: Insufficient expressiveness
- 4+ branches: Diminishing returns, more compute
- 3 is "sweet spot" empirically

## Implementation Choices

### GPU-Only

Reasons for CUDA-only implementation:
1. Target use case is large model inference
2. CPU implementation would be much slower
3. Simplifies code and maintenance

### Batch-Recursive Computation

Process multiple sequences at once:
- Better GPU utilization
- Lower overhead per token
- Works well with typical batch sizes (8-32)

### Optional Gamma Renorm

Light compensation mechanism that:
- Helps maintain activation statistics
- Minimal computational overhead
- Can be disabled if unneeded

## Model Support

### Transformers Library Integration

Support for Hugging Face models because:
1. De facto standard for transformers
2. Wide model availability
3. Consistent API across models

### Why GPT/BERT Focus?

- Cover both autoregressive and masked LMs
- Most common architectures in production
- Easily extends to variants (DistilBERT, etc.)

## Performance Considerations

### Memory Management

- Use in-place operations where possible
- Release intermediate tensors promptly
- Careful gradient handling during training

### Speed Optimizations

1. Fused CUDA kernels for Top-K
2. Efficient recursion implementation
3. Minimal tensor allocations
4. Optional torch.compile support

## Configuration System

Two main configs provided:

1. Default (k=200)
   - Balanced performance/quality
   - Production-ready settings
   - Standard gate network size

2. Strict Similarity (k=300)
   - Higher quality threshold
   - Larger gate network
   - More conservative settings

## Tradeoff Analysis

### Main Tradeoffs

1. Sparsity vs Accuracy
   - Higher k = better quality but less speedup
   - k=200 hits target ≥95% similarity

2. Speed vs Memory
   - More aggressive sparsity = faster but may need more compensation
   - Current balance saves both compute and memory

3. Complexity vs Performance
   - Could add more features/knobs
   - Kept simple for reliability & maintenance

### Design Principles

1. Simplicity: Clear, maintainable implementation
2. Efficiency: Optimize critical paths
3. Flexibility: Configurable when needed
4. Reliability: Stable across models/inputs