# Normalization from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/normalization-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `gradient_explosion.py` | Activation std growth across 50 layers |
| `batch_norm.py` | BatchNorm + batch dependence demo |
| `layer_norm.py` | LayerNorm batch-independence demo |
| `rms_norm.py` | RMSNorm vs BatchNorm vs LayerNorm statistics |
| `residual_comparison.py` | No-norm vs post-norm vs pre-norm stability |
| `performance_comparison.py` | Timing benchmark across all three norms |
| `llama_block.py` | LLaMA-style transformer block with RMSNorm |

## Usage

```bash
python gradient_explosion.py      # Why normalization matters
python batch_norm.py              # Batch normalization + dependence issue
python layer_norm.py              # Layer normalization
python rms_norm.py                # RMS normalization comparison
python residual_comparison.py     # Residual connection strategies
python performance_comparison.py  # Performance benchmark
python llama_block.py             # Full LLaMA block demo
```

Dependencies: numpy.
