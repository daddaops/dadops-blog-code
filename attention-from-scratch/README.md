# Attention from Scratch

Verified, runnable code from the DadOps blog post:
[Attention from Scratch](https://daddaops.com/blog/attention-from-scratch/)

## Scripts

| Script | Description |
|--------|-------------|
| `dot_product_demo.py` | Dot product as similarity measure between embeddings |
| `scaled_attention.py` | Scaled dot-product attention on a toy sentence |
| `multi_head_attention.py` | Multi-head attention with reshape trick |
| `positional_encoding.py` | Sinusoidal positional encoding |
| `full_pipeline.py` | Complete pipeline: embeddings + PE + multi-head attention |

## Usage

```bash
pip install -r requirements.txt
python3 dot_product_demo.py
python3 scaled_attention.py
python3 multi_head_attention.py
python3 positional_encoding.py
python3 full_pipeline.py
```

Note: The blog also shows a PyTorch `nn.MultiheadAttention` example (3 lines),
which is not included here as it requires PyTorch installation.
