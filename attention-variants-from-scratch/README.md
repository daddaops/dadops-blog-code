# Attention Variants from Scratch

Verified, runnable code from the DadOps blog post:
[Attention Variants from Scratch](https://daddaops.com/blog/attention-variants-from-scratch/)

## Scripts

| Script | Description |
|--------|-------------|
| `mha_vs_mqa.py` | Multi-Head vs Multi-Query Attention — KV cache comparison |
| `gqa.py` | Grouped-Query Attention — interpolates between MHA and MQA |
| `sliding_window.py` | Sliding Window Attention masks and cache bounds |
| `cross_attention.py` | Cross-Attention between encoder and decoder sequences |

## Usage

```bash
pip install -r requirements.txt
python3 mha_vs_mqa.py
python3 gqa.py
python3 sliding_window.py
python3 cross_attention.py
```
