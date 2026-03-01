# Serving LLMs at Scale: From Naive to vLLM

Verified, runnable code from the DadOps blog post:
**[Serving LLMs at Scale](https://dadops.dev/blog/serving-llms-at-scale/)**

## Scripts

| Script | Description |
|--------|-------------|
| `static_batching.py` | Simulates static batching and measures padding waste |
| `continuous_batching.py` | Simulates iteration-level scheduling (Orca-style) |
| `paged_attention.py` | PagedAttention memory manager with prefix sharing |
| `speculative_decoding.py` | Draft-verify pipeline cost simulator |
| `benchmark.py` | Head-to-head comparison of all four serving strategies |

## Run

```bash
pip install -r requirements.txt
python3 static_batching.py
python3 continuous_batching.py
python3 paged_attention.py
python3 speculative_decoding.py
python3 benchmark.py
```

## Dependencies

Pure Python standard library — no external packages required.
