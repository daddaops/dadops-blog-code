# Caching LLM Responses

Verified code from the DadOps blog post: [Caching LLM Responses](https://dadops.dev/blog/caching-llm-responses/)

## Scripts

| Script | Blog Blocks | Description |
|--------|-------------|-------------|
| `dataset.py` | Block 1 | 10K query dataset generator with 140 intents and paraphrase clusters |
| `caches.py` | Blocks 2-4 | Three cache implementations: ExactMatch, Semantic, StructuralHash |
| `benchmark.py` | Blocks 5-6 | Head-to-head benchmark + cost savings calculator |

## Running

```bash
pip install -r requirements.txt

# Generate and inspect the dataset
python dataset.py

# Run the full benchmark (includes threshold sweep and cost calculator)
python benchmark.py
```

## Notes

- No API keys required — all benchmarks are self-contained
- Semantic cache requires downloading `all-MiniLM-L6-v2` model (~120 MB)
- Full benchmark takes ~5-10 minutes due to embedding 10K queries
- Blog claims verified: hit rates, latency, false positive rates, cost savings
