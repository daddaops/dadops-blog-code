# KV Cache from Scratch

Verified, runnable code from the [KV Cache from Scratch](https://dadops.dev/blog/kv-cache-from-scratch/) blog post.

## Scripts

- **attention_setup.py** — Minimal attention layer with projection matrices (shared module)
- **naive_generate.py** — Naive generation: recomputes all K/V every step (O(n²))
- **cached_generate.py** — KV-cached generation: only projects new token (O(n))
- **prefill_decode.py** — Clean prefill + decode separation
- **gqa.py** — Grouped-Query Attention with KV cache (32 query heads, 8 KV groups)
- **rope_cache.py** — RoPE applied before caching, decode with rotated keys

## Run

```bash
pip install -r requirements.txt
python attention_setup.py
python naive_generate.py
python cached_generate.py
python prefill_decode.py
python gqa.py
python rope_cache.py
```
