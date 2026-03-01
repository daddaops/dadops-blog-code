# LLM Cost Optimization — Verified Code

Code from the DadOps blog post: [LLM Cost Optimization: Cutting Your API Bill by 80% Without Sacrificing Quality](https://dadops.dev/blog/llm-cost-optimization/)

## Scripts

| Script | Description | Blog Section |
|--------|-------------|--------------|
| `cost_profiler.py` | Per-feature cost tracking with pricing data | Section 1: Understanding Your LLM Cost Structure |
| `prompt_compressor.py` | Three prompt compression techniques | Section 2: Prompt Optimization |
| `model_router.py` | Keyword-based model tier routing | Section 3: Model Routing |
| `tiered_cache.py` | Three-tier cache (exact, semantic, prefix) | Section 4: Caching Strategies |
| `token_budget.py` | Per-feature token budgets with degradation | Section 5: Token Budget Management |
| `batch_processor.py` | Batch API queue with 50% discount | Section 6: Batching and Async Processing |
| `cost_audit.py` | Usage log analyzer with optimization recs | Section 7: The Complete Optimization Stack |

## Running

```bash
pip install -r requirements.txt
python3 cost_profiler.py
python3 prompt_compressor.py
python3 model_router.py
python3 tiered_cache.py
python3 token_budget.py
python3 batch_processor.py
python3 cost_audit.py
```

All scripts are pure Python with no external dependencies — they use only the standard library.

## Notes

- All pricing data is from February 2026
- No API keys required — scripts use simulated data to verify math
- Each script verifies the blog's numerical claims in its output
