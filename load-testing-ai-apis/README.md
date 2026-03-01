# Load Testing AI APIs — Verified Code

Code from the DadOps blog post: [Load Testing AI APIs: Why Standard Tools Fail and What to Use Instead](https://dadops.dev/blog/load-testing-ai-apis/)

## Scripts

| Script | Description | Blog Section | Requires API? |
|--------|-------------|--------------|---------------|
| `kv_cache_math.py` | KV cache memory calculation verification | Section 1: Why AI APIs Break Differently | No |
| `load_tester.py` | Async SSE load tester + concurrency sweep + metrics | Sections 2, 3, 5 | Yes |
| `load_patterns.py` | Ramp, sustained, burst, soak pattern generators | Section 4: Load Patterns | No |
| `regression_detector.py` | Welch's t-test regression detection | Section 6: Detecting Degradation | No |

## Running

```bash
pip install -r requirements.txt

# These run without any API:
python3 kv_cache_math.py
python3 load_patterns.py
python3 regression_detector.py

# This requires a running LLM API endpoint:
export LLM_API_URL="https://api.example.com/v1/chat/completions"
export LLM_API_KEY="your-key-here"
python3 load_tester.py
```

## Notes

- `kv_cache_math.py`, `load_patterns.py`, and `regression_detector.py` are pure Python (stdlib only)
- `load_tester.py` requires `aiohttp` and a running LLM API with SSE streaming
- Without API credentials, `load_tester.py` runs in verification mode (tests code structure only)
- The blog's benchmark table (Llama-3 8B on A100) cannot be reproduced without GPU hardware
