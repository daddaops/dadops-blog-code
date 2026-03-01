# LLM API Latency Benchmarks — Verified Code

Runnable code from the DadOps blog post: [LLM API Latency Benchmarks](https://dadops.co/blog/llm-api-latency-benchmarks/)

## Scripts

- `benchmark_client.py` — Streaming LLM API benchmark client with per-phase timing (TTFT, ITL, total). **Requires API keys** — set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` environment variables.
- `cost_calculator.py` — Effective cost calculator accounting for retries, timeouts, and error rates. **Pure math, no API keys needed.**

## Quick Start

```bash
# Cost calculator (no API keys needed)
pip install -r requirements.txt
python cost_calculator.py

# Benchmark client (requires API keys)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
python benchmark_client.py
```

## Dependencies

- `httpx[http2]` — Async HTTP client with HTTP/2 support (for benchmark_client.py)

## Notes

- Benchmark numbers in the blog are point-in-time measurements from late 2024 — API performance changes over time
- The cost calculator is the only fully runnable script without external dependencies
