# Batch Processing with LLMs: 10,000 API Calls Without Going Broke

Verified, runnable code from the DadOps blog post:
[Batch Processing with LLMs](https://dadops.dev/blog/batch-processing-llms/)

## Scripts

- `utilities.py` — Standalone utility classes: TokenBucket rate limiter, PromptCache (SHA-256 dedup), CostTracker (token accounting), Checkpoint (JSONL crash recovery). Includes self-tests that run without API keys.
- `retry_logic.py` — Exponential backoff with full jitter retry pattern. Includes unit tests with mock HTTP responses.
- `batch_processor.py` — Complete BatchProcessor class combining all utilities (async concurrency, rate limiting, retry, caching, checkpointing, cost tracking). Requires OpenAI API key to run.
- `example_usage.py` — Usage example and OpenAI Batch API workflow. Requires OpenAI API key.

## Usage

```bash
pip install -r requirements.txt

# These run without API keys:
python utilities.py
python retry_logic.py

# These require OPENAI_API_KEY:
# OPENAI_API_KEY=sk-... python example_usage.py
```
