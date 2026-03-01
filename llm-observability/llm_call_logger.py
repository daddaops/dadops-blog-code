"""
Code Block 1 & 2: Structured logging decorator for async LLM calls.

From: https://dadops.dev/blog/llm-observability/

Wraps async LLM calls with structured JSON logging, capturing:
timestamp, trace ID, model, feature, tokens, cost, latency, TTFT,
temperature, prompt hash, and finish reason.

No external dependencies required — uses only standard library.
The decorator assumes the wrapped function returns an OpenAI-compatible
response object (with .model, .usage.input_tokens, etc.).
"""

import time, json, hashlib, uuid
from dataclasses import dataclass, asdict
from functools import wraps
import logging

logger = logging.getLogger("llm_observability")

MODEL_PRICING = {           # (input, output) per 1M tokens
    "gpt-4o":           (2.50,  10.00),
    "gpt-4o-mini":      (0.15,   0.60),
    "claude-sonnet-4":  (3.00,  15.00),
}

@dataclass
class LLMCallLog:
    timestamp: str;  trace_id: str;  model: str
    feature: str;    input_tokens: int
    output_tokens: int;  cost_usd: float
    latency_ms: float;   ttft_ms: float
    temperature: float;  prompt_hash: str
    finish_reason: str

def log_llm_call(feature: str):
    """Decorator that wraps an async LLM call with structured logging."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            trace_id = kwargs.pop("trace_id", uuid.uuid4().hex[:16])
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            model = result.model
            inp, out = MODEL_PRICING.get(model, (0, 0))
            cost = (result.usage.input_tokens * inp
                    + result.usage.output_tokens * out) / 1_000_000

            entry = LLMCallLog(
                timestamp     = time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                trace_id      = trace_id,
                model         = model,
                feature       = feature,
                input_tokens  = result.usage.input_tokens,
                output_tokens = result.usage.output_tokens,
                cost_usd      = round(cost, 6),
                latency_ms    = round(elapsed, 1),
                ttft_ms       = round(elapsed * 0.15, 1),
                temperature   = kwargs.get("temperature", 1.0),
                prompt_hash   = hashlib.sha256(
                    str(args[0]).encode()).hexdigest()[:12],
                finish_reason = result.choices[0].finish_reason,
            )
            logger.info(json.dumps(asdict(entry)))
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    import asyncio

    # Mock an OpenAI-compatible response object for testing
    class MockUsage:
        def __init__(self, inp, out):
            self.input_tokens = inp
            self.output_tokens = out

    class MockChoice:
        def __init__(self, finish_reason="stop"):
            self.finish_reason = finish_reason

    class MockResponse:
        def __init__(self, model, inp, out):
            self.model = model
            self.usage = MockUsage(inp, out)
            self.choices = [MockChoice()]

    @log_llm_call(feature="document-summarizer")
    async def mock_llm_call(prompt, **kwargs):
        await asyncio.sleep(0.01)  # simulate latency
        return MockResponse("gpt-4o", 2847, 312)

    # Verify the cost calculation from the blog's JSON example
    # Blog claims: cost_usd = 0.010237 for gpt-4o with 2847 input, 312 output
    # Calculation: (2847 * 2.50 + 312 * 10.00) / 1_000_000
    cost = (2847 * 2.50 + 312 * 10.00) / 1_000_000
    print(f"=== LLM Call Logger ===")
    print(f"Cost calculation: (2847 * 2.50 + 312 * 10.00) / 1_000_000 = {cost}")
    print(f"Rounded to 6 decimals: {round(cost, 6)}")
    print(f"Blog claims: 0.010237")
    print()

    # Run the decorated function and capture the log
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = asyncio.run(mock_llm_call("Summarize this document"))
    print(f"\nResponse model: {result.model}")
    print(f"Input tokens: {result.usage.input_tokens}")
    print(f"Output tokens: {result.usage.output_tokens}")

    # Expected output:
    # Cost calculation shows 0.010238 (blog says 0.010237 — minor rounding difference)
    # JSON log entry is printed via logger.info
