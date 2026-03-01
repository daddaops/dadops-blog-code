"""LLM API Latency Benchmark Client

Measures streaming LLM API latency with per-phase timing:
- TTFT (Time to First Token)
- ITL (Inter-Token Latency) distribution
- Total end-to-end time

Requires API keys: set OPENAI_API_KEY and/or ANTHROPIC_API_KEY environment variables.

Blog post: https://dadops.co/blog/llm-api-latency-benchmarks/
Code Blocks 1-4 from the blog.
"""

import asyncio
import os
import time
import httpx
import json
import statistics
from dataclasses import dataclass, field

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class RequestMetrics:
    ttft: float = 0.0           # Time to first token (seconds)
    itl_values: list = field(default_factory=list)  # Inter-token latencies
    total_time: float = 0.0     # End-to-end wall clock
    output_tokens: int = 0
    error: str | None = None


async def benchmark_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
) -> RequestMetrics:
    """Measure a single streaming LLM API request with per-phase timing."""
    metrics = RequestMetrics()
    t_start = time.perf_counter()
    last_token_time = t_start
    first_token_seen = False

    try:
        async with client.stream("POST", url, json=payload,
                                 headers=headers, timeout=30.0) as resp:
            if resp.status_code != 200:
                metrics.error = f"HTTP {resp.status_code}"
                return metrics

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break

                chunk = json.loads(data)
                # Extract token content (OpenAI format)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if content and not first_token_seen:
                    now = time.perf_counter()
                    metrics.ttft = now - t_start
                    last_token_time = now
                    first_token_seen = True
                    metrics.output_tokens += 1
                elif content:
                    now = time.perf_counter()
                    metrics.itl_values.append(now - last_token_time)
                    last_token_time = now
                    metrics.output_tokens += 1

    except (httpx.TimeoutException, httpx.ConnectError) as e:
        metrics.error = str(type(e).__name__)
    finally:
        metrics.total_time = time.perf_counter() - t_start

    return metrics


async def measure_single_request(
    client: httpx.AsyncClient,
    provider: str,
    model: str,
    prompt: str,
    max_tokens: int = 200,
) -> RequestMetrics:
    """Run a single timed request against any provider."""
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
    else:  # local vLLM — OpenAI-compatible endpoint
        url = "http://localhost:8000/v1/chat/completions"
        headers = {}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }

    return await benchmark_streaming_request(client, url, headers, payload)


async def run_concurrency_benchmark(
    provider: str,
    model: str,
    prompt: str,
    concurrency: int,
    num_requests: int = 100,
    warmup: int = 5,
) -> list[RequestMetrics]:
    """Run num_requests with exactly `concurrency` in flight at once."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestMetrics] = []
    completed = 0

    async with httpx.AsyncClient(http2=True) as client:
        # Warm-up phase — discard these results
        warmup_tasks = [
            measure_single_request(client, provider, model, prompt)
            for _ in range(warmup)
        ]
        await asyncio.gather(*warmup_tasks)

        # Measurement phase
        async def bounded_request():
            nonlocal completed
            async with semaphore:
                result = await measure_single_request(
                    client, provider, model, prompt
                )
                completed += 1
                return result

        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    # Filter out errors for metric computation
    successful = [r for r in results if r.error is None]
    errors = [r for r in results if r.error is not None]

    print(f"\n{provider}/{model} @ concurrency={concurrency}")
    print(f"  Successful: {len(successful)}/{num_requests}")
    print(f"  Error rate: {len(errors)/num_requests*100:.1f}%")

    if successful:
        ttfts = sorted([r.ttft for r in successful])
        print(f"  TTFT  p50={ttfts[len(ttfts)//2]*1000:.0f}ms  "
              f"p95={ttfts[int(len(ttfts)*0.95)]*1000:.0f}ms  "
              f"p99={ttfts[int(len(ttfts)*0.99)]*1000:.0f}ms")

    return results


async def measure_prompt_length_scaling(
    provider: str,
    model: str,
    base_prompt: str,
    token_counts: list[int],
    runs_per_count: int = 50,
) -> dict[int, dict]:
    """Measure TTFT and ITL at different input prompt lengths."""
    results = {}
    # Pad prompt to approximate target token counts
    # (~4 chars per token is a rough English estimate)
    padding_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    )

    async with httpx.AsyncClient(http2=True) as client:
        for target_tokens in token_counts:
            chars_needed = target_tokens * 4
            padded = (padding_text * (chars_needed // len(padding_text) + 1))
            prompt = padded[:chars_needed] + "\n\n" + base_prompt

            metrics_list = []
            for _ in range(runs_per_count):
                m = await measure_single_request(
                    client, provider, model, prompt, max_tokens=50
                )
                if m.error is None:
                    metrics_list.append(m)

            ttfts = sorted([m.ttft for m in metrics_list])
            itls_flat = []
            for m in metrics_list:
                itls_flat.extend(m.itl_values)
            itls_flat.sort()

            results[target_tokens] = {
                "ttft_p50": ttfts[len(ttfts) // 2] * 1000,
                "ttft_p95": ttfts[int(len(ttfts) * 0.95)] * 1000,
                "itl_p50": (itls_flat[len(itls_flat) // 2] * 1000
                            if itls_flat else 0),
            }
            print(f"  {target_tokens} tokens: "
                  f"TTFT p50={results[target_tokens]['ttft_p50']:.0f}ms "
                  f"ITL p50={results[target_tokens]['itl_p50']:.1f}ms")

    return results


if __name__ == "__main__":
    print("LLM API Latency Benchmark Client")
    print("=" * 50)

    if not OPENAI_KEY and not ANTHROPIC_KEY:
        print("\nNo API keys found. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY.")
        print("Skipping live benchmarks.")
        print("\nTo run:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  python benchmark_client.py")
    else:
        prompt = (
            "Explain the key differences between batch processing and stream "
            "processing in data engineering. Include specific use cases for each "
            "approach and discuss when you might use a hybrid strategy."
        )

        if OPENAI_KEY:
            print("\n--- OpenAI GPT-4o-mini ---")
            asyncio.run(run_concurrency_benchmark(
                "openai", "gpt-4o-mini", prompt, concurrency=1, num_requests=10
            ))

        if ANTHROPIC_KEY:
            print("\n--- Anthropic Claude 3.5 Haiku ---")
            asyncio.run(run_concurrency_benchmark(
                "anthropic", "claude-3-5-haiku-latest", prompt,
                concurrency=1, num_requests=10
            ))
