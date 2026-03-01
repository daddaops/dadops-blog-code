"""
Async benchmark for vLLM concurrent throughput.

From: https://dadops.dev/blog/local-llm-deployment/

Fires N requests simultaneously to measure aggregate throughput
with continuous batching. Requires vLLM running locally.

Dependencies: openai
"""

import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)


async def single_request(prompt: str) -> dict:
    """Send one request and measure timing."""
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.7
    )
    elapsed = time.perf_counter() - start
    tokens = response.usage.completion_tokens
    return {"tokens": tokens, "time": elapsed, "tok_per_sec": tokens / elapsed}


async def benchmark_concurrent(num_requests: int):
    """Fire N requests simultaneously and measure aggregate throughput."""
    prompts = [f"Write a haiku about the number {i}" for i in range(num_requests)]
    start = time.perf_counter()
    results = await asyncio.gather(*[single_request(p) for p in prompts])
    wall_time = time.perf_counter() - start

    total_tokens = sum(r["tokens"] for r in results)
    print(f"Concurrent requests: {num_requests}")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Aggregate throughput: {total_tokens / wall_time:.1f} tok/s")
    print(f"  Avg per-request: {sum(r['tok_per_sec'] for r in results) / len(results):.1f} tok/s")
    print()


async def main():
    for n in [1, 5, 10, 20]:
        await benchmark_concurrent(n)


if __name__ == "__main__":
    print("=== vLLM Concurrent Benchmark ===")
    print("Requires vLLM running: vllm serve meta-llama/Llama-3.1-8B")
    print()
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Connection failed (vLLM not running?): {e}")
