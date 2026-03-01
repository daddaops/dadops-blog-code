"""
Async Load Tester for LLM APIs — core tester + concurrency sweep + metrics.

From: https://dadops.dev/blog/load-testing-ai-apis/
Code Blocks 2, 3, 5: "Building an Async Load Tester", "Concurrency Curve",
                       "AI-Specific Metrics"

REQUIRES: A running LLM API endpoint with SSE streaming.
          Set LLM_API_URL and LLM_API_KEY environment variables.

Usage:
  export LLM_API_URL="https://api.example.com/v1/chat/completions"
  export LLM_API_KEY="your-key-here"
  python3 load_tester.py
"""

import asyncio
import os
import time
from dataclasses import dataclass, field

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class RequestMetrics:
    ttft: float = 0.0          # time to first token
    ttlt: float = 0.0          # time to last token
    token_count: int = 0
    tpot: float = 0.0          # time per output token
    token_times: list = field(default_factory=list)
    error: str = ""


async def stream_request(session, url, payload, semaphore):
    """Send one request, parse SSE stream, collect timing."""
    async with semaphore:
        m = RequestMetrics()
        start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                async for raw_line in resp.content:
                    line = raw_line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    now = time.perf_counter()
                    if m.token_count == 0:
                        m.ttft = now - start
                    m.token_times.append(now)
                    m.token_count += 1
                m.ttlt = time.perf_counter() - start
                if m.token_count > 1:
                    m.tpot = (m.ttlt - m.ttft) / (m.token_count - 1)
        except Exception as e:
            m.error = str(e)
        return m


def percentile(sorted_data, p):
    """Compute the p-th percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def run_load_test(url, payload, concurrency, num_requests):
    """Fire num_requests at url with given concurrency cap."""
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_request(session, url, payload, sem)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

    ok = [r for r in results if not r.error]
    ttfts = sorted(r.ttft for r in ok)
    total_tokens = sum(r.token_count for r in ok)
    wall = max(r.ttlt for r in ok) if ok else 1

    print(f"Concurrency: {concurrency}")
    print(f"  Successful: {len(ok)}/{num_requests}")
    print(f"  Throughput: {total_tokens / wall:.0f} tokens/sec")
    print(f"  TTFT  p50={percentile(ttfts, 50)*1000:.0f}ms"
          f"  p95={percentile(ttfts, 95)*1000:.0f}ms"
          f"  p99={percentile(ttfts, 99)*1000:.0f}ms")
    return results


async def concurrency_sweep(url, payload, levels=None, requests_per=100):
    """Sweep concurrency levels and collect the two curves."""
    if levels is None:
        levels = [1, 2, 4, 8, 16, 32, 64, 128]

    rows = []
    for conc in levels:
        results = await run_load_test(url, payload, conc, requests_per)
        ok = [r for r in results if not r.error]
        ttfts = sorted(r.ttft for r in ok)
        total_tok = sum(r.token_count for r in ok)
        wall = max(r.ttlt for r in ok) if ok else 1
        err_rate = (len(results) - len(ok)) / len(results) * 100

        rows.append({
            "concurrency": conc,
            "throughput": total_tok / wall,
            "ttft_p50": percentile(ttfts, 50) * 1000,
            "ttft_p95": percentile(ttfts, 95) * 1000,
            "ttft_p99": percentile(ttfts, 99) * 1000,
            "error_pct": err_rate,
        })

    # Print results table
    print(f"{'Conc':>6} {'TPS':>8} {'p50ms':>8} {'p95ms':>8} "
          f"{'p99ms':>8} {'Err%':>6}")
    print("-" * 50)
    for r in rows:
        print(f"{r['concurrency']:>6} {r['throughput']:>8.0f} "
              f"{r['ttft_p50']:>8.0f} {r['ttft_p95']:>8.0f} "
              f"{r['ttft_p99']:>8.0f} {r['error_pct']:>5.1f}%")
    return rows


def compute_ai_metrics(results, cost_per_token=0.0):
    """Compute all six AI-specific metrics from load test results."""
    ok = [r for r in results if not r.error]
    if not ok:
        return {"error": "No successful requests"}

    ttfts = sorted(r.ttft for r in ok)
    tpots = sorted(r.tpot for r in ok if r.tpot > 0)
    total_tokens = sum(r.token_count for r in ok)
    wall = max(r.ttlt for r in ok)

    # Inter-token latency jitter
    all_itls = []
    for r in ok:
        for i in range(1, len(r.token_times)):
            all_itls.append(r.token_times[i] - r.token_times[i - 1])
    itl_cv = 0.0
    if all_itls:
        mean_itl = sum(all_itls) / len(all_itls)
        var_itl = sum((x - mean_itl) ** 2 for x in all_itls) / len(all_itls)
        itl_cv = (var_itl ** 0.5) / mean_itl if mean_itl > 0 else 0

    metrics = {
        "ttft_p50":   percentile(ttfts, 50) * 1000,
        "ttft_p95":   percentile(ttfts, 95) * 1000,
        "tpot_p50":   percentile(tpots, 50) * 1000 if tpots else 0,
        "system_tps": total_tokens / wall,
        "itl_cv":     itl_cv,
        "p99_p50":    percentile(ttfts, 99) / max(percentile(ttfts, 50), 1e-9),
        "error_pct":  (len(results) - len(ok)) / len(results) * 100,
        "cost":       total_tokens * cost_per_token,
    }

    # Threshold checks
    checks = [
        ("TTFT p95",    metrics["ttft_p95"],   200, 1000,  "ms"),
        ("TPOT p50",    metrics["tpot_p50"],   30,  80,    "ms"),
        ("System TPS",  metrics["system_tps"], 2000, 1000, "tok/s"),  # inverted
        ("ITL jitter",  metrics["itl_cv"],     0.3, 0.5,   "CV"),
        ("p99/p50",     metrics["p99_p50"],    3,   10,    "x"),
        ("Error rate",  metrics["error_pct"],  0.1, 1.0,   "%"),
    ]

    for name, val, good_thresh, warn_thresh, unit in checks:
        if name == "System TPS":
            status = "GOOD" if val > good_thresh else (
                     "WARN" if val > warn_thresh else "FAIL")
        else:
            status = "GOOD" if val < good_thresh else (
                     "WARN" if val < warn_thresh else "FAIL")
        marker = {"GOOD": "+", "WARN": "~", "FAIL": "!"}[status]
        print(f"  [{marker}] {name:<14} {val:>8.1f} {unit:<6} [{status}]")
    return metrics


if __name__ == "__main__":
    if not HAS_AIOHTTP:
        print("ERROR: aiohttp not installed. Run: pip install aiohttp")
        print("SKIP: Load tester requires aiohttp and a running LLM API endpoint")
        exit(1)

    url = os.environ.get("LLM_API_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")

    if not url:
        print("SKIP: No LLM_API_URL set. This script requires a running LLM API.")
        print("  Set LLM_API_URL and LLM_API_KEY environment variables to run.")
        print("\nVerifying code structure only (no API calls)...")

        # Verify the code compiles and classes work
        m = RequestMetrics()
        m.ttft = 0.045
        m.ttlt = 1.2
        m.token_count = 50
        m.tpot = 0.023
        m.token_times = [0.045 + i * 0.023 for i in range(50)]

        print(f"\n  RequestMetrics structure: OK")
        print(f"    TTFT: {m.ttft*1000:.0f} ms")
        print(f"    TTLT: {m.ttlt*1000:.0f} ms")
        print(f"    Tokens: {m.token_count}")
        print(f"    TPOT: {m.tpot*1000:.0f} ms")

        # Test percentile function
        data = sorted([0.04, 0.045, 0.05, 0.052, 0.06, 0.048, 0.055, 0.047,
                       0.051, 0.058])
        print(f"\n  percentile() test:")
        print(f"    p50={percentile(data, 50)*1000:.1f} ms")
        print(f"    p95={percentile(data, 95)*1000:.1f} ms")
        print(f"    p99={percentile(data, 99)*1000:.1f} ms")

        # Test compute_ai_metrics with synthetic data
        print(f"\n  compute_ai_metrics() with synthetic data:")
        synthetic = []
        for i in range(20):
            r = RequestMetrics()
            r.ttft = 0.045 + i * 0.002
            r.ttlt = 1.0 + i * 0.05
            r.token_count = 40 + i
            r.tpot = 0.022 + i * 0.001
            r.token_times = [r.ttft + j * r.tpot for j in range(r.token_count)]
            synthetic.append(r)
        compute_ai_metrics(synthetic)

        print("\nAll code structures verified OK (no API calls made)")
    else:
        payload = {
            "model": "meta-llama/Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": "Explain quicksort."}],
            "stream": True,
            "max_tokens": 200,
        }
        asyncio.run(run_load_test(url, payload, concurrency=4, num_requests=20))
