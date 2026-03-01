"""
Benchmark all three cache strategies on the 10K query dataset.

Blog post: https://dadops.dev/blog/caching-llm-responses/
Code Blocks 5 and 6.

Measures: hit rate, false positive rate, latency percentiles, and cost savings.
No API keys required — this is a self-contained benchmark.

Blog claims:
  - Exact Match: 23% hit rate, 0.2% FP, p50=0.02ms
  - Structural Hash: 46% hit rate, 0.3% FP, p50=0.04ms
  - Semantic (θ=0.90): 80% hit rate, 0.3% FP, p50=15ms
  - Cost savings at 100K queries/day with GPT-4o: $7,800/month (80% hit rate)
"""
import time
import numpy as np

from dataset import generate_query_dataset
from caches import ExactMatchCache, SemanticCache, StructuralHashCache


# ── Code Block 5: Benchmark Function ──

def benchmark_strategy(cache, queries):
    """Run a cache strategy against the full query dataset.

    Returns metrics: hit_rate, latencies, false_positives.
    """
    hits, misses, false_positives = 0, 0, 0
    latencies = []
    seen_intents = {}  # intent -> first cached response

    for q in queries:
        result, latency_ms = cache.get(q["text"])
        latencies.append(latency_ms)

        if result is not None:
            hits += 1
            # Check for false positives: did we return a response
            # originally cached for a DIFFERENT intent?
            if q["intent"] in seen_intents:
                if result != seen_intents[q["intent"]]:
                    false_positives += 1
        else:
            misses += 1
            response = f"Response for intent: {q['intent']}"
            cache.put(q["text"], response)
            if q["intent"] not in seen_intents:
                seen_intents[q["intent"]] = response

    total = hits + misses
    latencies.sort()
    return {
        "hit_rate": hits / total * 100,
        "hits": hits,
        "misses": misses,
        "false_positives": false_positives,
        "fp_rate": false_positives / max(hits, 1) * 100,
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
    }


# ── Code Block 6: Cost Savings Calculator ──

def monthly_savings(queries_per_day, avg_input_tokens, avg_output_tokens,
                     input_cost_per_m, output_cost_per_m, hit_rate):
    """Calculate monthly cost savings from LLM response caching.

    Args:
        queries_per_day: Daily query volume
        avg_input_tokens: Average input tokens per query
        avg_output_tokens: Average output tokens per query
        input_cost_per_m: $ per 1M input tokens
        output_cost_per_m: $ per 1M output tokens
        hit_rate: Cache hit rate as a decimal (0.0 - 1.0)

    Returns:
        Dict with cost breakdown
    """
    daily_input_cost = (queries_per_day * avg_input_tokens / 1e6) * input_cost_per_m
    daily_output_cost = (queries_per_day * avg_output_tokens / 1e6) * output_cost_per_m
    daily_total = daily_input_cost + daily_output_cost

    # Cache hits avoid the full API call
    daily_savings = daily_total * hit_rate
    monthly_total = daily_total * 30
    monthly_sav = daily_savings * 30

    return {
        "daily_cost_no_cache": round(daily_total, 2),
        "daily_cost_with_cache": round(daily_total - daily_savings, 2),
        "monthly_savings": round(monthly_sav, 2),
        "monthly_cost_no_cache": round(monthly_total, 2),
        "annual_savings": round(monthly_sav * 12, 2),
    }


if __name__ == "__main__":
    print("=== LLM Response Cache Benchmark ===\n")

    # Generate dataset
    print("Generating 10K query dataset...")
    dataset = generate_query_dataset(10000)
    n_unique = len(set(q['intent'] for q in dataset))
    print(f"Total queries: {len(dataset)}")
    print(f"Unique intents: {n_unique}\n")

    # Benchmark exact match
    print("Benchmarking Exact Match Cache...")
    t0 = time.perf_counter()
    exact_results = benchmark_strategy(ExactMatchCache(), dataset)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.1f}s")

    # Benchmark structural hash
    print("Benchmarking Structural Hash Cache...")
    t0 = time.perf_counter()
    structural_results = benchmark_strategy(StructuralHashCache(), dataset)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.1f}s")

    # Benchmark semantic cache (this takes longer due to embedding)
    print("Benchmarking Semantic Cache (θ=0.90)...")
    print("  (Loading embedding model, this may take a moment...)")
    t0 = time.perf_counter()
    semantic_results = benchmark_strategy(SemanticCache(threshold=0.90), dataset)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.1f}s")

    # ── Hit Rate Comparison ──
    print("\n" + "=" * 60)
    print("HIT RATE COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Hit Rate':>10} {'TP%':>8} {'FP%':>8}")
    print("-" * 48)
    for name, r in [("Exact Match", exact_results),
                    ("Structural Hash", structural_results),
                    ("Semantic θ=0.90", semantic_results)]:
        tp_rate = 100 - r['fp_rate']
        print(f"{name:<20} {r['hit_rate']:>9.1f}% {tp_rate:>7.1f}% {r['fp_rate']:>7.1f}%")

    print(f"\nBlog claims: Exact=23%, Structural=46%, Semantic=80%")

    # ── Latency Comparison ──
    print("\n" + "=" * 60)
    print("LATENCY COMPARISON (ms)")
    print("=" * 60)
    print(f"{'Strategy':<20} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 46)
    for name, r in [("Exact Match", exact_results),
                    ("Structural Hash", structural_results),
                    ("Semantic θ=0.90", semantic_results)]:
        print(f"{name:<20} {r['p50_ms']:>7.2f} {r['p95_ms']:>7.2f} {r['p99_ms']:>7.2f}")

    print(f"\nBlog claims: Exact p50=0.02ms, Structural p50=0.04ms, Semantic p50=15ms")

    # ── Threshold Sweep (semantic cache) ──
    print("\n" + "=" * 60)
    print("SEMANTIC CACHE THRESHOLD SWEEP")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Hit Rate':>10} {'TP%':>8} {'FP%':>8}")
    print("-" * 38)
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        r = benchmark_strategy(SemanticCache(threshold=threshold), dataset)
        tp_rate = 100 - r['fp_rate']
        marker = " <-- recommended" if threshold == 0.90 else ""
        print(f"{threshold:>10.2f} {r['hit_rate']:>9.1f}% {tp_rate:>7.1f}% {r['fp_rate']:>7.1f}%{marker}")

    print(f"\nBlog claims: θ=0.80→91%/2.3%FP, θ=0.85→88%/1.2%FP, "
          f"θ=0.90→80%/0.3%FP, θ=0.95→65%/0.2%FP")

    # ── Cost Savings (using ACTUAL measured hit rates) ──
    print("\n" + "=" * 60)
    print("COST SAVINGS (GPT-4o, 100K queries/day)")
    print("=" * 60)

    actual_rates = {
        "Exact Match": exact_results['hit_rate'] / 100,
        "Structural Hash": structural_results['hit_rate'] / 100,
        "Semantic θ=0.90": semantic_results['hit_rate'] / 100,
    }

    for name, hit_rate in actual_rates.items():
        savings = monthly_savings(
            queries_per_day=100_000,
            avg_input_tokens=500,
            avg_output_tokens=200,
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
            hit_rate=hit_rate
        )
        pct = hit_rate * 100
        print(f"{name} ({pct:.0f}%)".ljust(28)
              + f"Monthly savings: ${savings['monthly_savings']:,.2f}  "
              f"(No cache: ${savings['monthly_cost_no_cache']:,.2f}/mo)")

    # Show the baseline (no cache) cost
    baseline = monthly_savings(
        queries_per_day=100_000,
        avg_input_tokens=500,
        avg_output_tokens=200,
        input_cost_per_m=2.50,
        output_cost_per_m=10.00,
        hit_rate=0.0
    )
    print(f"\nBaseline: ${baseline['monthly_cost_no_cache']:,.2f}/mo without cache")

    print("\nAll benchmarks complete!")
