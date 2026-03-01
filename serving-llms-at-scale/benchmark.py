"""
Serving Framework Benchmark Suite — head-to-head comparison of 4 serving strategies.

Blog post: https://dadops.dev/blog/serving-llms-at-scale/
Code Block 5 from "Serving LLMs at Scale: From Naive to vLLM"
"""
import random
import statistics

def benchmark_serving_approaches(
    num_requests=200, batch_size=16,
    min_tokens=20, max_tokens=1000,
    max_memory_pages=512, tokens_per_page=16,
    seed=42
):
    """Benchmark four serving strategies on the same workload."""
    rng = random.Random(seed)
    output_lengths = [rng.randint(min_tokens, max_tokens) for _ in range(num_requests)]
    total_tokens = sum(output_lengths)

    results = {}

    # 1. Naive sequential
    naive_steps = sum(output_lengths)
    naive_throughput = total_tokens / naive_steps  # 1.0 tokens/step always
    results["Naive Sequential"] = {
        "throughput": naive_throughput,
        "total_steps": naive_steps,
        "avg_latency": statistics.mean(output_lengths),
        "gpu_util": "~15%",
    }

    # 2. Static batching
    static_steps = 0
    static_latencies = []
    for i in range(0, num_requests, batch_size):
        batch = output_lengths[i:i + batch_size]
        max_len = max(batch)
        static_steps += max_len
        static_latencies.extend([max_len] * len(batch))

    results["Static Batching"] = {
        "throughput": total_tokens / static_steps,
        "total_steps": static_steps,
        "avg_latency": statistics.mean(static_latencies),
        "gpu_util": "~55%",
    }

    # 3. Continuous batching
    queue = list(range(num_requests))
    active = {}
    cont_steps = 0
    cont_latencies = {}
    arrival = {i: 0 for i in range(num_requests)}

    for _ in range(min(batch_size, len(queue))):
        rid = queue.pop(0)
        active[rid] = output_lengths[rid]

    while active or queue:
        cont_steps += 1
        done = [r for r in active if active[r] <= 1]
        for r in done:
            del active[r]
            cont_latencies[r] = cont_steps - arrival[r]
        for r in list(active):
            active[r] -= 1
        while len(active) < batch_size and queue:
            rid = queue.pop(0)
            active[rid] = output_lengths[rid]
            arrival[rid] = cont_steps

    results["Continuous Batching"] = {
        "throughput": total_tokens / cont_steps,
        "total_steps": cont_steps,
        "avg_latency": statistics.mean(cont_latencies.values()),
        "gpu_util": "~78%",
    }

    # 4. PagedAttention + continuous batching (higher effective batch)
    # PagedAttention allows ~3x more concurrent requests due to memory savings
    effective_batch = min(batch_size * 3, num_requests)
    queue2 = list(range(num_requests))
    active2 = {}
    paged_steps = 0
    paged_latencies = {}
    arrival2 = {i: 0 for i in range(num_requests)}

    for _ in range(min(effective_batch, len(queue2))):
        rid = queue2.pop(0)
        active2[rid] = output_lengths[rid]

    while active2 or queue2:
        paged_steps += 1
        done = [r for r in active2 if active2[r] <= 1]
        for r in done:
            del active2[r]
            paged_latencies[r] = paged_steps - arrival2[r]
        for r in list(active2):
            active2[r] -= 1
        while len(active2) < effective_batch and queue2:
            rid = queue2.pop(0)
            active2[rid] = output_lengths[rid]
            arrival2[rid] = paged_steps

    results["PagedAttention (vLLM)"] = {
        "throughput": total_tokens / paged_steps,
        "total_steps": paged_steps,
        "avg_latency": statistics.mean(paged_latencies.values()),
        "gpu_util": "~94%",
    }

    # Print comparison table
    print(f"Benchmark: {num_requests} requests, {min_tokens}-{max_tokens} tokens, batch={batch_size}")
    print(f"Total tokens to generate: {total_tokens:,}\n")
    print(f"{'Approach':<28} {'Tok/Step':>10} {'Steps':>10} {'Avg Lat':>10} {'GPU Util':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<28} {r['throughput']:>10.1f} {r['total_steps']:>10,} "
              f"{r['avg_latency']:>10.0f} {r['gpu_util']:>10}")

    # Show scaling: throughput at different concurrency levels
    print(f"\n{'Concurrency':<15} {'Static':>10} {'Continuous':>12} {'Paged':>10}")
    print("-" * 50)
    for conc in [1, 4, 8, 16, 32, 64]:
        s_tp = min(conc, batch_size) * 0.6  # Static: limited by batch, 60% efficient
        c_tp = min(conc, batch_size) * 0.95  # Continuous: 95% efficient up to batch
        p_tp = min(conc, batch_size * 3) * 0.95  # Paged: 3x effective batch
        print(f"{conc:<15} {s_tp:>10.1f} {c_tp:>12.1f} {p_tp:>10.1f}")


if __name__ == "__main__":
    benchmark_serving_approaches()
