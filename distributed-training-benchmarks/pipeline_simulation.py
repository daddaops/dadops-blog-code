"""Pipeline Parallelism Simulation — Bubble Ratio Analysis

Simulates GPipe-style pipeline parallelism to compute bubble ratios
across different stage/micro-batch configurations. Pure Python, no
GPU required.

Blog post: https://dadops.co/blog/distributed-training-benchmarks/
Code Block 3 from the blog.
"""


def simulate_pipeline(stages, micro_batches, fwd_time=1.0, bwd_time=2.0):
    """Simulate pipeline execution and compute bubble ratio.

    Returns dict with bubble_ratio, total_time, and effective_throughput.
    """
    # Total work per micro-batch per stage: forward + backward
    work_per_mb = fwd_time + bwd_time
    # In an ideal (no-bubble) pipeline, all stages are busy:
    ideal_time = micro_batches * work_per_mb
    # Actual pipeline time (GPipe schedule):
    # Fill: (P-1) forward steps, then M forward+backward, then (P-1) backward drain
    pipeline_time = (stages - 1) * fwd_time + micro_batches * work_per_mb \
                    + (stages - 1) * bwd_time
    bubble_time = pipeline_time - ideal_time  # = (P-1) * (fwd + bwd)
    bubble_ratio = bubble_time / pipeline_time
    throughput = micro_batches / pipeline_time  # batches per unit time

    return {
        "stages": stages,
        "micro_batches": micro_batches,
        "bubble_ratio": bubble_ratio,
        "pipeline_time": pipeline_time,
        "ideal_time": ideal_time,
        "throughput": throughput,
    }


if __name__ == "__main__":
    # Run the simulation across configurations
    print(f"{'Stages':>6} {'MBs':>5} {'Bubble%':>8} {'Pipeline T':>11} {'Ideal T':>9} "
          f"{'Throughput':>11}")
    print("-" * 58)

    for stages in [2, 4, 8]:
        for mbs in [4, 8, 16, 32, 64]:
            r = simulate_pipeline(stages, mbs)
            marker = " <-- ok" if r["bubble_ratio"] < 0.20 else ""
            print(f"{r['stages']:>6} {r['micro_batches']:>5} "
                  f"{r['bubble_ratio']:>7.1%} {r['pipeline_time']:>10.1f} "
                  f"{r['ideal_time']:>9.1f} {r['throughput']:>10.3f}{marker}")
        print()
