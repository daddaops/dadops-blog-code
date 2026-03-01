"""
Head-to-head profiler overhead benchmark.

Measures overhead of cProfile, py-spy, and Scalene on five workloads.
py-spy and scalene are optional — if not installed, they're skipped.

From: https://dadops.dev/blog/profiling-python-ai-code/
"""
import time
import subprocess
import shutil
import os

WORKLOADS = {
    "tokenization":  "benchmarks/tokenize_100k.py",
    "async_io":      "benchmarks/async_api_calls.py",
    "rag_pipeline":  "benchmarks/rag_mixed.py",
    "numpy_linalg":  "benchmarks/numpy_ops.py",
    "multiprocess":  "benchmarks/mp_pipeline.py",
}

def measure_baseline(script, runs=3):
    """Run script without any profiler, return median wall time."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        subprocess.run(["python3", script], capture_output=True)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]

def measure_with_profiler(script, profiler, runs=3):
    """Run script under a profiler, return median wall time."""
    commands = {
        "cprofile": ["python3", "-m", "cProfile", "-o", "/dev/null", script],
        "pyspy":    ["py-spy", "record", "-o", "/dev/null", "--", "python3", script],
        "scalene":  ["scalene", "--cli", "--off", script],
    }
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = subprocess.run(commands[profiler], capture_output=True)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return None  # profiler not available or failed
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]

if __name__ == "__main__":
    # Check which profilers are available
    available_profilers = ["cprofile"]
    if shutil.which("py-spy"):
        available_profilers.append("pyspy")
    else:
        print("py-spy not found — skipping (pip install py-spy)")
    if shutil.which("scalene"):
        available_profilers.append("scalene")
    else:
        print("scalene not found — skipping (pip install scalene)")

    print(f"\nRunning with profilers: {available_profilers}")
    print("=" * 70)

    results = {}
    for name, script in WORKLOADS.items():
        if not os.path.exists(script):
            print(f"  SKIP {name}: {script} not found")
            continue

        print(f"\n{name}:")
        baseline = measure_baseline(script)
        print(f"  baseline: {baseline:.3f}s")
        results[name] = {"baseline_s": round(baseline, 3)}

        for profiler in available_profilers:
            t = measure_with_profiler(script, profiler)
            if t is not None:
                overhead = round(t / baseline, 2)
                results[name][profiler] = overhead
                print(f"  {profiler}: {t:.3f}s ({overhead:.2f}x)")
            else:
                print(f"  {profiler}: FAILED")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Workload':<20}", end="")
    for p in available_profilers:
        print(f"  {p:>10}", end="")
    print()
    print("-" * 70)
    for name, data in results.items():
        print(f"{name:<20}", end="")
        for p in available_profilers:
            if p in data:
                print(f"  {data[p]:>9.2f}x", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()
