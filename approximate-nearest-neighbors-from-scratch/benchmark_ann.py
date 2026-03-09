"""ANN Benchmark: Recall@k vs QPS for NSW with varying ef_search.

Sweeps the search effort parameter to show the fundamental
recall-latency tradeoff in approximate nearest neighbor search.
"""
import numpy as np
import time
from nsw import build_nsw, search_nsw

def benchmark_ann(data, queries, true_nn, search_fn, params, top_k=10):
    """Benchmark ANN: sweep a parameter, report recall@k vs QPS."""
    results = []
    for param in params:
        recalls = []
        t0 = time.time()
        for i in range(len(queries)):
            pred = search_fn(queries[i], param, top_k)
            recalls.append(len(set(pred) & set(true_nn[i])) / top_k)
        elapsed = time.time() - t0
        qps = len(queries) / elapsed
        avg_recall = np.mean(recalls)
        results.append((param, avg_recall, qps))
        print(f"  param={param:4d} | recall@{top_k}: {avg_recall:.3f} | "
              f"QPS: {qps:.0f}")
    return results

if __name__ == "__main__":
    # Build NSW graph
    rng = np.random.RandomState(0)
    n, d, n_queries = 5000, 32, 100
    data = rng.randn(n, d).astype(np.float32)
    queries = rng.randn(n_queries, d).astype(np.float32)

    # Ground truth top-10 for each query
    true_nn = []
    for q in queries:
        dists = np.sum((data - q) ** 2, axis=1)
        true_nn.append(list(np.argsort(dists)[:10]))

    print("Building NSW graph...")
    graph, entry = build_nsw(data, M=5, ef=20)

    # Wrap search for benchmark
    def search_fn(query, ef_param, top_k):
        return search_nsw(data, graph, entry, query, top_k=top_k, ef=ef_param)

    print("\nBenchmark: NSW with varying ef_search")
    benchmark_ann(data, queries, true_nn, search_fn,
                  params=[5, 10, 20, 50, 100])
