import numpy as np
import time
from lance_williams import hierarchical_cluster

def benchmark_clustering(max_n=500, step=100):
    """Time hierarchical clustering as n grows."""
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    results = []
    for n in range(step, max_n + 1, step):
        X = np.random.randn(n, 2)
        # Our implementation
        t0 = time.time()
        hierarchical_cluster(X, "ward")
        t_ours = time.time() - t0
        # Scipy's optimized version
        t0 = time.time()
        scipy_linkage(X, method="ward")
        t_scipy = time.time() - t0
        results.append((n, t_ours, t_scipy))
        print(f"n={n:4d}  ours={t_ours:.3f}s  scipy={t_scipy:.4f}s  "
              f"ratio={t_ours / max(t_scipy, 1e-6):.0f}x")
    return results

if __name__ == "__main__":
    # Uncomment to run:
    # benchmark_clustering()
    print("Benchmark script ready (uncomment to run)")
