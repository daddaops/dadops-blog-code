"""
NumPy brute-force vector search demo.

Blog post: https://dadops.dev/blog/vector-search-benchmarks/
Code Blocks 1 & 5 from "Vector Search at Small Scale"

Demonstrates cosine similarity search using pure NumPy:
- Pre-normalize vectors to unit length
- Dot product = cosine similarity for unit vectors
- argpartition for efficient top-k selection
"""
import numpy as np


def search_numpy(query, database, k=10):
    """Brute-force cosine similarity search using NumPy.

    Args:
        query: (d,) unit vector
        database: (n, d) array of unit vectors
        k: number of nearest neighbors to return

    Returns:
        (k,) array of indices, sorted by descending similarity
    """
    scores = query @ database.T          # (n,) dot products
    top_k = np.argpartition(-scores, k)[:k]
    return top_k[np.argsort(-scores[top_k])]


if __name__ == "__main__":
    # Generate random unit vectors
    rng = np.random.default_rng(42)
    n, d = 10_000, 768

    print(f"Generating {n} random {d}-dimensional unit vectors...")
    database = rng.standard_normal((n, d)).astype("float32")
    database = database / np.linalg.norm(database, axis=1, keepdims=True)

    # Search for neighbors of first vector
    query = database[0]
    results = search_numpy(query, database, k=10)
    scores = query @ database[results].T

    print(f"Top 10 neighbors of vector 0: {results}")
    print(f"Scores: {scores}")
    print(f"Self-similarity (should be ~1.0): {scores[0]:.6f}")

    # Quick timing
    import time
    times = []
    for _ in range(100):
        q = database[rng.integers(n)]
        t0 = time.perf_counter()
        search_numpy(q, database, k=10)
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    print(f"\nLatency over 100 queries ({n} vectors, {d}d):")
    print(f"  p50: {np.percentile(times_ms, 50):.3f} ms")
    print(f"  p95: {np.percentile(times_ms, 95):.3f} ms")
    print(f"  p99: {np.percentile(times_ms, 99):.3f} ms")
