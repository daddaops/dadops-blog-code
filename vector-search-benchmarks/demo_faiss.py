"""
FAISS vector search demo — Flat, IVF, and HNSW indices.

Blog post: https://dadops.dev/blog/vector-search-benchmarks/
Code Blocks 2 & 6 from "Vector Search at Small Scale"

Demonstrates three FAISS index types:
- IndexFlatIP: exact brute-force inner product
- IndexIVFFlat: inverted file index with Voronoi partitioning
- IndexHNSWFlat: hierarchical navigable small world graph
"""
import faiss
import numpy as np
import time


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n, d = 10_000, 768

    print(f"Generating {n} random {d}-dimensional unit vectors...")
    database = rng.standard_normal((n, d)).astype("float32")
    faiss.normalize_L2(database)  # in-place normalization

    # --- IndexFlatIP (exact brute force) ---
    print("\n--- IndexFlatIP (exact brute force) ---")
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(database)
    print(f"Index size: {index_flat.ntotal} vectors")

    query = database[:1]  # shape (1, d)
    D, I = index_flat.search(query, k=10)
    print(f"Top 10 neighbors: {I[0]}")
    print(f"Scores: {D[0]}")

    # --- IndexIVFFlat ---
    print("\n--- IndexIVFFlat (nlist=100, nprobe=10) ---")
    nlist = 100
    quantizer = faiss.IndexFlatIP(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(database)
    index_ivf.add(database)
    index_ivf.nprobe = 10
    print(f"Index size: {index_ivf.ntotal} vectors")

    D, I = index_ivf.search(query, k=10)
    print(f"Top 10 neighbors: {I[0]}")
    print(f"Scores: {D[0]}")

    # --- IndexHNSWFlat ---
    print("\n--- IndexHNSWFlat (M=32, efConstruction=64, efSearch=40) ---")
    index_hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = 64
    index_hnsw.add(database)
    index_hnsw.hnsw.efSearch = 40
    print(f"Index size: {index_hnsw.ntotal} vectors")

    D, I = index_hnsw.search(query, k=10)
    print(f"Top 10 neighbors: {I[0]}")
    print(f"Scores: {D[0]}")

    # --- Quick latency comparison ---
    print("\n--- Latency comparison (100 queries, 10K vectors, 768d) ---")
    queries = rng.standard_normal((100, d)).astype("float32")
    faiss.normalize_L2(queries)

    for name, idx in [("Flat", index_flat), ("IVF", index_ivf), ("HNSW", index_hnsw)]:
        times = []
        for i in range(100):
            q = queries[i:i+1]
            t0 = time.perf_counter()
            idx.search(q, k=10)
            times.append(time.perf_counter() - t0)
        times_ms = np.array(times) * 1000
        print(f"  FAISS {name:>5}: p50={np.percentile(times_ms, 50):.3f} ms, "
              f"p95={np.percentile(times_ms, 95):.3f} ms, "
              f"p99={np.percentile(times_ms, 99):.3f} ms")
