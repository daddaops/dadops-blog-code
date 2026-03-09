"""Random Hyperplane LSH for Approximate Nearest Neighbor Search.

Locality-Sensitive Hashing using random hyperplanes for cosine similarity.
Multi-table LSH increases recall by hashing into multiple independent tables.

Sweeps number of tables L to show recall-candidates tradeoff.
"""
import numpy as np

def random_hyperplane_lsh(data, n_tables=10, n_bits=8, seed=42):
    """Build multi-table LSH index for cosine similarity."""
    rng = np.random.RandomState(seed)
    n, d = data.shape
    norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-10
    normed = data / norms

    tables = []
    all_planes = []
    for t in range(n_tables):
        planes = rng.randn(n_bits, d)
        bits = (normed @ planes.T > 0).astype(np.uint8)
        # Pack bits into integer hash keys
        keys = np.packbits(bits, axis=1, bitorder='big')
        buckets = {}
        for i in range(n):
            k = keys[i].tobytes()
            buckets.setdefault(k, []).append(i)
        tables.append(buckets)
        all_planes.append(planes)

    def query(q, top_k=10):
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        candidates = set()
        for t in range(n_tables):
            bits = (q_norm @ all_planes[t].T > 0).astype(np.uint8)
            k = np.packbits(bits, bitorder='big').tobytes()
            candidates.update(tables[t].get(k, []))
        if not candidates:
            return [], 0
        cands = list(candidates)
        sims = normed[cands] @ q_norm
        top_idx = np.argsort(-sims)[:top_k]
        return [cands[i] for i in top_idx], len(cands)

    return query

if __name__ == "__main__":
    # Build index on 10,000 random 64-d vectors
    rng = np.random.RandomState(0)
    data = rng.randn(10000, 64)
    q = rng.randn(64)

    # Brute force ground truth
    norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-10
    sims = (data / norms) @ (q / (np.linalg.norm(q) + 1e-10))
    true_top10 = set(np.argsort(-sims)[:10])

    # Sweep number of tables
    for L in [1, 5, 10, 20]:
        search = random_hyperplane_lsh(data, n_tables=L, n_bits=8)
        result, n_cands = search(q, top_k=10)
        recall = len(set(result) & true_top10) / 10
        print(f"L={L:2d} | candidates: {n_cands:5d} | recall@10: {recall:.1f}")
