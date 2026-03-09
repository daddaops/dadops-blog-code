"""Product Quantization (PQ) for Memory-Efficient ANN Search.

Splits vectors into sub-spaces, clusters each independently, and replaces
sub-vectors with centroid indices. Asymmetric Distance Computation (ADC)
keeps the query exact while approximating database vectors.
"""
import numpy as np

def product_quantize(data, m=8, k_star=16, seed=42):
    """Product quantization: split, cluster, encode."""
    rng = np.random.RandomState(seed)
    n, d = data.shape
    sub_d = d // m

    codebooks = []   # m codebooks, each k_star x sub_d
    codes = np.zeros((n, m), dtype=np.uint8)

    for s in range(m):
        sub_data = data[:, s*sub_d:(s+1)*sub_d]
        # Mini k-means (20 iterations)
        centroids = sub_data[rng.choice(n, k_star, replace=False)]
        for _ in range(20):
            dists = np.sum((sub_data[:, None] - centroids[None]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            for c in range(k_star):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = sub_data[mask].mean(axis=0)
        # Encode
        dists = np.sum((sub_data[:, None] - centroids[None]) ** 2, axis=2)
        codes[:, s] = np.argmin(dists, axis=1)
        codebooks.append(centroids)

    return codebooks, codes

def adc_search(query, codebooks, codes, top_k=10):
    """Asymmetric distance computation: exact query vs PQ codes."""
    m = len(codebooks)
    sub_d = codebooks[0].shape[1]

    # Precompute distance tables: m tables of k_star distances
    dist_tables = []
    for s in range(m):
        q_sub = query[s*sub_d:(s+1)*sub_d]
        dists = np.sum((codebooks[s] - q_sub) ** 2, axis=1)
        dist_tables.append(dists)

    # Approximate distance = sum of m table lookups
    n = len(codes)
    approx_dists = np.zeros(n)
    for s in range(m):
        approx_dists += dist_tables[s][codes[:, s]]

    top_idx = np.argsort(approx_dists)[:top_k]
    return top_idx, approx_dists[top_idx]

if __name__ == "__main__":
    # Demo: 10,000 vectors, d=64, m=8 sub-spaces, k*=16 centroids
    rng = np.random.RandomState(0)
    data = rng.randn(10000, 64).astype(np.float32)
    query = rng.randn(64).astype(np.float32)

    codebooks, codes = product_quantize(data, m=8, k_star=16)
    pq_top10, pq_dists = adc_search(query, codebooks, codes, top_k=10)

    # Ground truth
    true_dists = np.sum((data - query) ** 2, axis=1)
    true_top10 = set(np.argsort(true_dists)[:10])

    recall = len(set(pq_top10) & true_top10) / 10
    orig_bytes = data.nbytes
    pq_bytes = codes.nbytes + sum(cb.nbytes for cb in codebooks)
    print(f"Original: {orig_bytes/1024:.0f} KB")
    print(f"PQ codes: {pq_bytes/1024:.1f} KB ({orig_bytes/pq_bytes:.0f}x compression)")
    print(f"Recall@10: {recall:.1f}")
