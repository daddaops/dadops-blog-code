import numpy as np

class ProductQuantizer:
    def __init__(self, dim=64, n_subvectors=8, n_centroids=256):
        self.m = n_subvectors
        self.k = n_centroids
        self.sub_dim = dim // n_subvectors
        self.codebooks = None       # (m, k, sub_dim)
        self.codes = None           # (N, m) -- encoded corpus

    def train(self, vectors, n_iter=20):
        """Learn codebooks via k-means on each sub-vector slice."""
        rng = np.random.RandomState(42)
        N, D = vectors.shape
        self.codebooks = np.zeros((self.m, self.k, self.sub_dim))
        for i in range(self.m):
            sub = vectors[:, i*self.sub_dim:(i+1)*self.sub_dim]
            # Mini k-means
            centroids = sub[rng.choice(N, self.k, replace=False)]
            for _ in range(n_iter):
                dists = np.linalg.norm(sub[:, None] - centroids[None], axis=2)
                assignments = dists.argmin(axis=1)
                for c in range(self.k):
                    mask = assignments == c
                    if mask.any():
                        centroids[c] = sub[mask].mean(axis=0)
            self.codebooks[i] = centroids

    def encode(self, vectors):
        """Encode vectors as sequences of centroid indices."""
        N = len(vectors)
        self.codes = np.zeros((N, self.m), dtype=np.int32)
        for i in range(self.m):
            sub = vectors[:, i*self.sub_dim:(i+1)*self.sub_dim]
            dists = np.linalg.norm(sub[:, None] - self.codebooks[i][None], axis=2)
            self.codes[:, i] = dists.argmin(axis=1)

    def search(self, query, top_k=5):
        """Asymmetric distance: exact query vs quantized documents."""
        # Precompute distance table: query sub-vector to each centroid
        dist_table = np.zeros((self.m, self.k))
        for i in range(self.m):
            q_sub = query[i*self.sub_dim:(i+1)*self.sub_dim]
            dist_table[i] = np.linalg.norm(q_sub - self.codebooks[i], axis=1)
        # Sum lookup distances for each document
        approx_dists = np.zeros(len(self.codes))
        for i in range(self.m):
            approx_dists += dist_table[i, self.codes[:, i]]
        return np.argsort(approx_dists)[:top_k]

# In production: 768-dim float32 vectors (3072 bytes) compress to
# ~96 bytes with m=96 sub-vectors and 256 centroids -- 32x savings.
# Our demo uses dim=64 and m=8 to keep the code short.
# The distance table trick avoids decompressing documents entirely.

# Demo: generate random corpus vectors, quantize, and search
rng = np.random.RandomState(42)
corpus_vecs = rng.randn(1000, 64).astype(np.float32)
query_vec = rng.randn(64).astype(np.float32)

pq = ProductQuantizer(dim=64, n_subvectors=8, n_centroids=256)
pq.train(corpus_vecs)
pq.encode(corpus_vecs)

# Exact nearest neighbors for comparison
exact_dists = np.linalg.norm(corpus_vecs - query_vec, axis=1)
exact_top5 = np.argsort(exact_dists)[:5]

# Approximate nearest neighbors via PQ
approx_top5 = pq.search(query_vec, top_k=5)

print("Exact top-5 indices: ", exact_top5)
print("PQ approx top-5 indices:", approx_top5)
overlap = len(set(exact_top5) & set(approx_top5))
print(f"Overlap: {overlap}/5")
