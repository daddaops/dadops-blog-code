import numpy as np

class BiEncoder:
    """Minimal bi-encoder with random projection (conceptual demo)."""
    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=128):
        rng = np.random.RandomState(42)
        self.embed = rng.randn(vocab_size, embed_dim) * 0.02
        self.W1 = rng.randn(embed_dim, hidden_dim) * 0.02
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def tokenize(self, text):
        return [hash(w) % 5000 for w in text.lower().split()]

    def encode(self, text):
        tokens = self.tokenize(text)
        token_vecs = self.embed[tokens]           # (T, embed_dim)
        pooled = token_vecs.mean(axis=0)          # mean pooling
        h = np.maximum(0, pooled @ self.W1 + self.b1)   # ReLU
        out = h @ self.W2 + self.b2
        return out / (np.linalg.norm(out) + 1e-8)       # L2 normalize

    def score(self, query, doc):
        q_vec = self.encode(query)
        d_vec = self.encode(doc)
        return float(q_vec @ d_vec)               # dot product

    def search(self, query, docs, top_k=3):
        q_vec = self.encode(query)
        d_vecs = np.array([self.encode(d) for d in docs])
        scores = d_vecs @ q_vec                   # batch dot product
        top_idx = np.argsort(-scores)[:top_k]
        return [(i, scores[i]) for i in top_idx]

# Without training, the bi-encoder produces near-random rankings.
# The magic happens when we train it with contrastive learning...
