"""NT-Xent contrastive loss (SimCLR)."""
import numpy as np

def nt_xent_loss(z, temperature=0.5):
    """NT-Xent: Normalized Temperature-scaled Cross-Entropy Loss.
    z: (2N, dim) where z[2k] and z[2k+1] are a positive pair."""
    N = len(z) // 2

    # L2 normalize embeddings onto the unit hypersphere
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    # Full cosine similarity matrix, scaled by temperature
    sim = z_norm @ z_norm.T / temperature  # (2N, 2N)

    # Numerical stability
    sim -= np.max(sim, axis=1, keepdims=True)
    exp_sim = np.exp(sim)

    # Mask out self-similarity (diagonal)
    mask = ~np.eye(2 * N, dtype=bool)

    loss = 0.0
    for i in range(2 * N):
        # Positive partner: (2k, 2k+1) are paired
        j = i + 1 if i % 2 == 0 else i - 1
        numerator = exp_sim[i, j]
        denominator = (exp_sim[i] * mask[i]).sum()
        loss -= np.log(numerator / denominator + 1e-10)

    return loss / (2 * N)

# Create a batch: 4 images, each with 2 augmented views
np.random.seed(42)
dim = 32
batch = []
for _ in range(4):
    anchor = np.random.randn(dim)
    positive = anchor + np.random.randn(dim) * 0.1  # similar view
    batch.extend([anchor, positive])

z = np.array(batch)

# How does temperature affect the loss?
for temp in [0.1, 0.5, 1.0, 2.0]:
    loss = nt_xent_loss(z, temperature=temp)
    print(f"Temperature {temp:.1f}: loss = {loss:.4f}")
