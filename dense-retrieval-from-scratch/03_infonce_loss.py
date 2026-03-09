import numpy as np

def infonce_loss(q_vecs, d_vecs, temperature=0.05):
    """
    InfoNCE loss with in-batch negatives.
    q_vecs: (B, dim) query embeddings
    d_vecs: (B, dim) positive document embeddings
    Each q_vecs[i] pairs with d_vecs[i] (positive).
    All other d_vecs[j!=i] are negatives for query i.
    """
    # All-pairs similarity matrix: (B, B)
    sim_matrix = q_vecs @ d_vecs.T / temperature

    # Labels: diagonal entries are the positives
    labels = np.arange(len(q_vecs))

    # Numerically stable log-sum-exp (subtract row max to prevent overflow)
    row_max = np.max(sim_matrix, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(sim_matrix - row_max), axis=1)) + row_max.squeeze()
    positive_scores = sim_matrix[labels, labels]
    loss = -np.mean(positive_scores - log_sum_exp)
    return loss

# Gradient of InfoNCE pushes positive pairs closer,
# negative pairs apart. The similarity matrix encodes
# B^2 comparisons from just B data pairs.

# Example: batch of 4 query-document pairs
rng = np.random.RandomState(7)
q = rng.randn(4, 64)
d = rng.randn(4, 64)
q = q / np.linalg.norm(q, axis=1, keepdims=True)
d = d / np.linalg.norm(d, axis=1, keepdims=True)

print(f"Loss (random embeddings): {infonce_loss(q, d):.3f}")
# Loss (random embeddings): 3.712
# High because tau=0.05 amplifies random similarities.
# Perfect alignment would push loss toward 0.
