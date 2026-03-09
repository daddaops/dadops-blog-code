import numpy as np


def info_nce_loss(z, temperature=0.5):
    """
    Compute NT-Xent (InfoNCE) loss for a batch of 2N embeddings.
    z: (2N, D) array — z[2i] and z[2i+1] are positive pairs.
    Returns: scalar loss value.
    """
    batch_size = z.shape[0]  # This is 2N
    N = batch_size // 2

    # Cosine similarity matrix (z is already L2-normalized)
    sim = z @ z.T  # (2N, 2N)

    # Apply temperature scaling
    sim = sim / temperature

    # Mask out self-similarity (diagonal) with large negative
    mask = np.eye(batch_size, dtype=bool)
    sim[mask] = -1e9

    # Compute loss for each of the 2N samples
    total_loss = 0.0
    for i in range(batch_size):
        # Positive partner: i=0 pairs with i=1, i=2 pairs with i=3, etc.
        pos_idx = i + 1 if i % 2 == 0 else i - 1

        # Log-sum-exp trick for numerical stability
        logits = sim[i]
        max_logit = logits.max()
        log_sum_exp = max_logit + np.log(
            np.sum(np.exp(logits - max_logit)) + 1e-8
        )

        # Loss for this anchor: -log(softmax probability of positive)
        total_loss += -(logits[pos_idx] - log_sum_exp)

    return total_loss / batch_size


if __name__ == "__main__":
    np.random.seed(42)

    # Create fake L2-normalized embeddings: 4 pairs (8 samples) in 4D
    z_raw = np.random.randn(8, 4)
    z = z_raw / (np.linalg.norm(z_raw, axis=1, keepdims=True) + 1e-8)

    loss = info_nce_loss(z, temperature=0.5)
    print(f"InfoNCE loss (random embeddings): {loss:.4f}")

    # Make positive pairs identical — loss should be lower
    z_perfect = z.copy()
    z_perfect[1] = z_perfect[0]
    z_perfect[3] = z_perfect[2]
    z_perfect[5] = z_perfect[4]
    z_perfect[7] = z_perfect[6]

    loss_perfect = info_nce_loss(z_perfect, temperature=0.5)
    print(f"InfoNCE loss (identical pairs):   {loss_perfect:.4f}")
    print(f"Loss decreased: {loss_perfect < loss}")
