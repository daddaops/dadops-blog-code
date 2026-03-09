"""Toy model of superposition: compress n sparse features into a bottleneck.

Demonstrates that sparse features can be packed into fewer dimensions
via superposition, while dense features can only preserve the top-k
(PCA-like behavior).
"""
import numpy as np


def train_superposition_model(n_features=8, bottleneck=2, sparsity=0.05,
                              steps=2000, lr=0.01):
    """Toy model of superposition: compress n sparse features into a bottleneck."""
    # Importance weights — first features matter more
    importance = np.array([0.7 ** i for i in range(n_features)])

    # Initialize encoder (n_features -> bottleneck) and decoder (bottleneck -> n_features)
    W_enc = np.random.randn(bottleneck, n_features) * 0.5
    W_dec = np.random.randn(n_features, bottleneck) * 0.5
    b_dec = np.zeros(n_features)

    for step in range(steps):
        # Generate sparse input: each feature active with probability = sparsity
        x = np.random.uniform(0, 1, (64, n_features))
        mask = (np.random.rand(64, n_features) < sparsity).astype(float)
        x = x * mask  # sparse activations

        # Forward: encode -> decode -> ReLU
        h = x @ W_enc.T                          # (64, bottleneck)
        x_hat = np.maximum(0, h @ W_dec.T + b_dec)  # (64, n_features)

        # Importance-weighted MSE loss
        diff = (x_hat - x) * importance
        loss = np.mean(diff ** 2)

        # Backward (manual gradients)
        grad_out = 2 * diff * importance / x.shape[0]
        grad_out = grad_out * (x_hat > 0)  # ReLU derivative
        W_dec -= lr * (grad_out.T @ h)
        b_dec -= lr * grad_out.sum(axis=0)
        W_enc -= lr * (grad_out @ W_dec).T @ x

    # Extract learned feature directions (columns of W_enc)
    feature_dirs = W_enc.T  # (n_features, bottleneck)
    norms = np.linalg.norm(feature_dirs, axis=1, keepdims=True) + 1e-8
    feature_dirs_normed = feature_dirs / norms

    # Compute interference matrix: |cos similarity| between all feature pairs
    cos_sim = np.abs(feature_dirs_normed @ feature_dirs_normed.T)
    np.fill_diagonal(cos_sim, 0)

    print(f"Features: {n_features}, Bottleneck: {bottleneck}, Sparsity: {sparsity}")
    print(f"Final loss: {loss:.4f}")
    print(f"Mean interference between features: {cos_sim.mean():.3f}")
    print(f"Max interference: {cos_sim.max():.3f}")
    return W_enc, cos_sim


if __name__ == "__main__":
    # Dense features: only top-2 features survive (PCA-like)
    print("=== Dense features (sparsity=0.9) ===")
    _, sim_dense = train_superposition_model(sparsity=0.9)

    # Sparse features: ALL 8 features packed into 2D (superposition!)
    print("\n=== Sparse features (sparsity=0.05) ===")
    _, sim_sparse = train_superposition_model(sparsity=0.05)
