"""Sparse autoencoder for disentangling polysemantic activations.

Generates synthetic polysemantic activations (5 concepts in 3D),
trains an SAE to recover monosemantic features, and shows which
SAE features correspond to which ground-truth concepts.
"""
import numpy as np


def sae_interpretability_demo():
    """Train a sparse autoencoder to disentangle polysemantic activations."""
    np.random.seed(99)
    n_concepts = 5    # ground-truth concepts (sparse)
    d_model = 3       # model's hidden dim (bottleneck — forces superposition)
    d_sae = 12        # SAE latent dim (expanded — room for monosemantic features)
    n_samples = 2000

    # Ground-truth concept embeddings (randomly oriented in 3D)
    concept_dirs = np.random.randn(n_concepts, d_model)
    concept_dirs /= np.linalg.norm(concept_dirs, axis=1, keepdims=True)
    concept_names = ["code", "math", "music", "sports", "cooking"]

    # Generate polysemantic activations: sparse mixtures of concepts in 3D
    X = np.zeros((n_samples, d_model))
    labels = np.zeros((n_samples, n_concepts))
    for i in range(n_samples):
        active = np.random.rand(n_concepts) < 0.15  # each concept 15% active
        strengths = np.random.uniform(0.5, 2.0, n_concepts) * active
        X[i] = strengths @ concept_dirs
        labels[i] = active

    # Train sparse autoencoder
    W_enc = np.random.randn(d_model, d_sae) * 0.3
    W_dec = np.random.randn(d_sae, d_model) * 0.3
    b_enc = np.zeros(d_sae)
    lam = 0.05  # L1 sparsity penalty

    for step in range(1500):
        idx = np.random.choice(n_samples, 128)
        x = X[idx]

        # Forward
        h = np.maximum(0, x @ W_enc + b_enc)      # (128, d_sae) — sparse codes
        x_hat = h @ W_dec                           # (128, d_model) — reconstruction

        # Loss = MSE + L1
        recon_loss = np.mean((x_hat - x) ** 2)
        sparse_loss = lam * np.mean(np.abs(h))

        # Backward
        grad_xhat = 2 * (x_hat - x) / x.shape[0]
        grad_h = grad_xhat @ W_dec.T + lam * np.sign(h) / x.shape[0]
        grad_h = grad_h * (h > 0)  # ReLU derivative
        W_dec -= 0.02 * (h.T @ grad_xhat)
        W_enc -= 0.02 * (x.T @ grad_h)
        b_enc -= 0.02 * grad_h.sum(axis=0)

    # Interpret: for each SAE feature, find which concept it best matches
    print("SAE Feature Analysis:")
    print(f"{'Feature':<10} {'Best concept':<12} {'Correlation':<14} {'Avg L0'}")
    print("-" * 50)
    h_all = np.maximum(0, X @ W_enc + b_enc)
    active_features = np.where(h_all.mean(axis=0) > 0.01)[0]

    for feat_idx in active_features[:8]:  # show top 8 active features
        activations = h_all[:, feat_idx]
        # Correlate with each ground-truth concept
        best_corr, best_concept = 0, 0
        for c in range(n_concepts):
            corr = np.corrcoef(activations, labels[:, c])[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr, best_concept = corr, c
        avg_l0 = (h_all > 0).sum(axis=1).mean()
        print(f"f{feat_idx:<8} {concept_names[best_concept]:<12} {best_corr:<14.3f} {avg_l0:.1f}")


if __name__ == "__main__":
    sae_interpretability_demo()
