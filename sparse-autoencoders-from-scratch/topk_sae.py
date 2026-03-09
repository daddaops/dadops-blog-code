"""Top-K sparse autoencoder variant: sparsity by selection instead of penalization.

From: https://dadops.substack.com/sparse-autoencoders-from-scratch
"""
import numpy as np


def topk_encode(x, W_enc, b_enc, b_dec, k):
    """TopK SAE encoder: keep k largest activations, zero the rest."""
    x_centered = x - b_dec
    z_pre = x_centered @ W_enc.T + b_enc
    z_pre = np.maximum(0, z_pre)  # ReLU first

    # Keep only top-k activations per sample
    z = np.zeros_like(z_pre)
    for i in range(len(z_pre)):
        if np.sum(z_pre[i] > 0) <= k:
            z[i] = z_pre[i]  # fewer than k active: keep all
        else:
            topk_idx = np.argpartition(z_pre[i], -k)[-k:]
            z[i, topk_idx] = z_pre[i, topk_idx]
    return z


# With TopK, the loss is pure reconstruction — no L1 term!
# L_topk = ||x - x_hat||^2
# Sparsity is exact: always <= k active features per input.
# No shrinkage: active feature magnitudes are preserved exactly.

# Compare activation distributions:
# L1-SAE:  many small values clustered near zero (shrinkage)
# TopK-SAE: clean bimodal distribution (zero or full magnitude)


if __name__ == "__main__":
    np.random.seed(42)

    # Demo: compare L1 vs TopK sparsity patterns
    d_model, d_sae = 16, 64
    n_samples = 100
    k = 5

    # Random SAE weights
    scale_enc = np.sqrt(2.0 / d_model)
    scale_dec = np.sqrt(2.0 / d_sae)
    W_enc = np.random.randn(d_sae, d_model) * scale_enc
    b_enc = np.zeros(d_sae)
    W_dec = np.random.randn(d_model, d_sae) * scale_dec
    b_dec = np.zeros(d_model)

    # Random input
    x = np.random.randn(n_samples, d_model)

    # TopK encoding
    z_topk = topk_encode(x, W_enc, b_enc, b_dec, k)

    # Standard ReLU encoding (no sparsity constraint)
    z_pre = (x - b_dec) @ W_enc.T + b_enc
    z_relu = np.maximum(0, z_pre)

    print(f"TopK (k={k}):")
    print(f"  Avg nonzero features per input: {np.mean(np.sum(z_topk > 0, axis=1)):.1f}")
    print(f"  Max nonzero features per input: {np.max(np.sum(z_topk > 0, axis=1))}")
    print(f"  Mean activation (nonzero only): {z_topk[z_topk > 0].mean():.4f}")

    print(f"\nStandard ReLU (no sparsity):")
    print(f"  Avg nonzero features per input: {np.mean(np.sum(z_relu > 0, axis=1)):.1f}")
    print(f"  Max nonzero features per input: {np.max(np.sum(z_relu > 0, axis=1))}")
    print(f"  Mean activation (nonzero only): {z_relu[z_relu > 0].mean():.4f}")

    # Reconstruction comparison
    x_hat_topk = z_topk @ W_dec.T + b_dec
    x_hat_relu = z_relu @ W_dec.T + b_dec
    print(f"\nReconstruction MSE (untrained, for comparison):")
    print(f"  TopK: {np.mean((x - x_hat_topk) ** 2):.4f}")
    print(f"  ReLU: {np.mean((x - x_hat_relu) ** 2):.4f}")
