"""Sparse Autoencoder class: overcomplete autoencoder with sparsity constraint.

From: https://dadops.substack.com/sparse-autoencoders-from-scratch
"""
import numpy as np


class SparseAutoencoder:
    """Overcomplete autoencoder with sparsity constraint."""

    def __init__(self, d_model, d_sae):
        self.d_model = d_model
        self.d_sae = d_sae
        # He initialization for ReLU
        scale_enc = np.sqrt(2.0 / d_model)
        scale_dec = np.sqrt(2.0 / d_sae)
        self.W_enc = np.random.randn(d_sae, d_model) * scale_enc
        self.b_enc = np.zeros(d_sae)
        self.W_dec = np.random.randn(d_model, d_sae) * scale_dec
        self.b_dec = np.zeros(d_model)
        # Normalize decoder columns to unit norm
        self.normalize_decoder()

    def normalize_decoder(self):
        """Constrain each decoder column to unit L2 norm."""
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        self.W_dec /= (norms + 1e-8)

    def encode(self, x):
        """Encode with pre-centering and ReLU activation."""
        x_centered = x - self.b_dec               # pre-center
        z_pre = x_centered @ self.W_enc.T + self.b_enc
        z = np.maximum(0, z_pre)                   # ReLU: creates true zeros
        return z, z_pre

    def decode(self, z):
        """Reconstruct from sparse features."""
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x):
        """Full forward pass: encode then decode."""
        z, z_pre = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, z_pre


if __name__ == "__main__":
    np.random.seed(42)
    d_model, d_sae = 16, 64
    sae = SparseAutoencoder(d_model, d_sae)

    # Test with random input
    x = np.random.randn(10, d_model)
    x_hat, z, z_pre = sae.forward(x)

    print(f"Input shape:  {x.shape}")
    print(f"Hidden shape: {z.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Expansion ratio: {d_sae / d_model:.0f}x")
    print(f"Avg nonzero features per input: {np.mean(np.sum(z > 0, axis=1)):.1f}")
    print(f"Decoder column norms (should be ~1.0): {np.linalg.norm(sae.W_dec, axis=0)[:5].round(4)}")
