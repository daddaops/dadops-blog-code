import numpy as np

class SimCLREncoder:
    """Encoder + projection head for contrastive learning."""

    def __init__(self, input_dim, hidden_dim=64, proj_dim=32):
        # Encoder: input -> hidden representation
        scale = np.sqrt(2.0 / input_dim)  # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)

        # Projection head: hidden -> projection space
        scale = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, proj_dim) * scale
        self.b2 = np.zeros(proj_dim)

    def encode(self, x):
        """Get representation h (what we KEEP after training)."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h

    def project(self, h):
        """Map to projection space z (used ONLY during training)."""
        z = h @ self.W2 + self.b2
        # L2 normalize — critical for cosine similarity
        z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        return z

    def forward(self, x):
        """Full forward: input -> representation -> projection."""
        h = self.encode(x)
        z = self.project(h)
        return h, z


if __name__ == "__main__":
    np.random.seed(42)

    # Create a small batch of 2D data
    x = np.random.randn(8, 2)
    encoder = SimCLREncoder(input_dim=2, hidden_dim=64, proj_dim=32)

    h, z = encoder.forward(x)
    print(f"Input shape:          {x.shape}")
    print(f"Representation shape: {h.shape}")
    print(f"Projection shape:     {z.shape}")

    # Verify L2 normalization of projections
    norms = np.linalg.norm(z, axis=1)
    print(f"Projection norms:     {norms.round(4)}")
    print("All norms ≈ 1.0:", np.allclose(norms, 1.0, atol=1e-6))
