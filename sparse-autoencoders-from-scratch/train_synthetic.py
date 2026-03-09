"""Training SAE on synthetic data with known ground-truth features.

From: https://dadops.substack.com/sparse-autoencoders-from-scratch
"""
import numpy as np


class SparseAutoencoder:
    """Overcomplete autoencoder with sparsity constraint."""

    def __init__(self, d_model, d_sae):
        self.d_model = d_model
        self.d_sae = d_sae
        scale_enc = np.sqrt(2.0 / d_model)
        scale_dec = np.sqrt(2.0 / d_sae)
        self.W_enc = np.random.randn(d_sae, d_model) * scale_enc
        self.b_enc = np.zeros(d_sae)
        self.W_dec = np.random.randn(d_model, d_sae) * scale_dec
        self.b_dec = np.zeros(d_model)
        self.normalize_decoder()

    def normalize_decoder(self):
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        self.W_dec /= (norms + 1e-8)

    def encode(self, x):
        x_centered = x - self.b_dec
        z_pre = x_centered @ self.W_enc.T + self.b_enc
        z = np.maximum(0, z_pre)
        return z, z_pre

    def decode(self, z):
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x):
        z, z_pre = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, z_pre


def sae_loss(x, x_hat, z, lambda_l1):
    recon_loss = np.mean((x - x_hat) ** 2)
    l1_loss = np.mean(np.sum(np.abs(z), axis=1))
    total_loss = recon_loss + lambda_l1 * l1_loss
    l0 = np.mean(np.sum(z > 0, axis=1))
    return total_loss, recon_loss, l1_loss, l0


def l1_gradient(z):
    return np.sign(z)


def train_sae_synthetic():
    """Train SAE on synthetic data with known ground-truth features."""
    np.random.seed(0)
    d_model, d_sae = 4, 32
    n_true_features = 8

    # Ground-truth features: 8 random unit vectors in 4D
    true_features = np.random.randn(n_true_features, d_model)
    true_features /= np.linalg.norm(true_features, axis=1, keepdims=True)

    # Generate sparse data: each sample uses 1-3 features
    n_samples = 5000
    X = np.zeros((n_samples, d_model))
    for i in range(n_samples):
        k = np.random.randint(1, 4)  # 1-3 active features
        active = np.random.choice(n_true_features, k, replace=False)
        coeffs = np.random.uniform(0.5, 2.0, k)
        for j, a in enumerate(active):
            X[i] += coeffs[j] * true_features[a]

    # Initialize SAE
    sae = SparseAutoencoder(d_model, d_sae)
    lambda_l1 = 1.0
    lr = 0.003

    # Adam optimizer state
    m = {k: np.zeros_like(v) for k, v in
         [('W_enc', sae.W_enc), ('b_enc', sae.b_enc),
          ('W_dec', sae.W_dec), ('b_dec', sae.b_dec)]}
    v = {k: np.zeros_like(v) for k, v in
         [('W_enc', sae.W_enc), ('b_enc', sae.b_enc),
          ('W_dec', sae.W_dec), ('b_dec', sae.b_dec)]}

    for epoch in range(300):
        # Forward pass
        x_hat, z, z_pre = sae.forward(X)
        total, recon, l1, l0 = sae_loss(X, x_hat, z, lambda_l1)

        # Backward pass: gradients through decoder and encoder
        d_xhat = 2 * (x_hat - X) / X.shape[0]   # d(recon)/d(x_hat)
        # Gradient through decoder: x_hat = z @ W_dec.T + b_dec
        d_z = d_xhat @ sae.W_dec + lambda_l1 * l1_gradient(z) / X.shape[0]
        d_z_pre = d_z * (z_pre > 0)  # ReLU mask
        d_Wdec = d_xhat.T @ z                     # (d_model, d_sae)
        # Gradient through encoder: z_pre = (X - b_dec) @ W_enc.T + b_enc
        d_Wenc = d_z_pre.T @ (X - sae.b_dec)      # (d_sae, d_model)
        d_benc = np.sum(d_z_pre, axis=0)           # (d_sae,)
        # b_dec appears in both encoder (pre-centering) and decoder
        d_bdec = d_xhat.sum(axis=0) - (d_z_pre @ sae.W_enc).sum(axis=0)

        # Adam updates (simplified: beta1=0.9, beta2=0.999, eps=1e-8)
        grads = {'W_enc': d_Wenc, 'b_enc': d_benc,
                 'W_dec': d_Wdec, 'b_dec': d_bdec}
        t = epoch + 1
        for key in grads:
            m[key] = 0.9 * m[key] + 0.1 * grads[key]
            v[key] = 0.999 * v[key] + 0.001 * grads[key] ** 2
            m_hat = m[key] / (1 - 0.9 ** t)
            v_hat = v[key] / (1 - 0.999 ** t)
            param = getattr(sae, key)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        sae.normalize_decoder()  # unit norm constraint on decoder columns

        if epoch % 50 == 0:
            dead = np.mean(np.all(z == 0, axis=0))
            print(f"Epoch {epoch:3d} | Recon: {recon:.4f} | "
                  f"L0: {l0:.1f} | Dead: {dead:.0%}")

    # Verify feature recovery: match SAE features to ground truth
    alive_mask = np.any(z > 0, axis=0)
    alive_cols = sae.W_dec[:, alive_mask].T  # (n_alive, d_model)
    alive_cols /= np.linalg.norm(alive_cols, axis=1, keepdims=True)
    similarity = np.abs(alive_cols @ true_features.T)  # cosine similarity
    best_match = np.max(similarity, axis=0)
    print(f"\nFeature recovery (cosine similarity to ground truth):")
    for i in range(n_true_features):
        print(f"  True feature {i}: best match = {best_match[i]:.3f}")


if __name__ == "__main__":
    train_sae_synthetic()
