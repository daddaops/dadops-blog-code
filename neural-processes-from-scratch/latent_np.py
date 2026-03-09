"""Latent Neural Process with reparameterization trick.

Adds a stochastic latent variable z to enable coherent
function sampling (different z -> different but consistent predictions).
"""
import numpy as np
from cnp import make_sine_task


class LatentNP:
    def __init__(self, h_dim=64, z_dim=16):
        self.h_dim, self.z_dim = h_dim, z_dim
        # Shared encoder: (x, y) -> hidden
        self.enc_W1 = np.random.randn(2, h_dim) * 0.1
        self.enc_b1 = np.zeros(h_dim)
        # Deterministic path: hidden -> r
        self.det_W = np.random.randn(h_dim, h_dim) * 0.1
        self.det_b = np.zeros(h_dim)
        # Latent path: hidden -> (mu_z, log_sigma_z)
        self.lat_W = np.random.randn(h_dim, z_dim * 2) * 0.1
        self.lat_b = np.zeros(z_dim * 2)
        # Decoder: (r, z, x*) -> (mu, log_sigma)
        self.dec_W1 = np.random.randn(h_dim + z_dim + 1, h_dim) * 0.1
        self.dec_b1 = np.zeros(h_dim)
        self.dec_W2 = np.random.randn(h_dim, 2) * 0.1
        self.dec_b2 = np.zeros(2)

    def encode_context(self, x_ctx, y_ctx):
        pairs = np.column_stack([x_ctx, y_ctx])
        h = np.maximum(0, pairs @ self.enc_W1 + self.enc_b1)
        # Deterministic representation
        r = np.maximum(0, h @ self.det_W + self.det_b).mean(axis=0)
        # Latent distribution parameters
        lat = (h @ self.lat_W + self.lat_b).mean(axis=0)
        mu_z = lat[:self.z_dim]
        sigma_z = np.exp(lat[self.z_dim:]) + 1e-4
        return r, mu_z, sigma_z

    def sample_z(self, mu_z, sigma_z, rng):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        eps = rng.standard_normal(self.z_dim)
        return mu_z + sigma_z * eps

    def decode(self, r, z, x_target):
        n = len(x_target)
        inp = np.column_stack([np.tile(r, (n, 1)),
                               np.tile(z, (n, 1)), x_target])
        h = np.maximum(0, inp @ self.dec_W1 + self.dec_b1)
        out = h @ self.dec_W2 + self.dec_b2
        return out[:, 0], np.exp(out[:, 1]) + 1e-4

    def predict_samples(self, x_ctx, y_ctx, x_target, n_samples=5):
        """Draw multiple coherent function samples."""
        r, mu_z, sigma_z = self.encode_context(x_ctx, y_ctx)
        rng = np.random.default_rng(0)
        samples = []
        for _ in range(n_samples):
            z = self.sample_z(mu_z, sigma_z, rng)
            mu, _ = self.decode(r, z, x_target)
            samples.append(mu)
        return np.array(samples)  # (n_samples, n_targets)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_all, y_all = make_sine_task(rng)
    x_ctx, y_ctx = x_all[:5], y_all[:5]
    x_tgt = x_all[5:]

    lnp = LatentNP()
    samples = lnp.predict_samples(x_ctx, y_ctx, x_tgt, n_samples=5)
    print(f"5 function samples, each at {samples.shape[1]} targets")
    print(f"Sample std across functions: {samples.std(axis=0).mean():.3f}")
