"""Variational Autoencoder from Scratch — with reparameterization trick.

Implements a VAE with manual backpropagation, BCE + KL divergence loss,
and gradient clipping. Trained with SGD + momentum on 8×8 digit images.

From: https://www.dadops.co/blog/autoencoders-from-scratch/
"""
import numpy as np
from sklearn.datasets import load_digits


class VAE:
    """Variational Autoencoder: 64 → 32 → (μ, logvar) → z → 32 → 64."""
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=2):
        scale = np.sqrt(2.0 / input_dim)
        # Encoder: shared hidden layer, then two heads
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b_enc1 = np.zeros(hidden_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W_mu    = np.random.randn(hidden_dim, latent_dim) * s2
        self.b_mu    = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * s2
        self.b_logvar = np.zeros(latent_dim)
        # Decoder (identical to vanilla AE)
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * s2
        self.b_dec2 = np.zeros(input_dim)

    def encode(self, x):
        """x → hidden → (μ, log_var)."""
        h = np.maximum(0, x @ self.W_enc1 + self.b_enc1)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var, h

    def reparameterize(self, mu, log_var):
        """Sample z = μ + σ * ε, where ε ~ N(0,1)."""
        std = np.exp(0.5 * log_var)     # σ = exp(log_var / 2)
        eps = np.random.randn(*mu.shape) # ε ~ N(0,1)
        z = mu + std * eps
        return z, eps

    def decode(self, z):
        """z → hidden → reconstruction."""
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        x_hat = 1.0 / (1.0 + np.exp(-(h @ self.W_dec2 + self.b_dec2)))  # sigmoid
        return x_hat, h

    def forward(self, x):
        """Encode → sample → decode."""
        mu, log_var, enc_h = self.encode(x)
        z, eps = self.reparameterize(mu, log_var)
        x_hat, dec_h = self.decode(z)
        return x_hat, mu, log_var, z, eps, enc_h, dec_h


def vae_loss(x, x_hat, mu, log_var):
    """Compute the VAE loss: reconstruction + KL divergence."""
    # Reconstruction loss (binary cross-entropy)
    eps = 1e-8  # numerical stability
    bce = -np.sum(x * np.log(x_hat + eps) + (1 - x) * np.log(1 - x_hat + eps))

    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

    return bce + kl, bce, kl


def train_vae(model, data, epochs=300, lr=0.003):
    """Train VAE with BCE + KL loss."""
    N = len(data)
    velocity = {}
    param_names = ['W_enc1','b_enc1','W_mu','b_mu','W_logvar','b_logvar',
                   'W_dec1','b_dec1','W_dec2','b_dec2']
    for name in param_names:
        velocity[name] = np.zeros_like(getattr(model, name))

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        total_loss, total_bce, total_kl = 0., 0., 0.

        for i in perm:
            x = data[i]
            x_hat, mu, log_var, z, eps, enc_h, dec_h = model.forward(x)

            loss, bce, kl = vae_loss(x, x_hat, mu, log_var)
            total_loss += loss; total_bce += bce; total_kl += kl

            # ── Backward pass ──
            e = 1e-8
            # Gradient of BCE w.r.t. x_hat
            d_x_hat = -(x / (x_hat + e) - (1 - x) / (1 - x_hat + e))

            # Decoder layer 2 (sigmoid output)
            d_pre_sig = d_x_hat * x_hat * (1 - x_hat)  # sigmoid derivative
            d_W_dec2 = np.outer(dec_h, d_pre_sig)
            d_b_dec2 = d_pre_sig
            d_dec_h = d_pre_sig @ model.W_dec2.T

            # Decoder layer 1 (ReLU)
            d_dec_h *= (dec_h > 0).astype(float)
            d_W_dec1 = np.outer(z, d_dec_h)
            d_b_dec1 = d_dec_h
            d_z = d_dec_h @ model.W_dec1.T

            # Reparameterization: z = mu + std * eps
            std = np.exp(0.5 * log_var)
            d_mu = d_z.copy()
            d_log_var_from_z = d_z * eps * 0.5 * std  # chain rule through exp

            # KL gradients
            d_mu += mu                          # d(KL)/d(mu) = mu
            d_log_var_kl = 0.5 * (np.exp(log_var) - 1)  # d(KL)/d(logvar)
            d_log_var = d_log_var_from_z + d_log_var_kl

            # Encoder head: mu = enc_h @ W_mu + b_mu
            d_W_mu = np.outer(enc_h, d_mu)
            d_b_mu = d_mu
            d_enc_h_from_mu = d_mu @ model.W_mu.T

            # Encoder head: log_var = enc_h @ W_logvar + b_logvar
            d_W_logvar = np.outer(enc_h, d_log_var)
            d_b_logvar = d_log_var
            d_enc_h_from_lv = d_log_var @ model.W_logvar.T

            # Encoder layer 1 (ReLU)
            d_enc_h = (d_enc_h_from_mu + d_enc_h_from_lv)
            d_enc_h *= (enc_h > 0).astype(float)
            d_W_enc1 = np.outer(x, d_enc_h)
            d_b_enc1 = d_enc_h

            # Update all parameters
            grads = {'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1,
                     'W_mu': d_W_mu, 'b_mu': d_b_mu,
                     'W_logvar': d_W_logvar, 'b_logvar': d_b_logvar,
                     'W_dec1': d_W_dec1, 'b_dec1': d_b_dec1,
                     'W_dec2': d_W_dec2, 'b_dec2': d_b_dec2}

            for name, grad in grads.items():
                grad_clipped = np.clip(grad, -1.0, 1.0)  # gradient clipping
                velocity[name] = 0.9 * velocity[name] - lr * grad_clipped
                setattr(model, name, getattr(model, name) + velocity[name])

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}  Loss: {total_loss/N:.1f}  "
                  f"BCE: {total_bce/N:.1f}  KL: {total_kl/N:.1f}")


if __name__ == "__main__":
    # Load sklearn digits (8×8 grayscale, values 0-16) and normalize to [0, 1]
    digits = load_digits()
    digit_images = digits.data / 16.0  # shape (1797, 64)

    np.random.seed(42)
    vae = VAE(input_dim=64, latent_dim=2)
    train_vae(vae, digit_images)
