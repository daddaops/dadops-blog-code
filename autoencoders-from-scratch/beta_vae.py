"""β-VAE from Scratch — comparing different β values.

Shows the effect of β on the reconstruction vs regularization tradeoff:
  β=0.5 → sharp reconstructions, poor generation
  β=1.0 → standard VAE balance
  β=4.0 → blurry reconstructions, smooth generation, disentangled dims

From: https://www.dadops.co/blog/autoencoders-from-scratch/
"""
import numpy as np
from sklearn.datasets import load_digits
from vae import VAE, train_vae


def vae_loss_beta(x, x_hat, mu, log_var, beta=1.0):
    """VAE loss with configurable β for the KL term."""
    eps = 1e-8
    bce = -np.sum(x * np.log(x_hat + eps) + (1 - x) * np.log(1 - x_hat + eps))
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    return bce + beta * kl, bce, kl


def train_beta_vae(model, data, epochs=300, lr=0.001, beta=1.0):
    """Train VAE with configurable β, using the same backprop as train_vae."""
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

            loss, bce, kl = vae_loss_beta(x, x_hat, mu, log_var, beta=beta)
            total_loss += loss; total_bce += bce; total_kl += kl

            # ── Backward pass (same as VAE but KL grads scaled by β) ──
            e = 1e-8
            d_x_hat = -(x / (x_hat + e) - (1 - x) / (1 - x_hat + e))

            d_pre_sig = d_x_hat * x_hat * (1 - x_hat)
            d_W_dec2 = np.outer(dec_h, d_pre_sig)
            d_b_dec2 = d_pre_sig
            d_dec_h = d_pre_sig @ model.W_dec2.T

            d_dec_h *= (dec_h > 0).astype(float)
            d_W_dec1 = np.outer(z, d_dec_h)
            d_b_dec1 = d_dec_h
            d_z = d_dec_h @ model.W_dec1.T

            std = np.exp(0.5 * log_var)
            d_mu = d_z.copy()
            d_log_var_from_z = d_z * eps * 0.5 * std

            # KL gradients scaled by β
            d_mu += beta * mu
            d_log_var_kl = beta * 0.5 * (np.exp(log_var) - 1)
            d_log_var = d_log_var_from_z + d_log_var_kl

            d_W_mu = np.outer(enc_h, d_mu)
            d_b_mu = d_mu
            d_enc_h_from_mu = d_mu @ model.W_mu.T

            d_W_logvar = np.outer(enc_h, d_log_var)
            d_b_logvar = d_log_var
            d_enc_h_from_lv = d_log_var @ model.W_logvar.T

            d_enc_h = (d_enc_h_from_mu + d_enc_h_from_lv)
            d_enc_h *= (enc_h > 0).astype(float)
            d_W_enc1 = np.outer(x, d_enc_h)
            d_b_enc1 = d_enc_h

            grads = {'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1,
                     'W_mu': d_W_mu, 'b_mu': d_b_mu,
                     'W_logvar': d_W_logvar, 'b_logvar': d_b_logvar,
                     'W_dec1': d_W_dec1, 'b_dec1': d_b_dec1,
                     'W_dec2': d_W_dec2, 'b_dec2': d_b_dec2}

            for name, grad in grads.items():
                grad_clipped = np.clip(grad, -5.0, 5.0)
                velocity[name] = 0.9 * velocity[name] - lr * grad_clipped
                setattr(model, name, getattr(model, name) + velocity[name])

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}  Loss: {total_loss/N:.1f}  "
                  f"BCE: {total_bce/N:.1f}  KL: {total_kl/N:.1f}")


if __name__ == "__main__":
    digits = load_digits()
    digit_images = digits.data / 16.0

    for beta in [0.5, 1.0, 4.0]:
        print(f"\n{'='*50}")
        print(f"Training β-VAE with β = {beta}")
        print(f"{'='*50}")
        np.random.seed(42)
        model = VAE(input_dim=64, latent_dim=2)
        train_beta_vae(model, digit_images, epochs=300, lr=0.003, beta=beta)
