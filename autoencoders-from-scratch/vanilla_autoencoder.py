"""Vanilla Autoencoder from Scratch — 64→32→2→32→64 on 8×8 digit images.

Implements a simple autoencoder with manual backpropagation through
encoder (ReLU) and decoder (ReLU + clip) layers, trained with SGD + momentum.

From: https://www.dadops.co/blog/autoencoders-from-scratch/
"""
import numpy as np
from sklearn.datasets import load_digits


class Autoencoder:
    """Vanilla autoencoder: 64 → 32 → 2 → 32 → 64."""
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=2):
        scale = np.sqrt(2.0 / input_dim)  # He initialization
        # Encoder weights
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_enc2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_enc2 = np.zeros(latent_dim)
        # Decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_dec2 = np.zeros(input_dim)

    def encode(self, x):
        """x → hidden → latent."""
        h = np.maximum(0, x @ self.W_enc1 + self.b_enc1)  # ReLU
        z = h @ self.W_enc2 + self.b_enc2                  # linear (no activation)
        return z, h  # return h for backprop

    def decode(self, z):
        """latent → hidden → reconstruction."""
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)   # ReLU
        x_hat = np.clip(h @ self.W_dec2 + self.b_dec2, 0, 1)  # sigmoid-like clamp
        return x_hat, h

    def forward(self, x):
        """Full forward pass: encode then decode."""
        z, enc_h = self.encode(x)
        x_hat, dec_h = self.decode(z)
        return x_hat, z, enc_h, dec_h


def train_autoencoder(model, data, epochs=200, lr=0.005):
    """Train with MSE loss and simple SGD + momentum."""
    N = len(data)
    velocity = {}  # momentum buffers for each parameter
    for name in ['W_enc1','b_enc1','W_enc2','b_enc2',
                  'W_dec1','b_dec1','W_dec2','b_dec2']:
        velocity[name] = np.zeros_like(getattr(model, name))

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(N)
        total_loss = 0.0

        for i in perm:
            x = data[i]  # shape (64,) — one 8×8 image flattened

            # Forward pass
            x_hat, z, enc_h, dec_h = model.forward(x)

            # MSE loss: (1/d) * sum((x - x_hat)^2)
            diff = x_hat - x
            loss = np.mean(diff ** 2)
            total_loss += loss

            # Backward pass (chain rule through decoder then encoder)
            d_x_hat = 2.0 * diff / len(x)

            # Decoder layer 2: x_hat = clip(dec_h @ W_dec2 + b_dec2)
            # Gradient of clip: 1 where 0 < output < 1, else 0
            mask_out = ((x_hat > 0) & (x_hat < 1)).astype(float)
            d_pre_clip = d_x_hat * mask_out
            d_W_dec2 = np.outer(dec_h, d_pre_clip)
            d_b_dec2 = d_pre_clip
            d_dec_h = d_pre_clip @ model.W_dec2.T

            # Decoder layer 1: dec_h = relu(z @ W_dec1 + b_dec1)
            d_dec_h *= (dec_h > 0).astype(float)  # ReLU gradient
            d_W_dec1 = np.outer(z, d_dec_h)
            d_b_dec1 = d_dec_h
            d_z = d_dec_h @ model.W_dec1.T

            # Encoder layer 2: z = enc_h @ W_enc2 + b_enc2
            d_W_enc2 = np.outer(enc_h, d_z)
            d_b_enc2 = d_z
            d_enc_h = d_z @ model.W_enc2.T

            # Encoder layer 1: enc_h = relu(x @ W_enc1 + b_enc1)
            d_enc_h *= (enc_h > 0).astype(float)
            d_W_enc1 = np.outer(x, d_enc_h)
            d_b_enc1 = d_enc_h

            # SGD update with momentum (0.9)
            grads = {'W_enc1': d_W_enc1, 'b_enc1': d_b_enc1,
                     'W_enc2': d_W_enc2, 'b_enc2': d_b_enc2,
                     'W_dec1': d_W_dec1, 'b_dec1': d_b_dec1,
                     'W_dec2': d_W_dec2, 'b_dec2': d_b_dec2}

            for name, grad in grads.items():
                velocity[name] = 0.9 * velocity[name] - lr * grad
                param = getattr(model, name)
                setattr(model, name, param + velocity[name])

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}  Loss: {total_loss/N:.6f}")


if __name__ == "__main__":
    # Load sklearn digits (8×8 grayscale, values 0-16) and normalize to [0, 1]
    digits = load_digits()
    digit_images = digits.data / 16.0  # shape (1797, 64)

    np.random.seed(42)
    ae = Autoencoder(input_dim=64, latent_dim=2)
    train_autoencoder(ae, digit_images)
