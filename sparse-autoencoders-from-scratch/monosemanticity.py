"""Monosemanticity experiment: train a classifier, then decompose its hidden layer with an SAE.

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


def monosemanticity_experiment():
    """Train a classifier, then decompose its hidden layer with an SAE."""
    np.random.seed(7)
    n_classes, d_input, d_hidden = 8, 20, 16

    # Generate clustered data: 8 classes with distinct centers
    centers = np.random.randn(n_classes, d_input) * 3
    X_all, Y_all = [], []
    for c in range(n_classes):
        pts = centers[c] + np.random.randn(200, d_input) * 0.8
        X_all.append(pts)
        Y_all.extend([c] * 200)
    X_all = np.vstack(X_all)
    Y_all = np.array(Y_all)

    # Train a simple 2-layer MLP classifier
    W1 = np.random.randn(d_input, d_hidden) * np.sqrt(2.0 / d_input)
    b1 = np.zeros(d_hidden)
    W2 = np.random.randn(d_hidden, n_classes) * np.sqrt(2.0 / d_hidden)
    b2 = np.zeros(n_classes)

    for step in range(500):
        h = np.maximum(0, X_all @ W1 + b1)  # hidden activations
        logits = h @ W2 + b2
        # Softmax + cross-entropy (stable)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        one_hot = np.eye(n_classes)[Y_all]
        d_logits = (probs - one_hot) / len(Y_all)
        # Backprop
        d_W2 = h.T @ d_logits
        d_b2 = d_logits.sum(axis=0)
        d_h = d_logits @ W2.T * (h > 0)
        d_W1 = X_all.T @ d_h
        d_b1 = d_h.sum(axis=0)
        lr = 0.01
        W1 -= lr * d_W1; b1 -= lr * d_b1
        W2 -= lr * d_W2; b2 -= lr * d_b2

    # Extract hidden activations
    H = np.maximum(0, X_all @ W1 + b1)

    # Measure polysemanticity of original neurons
    print("=== Original Hidden Neurons (Polysemantic) ===")
    for neuron in [0, 3, 7]:
        active_classes = []
        for c in range(n_classes):
            mask = Y_all == c
            mean_act = H[mask, neuron].mean()
            if mean_act > 0.5:
                active_classes.append(c)
        print(f"  Neuron {neuron} responds to classes: {active_classes}")

    # Train SAE on hidden activations
    d_sae = 64  # 4x overcomplete
    sae = SparseAutoencoder(d_hidden, d_sae)
    # (Training loop similar to above, abbreviated for clarity)
    lambda_l1 = 0.05
    lr = 0.005
    m_s = {k: np.zeros_like(getattr(sae, k))
           for k in ['W_enc', 'b_enc', 'W_dec', 'b_dec']}
    v_s = {k: np.zeros_like(getattr(sae, k))
           for k in ['W_enc', 'b_enc', 'W_dec', 'b_dec']}
    for epoch in range(300):
        x_hat, z, z_pre = sae.forward(H)
        d_xhat = 2 * (x_hat - H) / H.shape[0]
        d_z = d_xhat @ sae.W_dec + lambda_l1 * np.sign(z) / H.shape[0]
        d_z_pre = d_z * (z_pre > 0)
        grads = {
            'W_dec': d_xhat.T @ z,
            'W_enc': d_z_pre.T @ (H - sae.b_dec),
            'b_enc': d_z_pre.sum(axis=0),
            'b_dec': d_xhat.sum(axis=0) - (d_z_pre @ sae.W_enc).sum(axis=0)
        }
        t = epoch + 1
        for key in grads:
            m_s[key] = 0.9 * m_s[key] + 0.1 * grads[key]
            v_s[key] = 0.999 * v_s[key] + 0.001 * grads[key] ** 2
            param = getattr(sae, key)
            param -= lr * (m_s[key] / (1 - 0.9**t)) / (
                np.sqrt(v_s[key] / (1 - 0.999**t)) + 1e-8)
        sae.normalize_decoder()

    # Analyze SAE features: are they monosemantic?
    _, z_final, _ = sae.forward(H)
    print("\n=== SAE Features (Monosemantic) ===")
    alive_features = np.where(np.any(z_final > 0.1, axis=0))[0]
    for feat in alive_features[:8]:
        activations_by_class = []
        for c in range(n_classes):
            mask = Y_all == c
            activations_by_class.append(z_final[mask, feat].mean())
        dominant = np.argmax(activations_by_class)
        purity = max(activations_by_class) / (sum(activations_by_class) + 1e-8)
        print(f"  Feature {feat:2d}: dominant class = {dominant}, "
              f"purity = {purity:.1%}")


if __name__ == "__main__":
    monosemanticity_experiment()
