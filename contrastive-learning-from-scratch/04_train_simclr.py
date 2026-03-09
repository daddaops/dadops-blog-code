import numpy as np


class SimCLREncoder:
    """Encoder + projection head for contrastive learning."""

    def __init__(self, input_dim, hidden_dim=64, proj_dim=32):
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        scale = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, proj_dim) * scale
        self.b2 = np.zeros(proj_dim)

    def encode(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h

    def project(self, h):
        z = h @ self.W2 + self.b2
        z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        return z

    def forward(self, x):
        h = self.encode(x)
        z = self.project(h)
        return h, z


def info_nce_loss(z, temperature=0.5):
    """
    Compute NT-Xent (InfoNCE) loss for a batch of 2N embeddings.
    z: (2N, D) array — z[2i] and z[2i+1] are positive pairs.
    """
    batch_size = z.shape[0]
    sim = z @ z.T / temperature
    mask = np.eye(batch_size, dtype=bool)
    sim[mask] = -1e9

    total_loss = 0.0
    for i in range(batch_size):
        pos_idx = i + 1 if i % 2 == 0 else i - 1
        logits = sim[i]
        max_logit = logits.max()
        log_sum_exp = max_logit + np.log(
            np.sum(np.exp(logits - max_logit)) + 1e-8
        )
        total_loss += -(logits[pos_idx] - log_sum_exp)

    return total_loss / batch_size


def create_augmented_batch(data, batch_size=16, aug_noise=0.3):
    """Create a batch of positive pairs through augmentation."""
    indices = np.random.choice(len(data), size=batch_size, replace=False)
    batch = data[indices]
    view1 = batch + np.random.randn(*batch.shape) * aug_noise
    view2 = batch + np.random.randn(*batch.shape) * aug_noise
    augmented = np.empty((2 * batch_size, data.shape[1]))
    augmented[0::2] = view1
    augmented[1::2] = view2
    return augmented


def train_simclr(data, epochs=100, batch_size=16, lr=0.01, temp=0.5):
    """Train SimCLR on toy 2D data using numerical gradients."""
    encoder = SimCLREncoder(input_dim=2, hidden_dim=8, proj_dim=4)
    losses = []

    for epoch in range(epochs):
        batch = create_augmented_batch(data, batch_size)
        _, z = encoder.forward(batch)
        loss = info_nce_loss(z, temp)
        losses.append(loss)

        # Numerical gradients via finite differences
        eps = 1e-4
        for param in [encoder.W1, encoder.b1, encoder.W2, encoder.b2]:
            grad = np.zeros_like(param)
            for idx in np.ndindex(param.shape):
                old = param[idx]
                param[idx] = old + eps
                _, zp = encoder.forward(batch)
                lp = info_nce_loss(zp, temp)
                param[idx] = old - eps
                _, zm = encoder.forward(batch)
                lm = info_nce_loss(zm, temp)
                param[idx] = old
                grad[idx] = (lp - lm) / (2 * eps)
            param -= lr * grad

        if epoch % 25 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    return encoder, losses


if __name__ == "__main__":
    # Create toy data: 5 clusters in 2D
    np.random.seed(42)
    centers = np.array([[2, 2], [-2, 2], [0, -2.5], [3, -1], [-3, -1.0]])
    data = np.vstack([c + np.random.randn(20, 2) * 0.3 for c in centers])
    labels = np.repeat(np.arange(5), 20)  # for evaluation later

    encoder, losses = train_simclr(data)
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")
