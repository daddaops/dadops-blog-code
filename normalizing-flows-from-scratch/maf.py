"""Masked Autoregressive Flow (MAF).

Simple 2D MAF where x1 is unconditional and x2 depends on x1
through an MLP. Trains on two-moon data.
"""
import numpy as np


def masked_autoregressive_flow_2d(X, epochs=400, lr=0.01):
    """Simple 2D Masked Autoregressive Flow (MAF)."""
    np.random.seed(42)

    # For 2D MAF: x1 depends on nothing, x2 depends on x1
    # x1 = z1 * exp(s1) + t1 (s1, t1 are learned constants)
    # x2 = z2 * exp(s2(x1)) + t2(x1)
    s1, t1 = np.zeros(1), np.zeros(1)  # mutable arrays for in-place updates
    # MLP for dimension 2: x1 -> (s2, t2)
    W1 = np.random.randn(1, 8) * 0.3
    b1 = np.zeros(8)
    W2 = np.random.randn(8, 2) * 0.1
    b2 = np.zeros(2)

    def inverse_pass(x):
        """Map data x to latent z (used for training)."""
        z1 = (x[:, 0:1] - t1[0]) * np.exp(-s1[0])
        h = np.maximum(0, x[:, 0:1] @ W1 + b1)
        st = h @ W2 + b2
        s2, t2 = st[:, 0:1], st[:, 1:2]
        s2 = np.clip(s2, -3, 3)
        z2 = (x[:, 1:2] - t2) * np.exp(-s2)
        log_det = -(s1[0] + np.sum(s2, axis=1))
        return np.hstack([z1, z2]), log_det

    losses = []
    for epoch in range(epochs):
        z, log_det = inverse_pass(X)
        log_pz = -0.5 * np.sum(z**2, axis=1) - np.log(2 * np.pi)
        log_px = log_pz + log_det
        loss = -np.mean(log_px)
        losses.append(loss)

        # Numerical gradient update
        eps = 1e-4
        # Update s1, t1
        for param in [s1, t1]:
            old_val = param[0]
            param[0] = old_val + eps
            z2, ld2 = inverse_pass(X)
            lp2 = np.mean(-0.5*np.sum(z2**2,axis=1) - np.log(2*np.pi) + ld2)
            param[0] = old_val
            g = -(lp2 - (-loss)) / eps
            g = np.clip(g, -5, 5)
            param[0] -= lr * g

        # Update MLP params
        for key, arr in [('W1', W1), ('b1', b1), ('W2', W2), ('b2', b2)]:
            it = np.nditer(arr, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                old_val = arr[idx]
                arr[idx] = old_val + eps
                z2, ld2 = inverse_pass(X)
                lp2 = np.mean(-0.5*np.sum(z2**2,axis=1) - np.log(2*np.pi) + ld2)
                arr[idx] = old_val
                g = -(lp2 - (-loss)) / eps
                g = np.clip(g, -5, 5)
                arr[idx] -= lr * g
                it.iternext()

    print(f"MAF loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    return losses


if __name__ == "__main__":
    # Generate moons data (same as train_realnvp.py)
    from numpy import pi
    np.random.seed(42)
    n = 300
    t = np.linspace(0, pi, n//2)
    moon1 = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(n//2, 2)*0.08
    moon2 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5]) + np.random.randn(n//2, 2)*0.08
    X = np.vstack([moon1, moon2])

    losses = masked_autoregressive_flow_2d(X, epochs=400, lr=0.001)
