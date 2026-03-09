"""Training RealNVP on 2D two-moons data.

Trains a RealNVP flow with alternating coupling layers using
finite-difference gradient estimation.
"""
import numpy as np


def train_realnvp_2d(X, n_layers=6, hidden=32, lr=0.005, epochs=600):
    """Train a RealNVP flow on 2D data."""
    np.random.seed(42)
    d = 2
    layers = []

    # Build alternating coupling layers with permutations
    for k in range(n_layers):
        layer = {'W1': np.random.randn(1, hidden) * 0.5,
                 'b1': np.zeros(hidden),
                 'W2': np.random.randn(hidden, 2) * 0.1,
                 'b2': np.zeros(2), 'flip': k % 2 == 1}
        layers.append(layer)

    def flow_inverse(x):
        log_det_total = np.zeros(len(x))
        z = x.copy()
        for layer in reversed(layers):
            if layer['flip']:
                z = z[:, ::-1]
            z1, z2 = z[:, 0:1], z[:, 1:2]
            h = np.maximum(0, z1 @ layer['W1'] + layer['b1'])
            out = h @ layer['W2'] + layer['b2']
            s, t = out[:, 0:1], out[:, 1:2]
            s = np.clip(s, -3, 3)
            z2 = (z2 - t) * np.exp(-s)
            z = np.hstack([z1, z2])
            if layer['flip']:
                z = z[:, ::-1]
            log_det_total -= np.sum(s, axis=1)
        return z, log_det_total

    # Training loop
    losses = []
    for epoch in range(epochs):
        z, log_det = flow_inverse(X)
        log_pz = -0.5 * np.sum(z**2, axis=1) - np.log(2 * np.pi)
        log_px = log_pz + log_det
        loss = -np.mean(log_px)
        losses.append(loss)

        # Gradient step (finite differences for simplicity)
        for layer in layers:
            for key in ['W1', 'b1', 'W2', 'b2']:
                grad = np.zeros_like(layer[key])
                it = np.nditer(layer[key], flags=['multi_index'])
                while not it.finished:
                    idx = it.multi_index
                    old_val = layer[key][idx]
                    layer[key][idx] = old_val + 1e-4
                    z2, ld2 = flow_inverse(X)
                    lp2 = -0.5*np.sum(z2**2,axis=1) - np.log(2*np.pi) + ld2
                    layer[key][idx] = old_val
                    grad[idx] = -(np.mean(lp2) - (-loss)) / 1e-4
                    it.iternext()
                grad = np.clip(grad, -5, 5)
                layer[key] -= lr * grad

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: loss = {loss:.3f}")

    return layers, losses


if __name__ == "__main__":
    # Generate moons data
    from numpy import pi
    np.random.seed(42)
    n = 300
    t = np.linspace(0, pi, n//2)
    moon1 = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(n//2, 2)*0.08
    moon2 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5]) + np.random.randn(n//2, 2)*0.08
    X = np.vstack([moon1, moon2])

    layers, losses = train_realnvp_2d(X, n_layers=4, hidden=8, lr=0.001, epochs=600)
    print(f"Final loss: {losses[-1]:.3f} (started at {losses[0]:.3f})")
