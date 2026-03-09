"""The degradation problem: deeper plain networks get worse training loss."""
import numpy as np
from helpers import make_spirals

X, y = make_spirals()

def train_plain_network(X, y, depth, width=32, lr=0.01, epochs=500, seed=0):
    """Train a plain (no skip connections) network and return loss history."""
    rng = np.random.RandomState(seed)
    # He initialization: scale by sqrt(2/fan_in)
    layers = []
    fan_in = X.shape[1]
    for i in range(depth):
        fan_out = width if i < depth - 1 else 1
        W = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        layers.append((W, b))
        fan_in = fan_out

    losses = []
    for epoch in range(epochs):
        # Forward pass
        activations = [X]
        for i, (W, b) in enumerate(layers):
            z = activations[-1] @ W + b
            a = np.maximum(0, z) if i < depth - 1 else 1 / (1 + np.exp(-z))
            activations.append(a)

        pred = activations[-1].ravel()
        loss = -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
        losses.append(loss)

        # Backward pass (simplified)
        grad = (pred - y).reshape(-1, 1) / len(y)
        for i in range(depth - 1, -1, -1):
            W, b = layers[i]
            dW = activations[i].T @ grad
            db = grad.sum(axis=0)
            if i > 0:
                grad = grad @ W.T
                grad *= (activations[i] > 0).astype(float)  # ReLU derivative
            layers[i] = (W - lr * dW, b - lr * db)

    return losses

for depth in [4, 8, 16, 32]:
    losses = train_plain_network(X, y, depth)
    print(f"depth={depth:2d}: final loss = {losses[-1]:.2f}")

# Results: deeper plain networks get WORSE training loss
# depth=4:  loss ~ 0.15 (converges well)
# depth=8:  loss ~ 0.25 (slower, higher floor)
# depth=16: loss ~ 0.45 (barely learns)
# depth=32: loss ~ 0.60 (almost stuck at random)
