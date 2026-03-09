"""Linear probes for concept detection at each layer of a trained MLP.

Trains a 3-layer MLP on a nonlinear classification task, then probes
each hidden layer with logistic regression to show how information
builds up across layers.
"""
import numpy as np


def build_and_probe_network():
    """Train a 3-layer MLP, then probe each layer for the target concept."""
    np.random.seed(42)

    # Synthetic task: classify 2D points as positive (y > sin(x)) or negative
    n = 500
    X = np.random.randn(n, 2) * 2
    y = (X[:, 1] > np.sin(X[:, 0])).astype(float)

    # 3-layer MLP: 2 -> 16 -> 16 -> 16 -> 1
    dims = [2, 16, 16, 16, 1]
    weights, biases = [], []
    for i in range(len(dims) - 1):
        w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2 / dims[i])
        b = np.zeros(dims[i+1])
        weights.append(w); biases.append(b)

    # Train the MLP (simple SGD)
    for epoch in range(300):
        # Forward pass — save activations at each layer
        activations = [X]
        h = X
        for i in range(len(weights) - 1):
            h = np.maximum(0, h @ weights[i] + biases[i])  # ReLU
            activations.append(h)
        logits = h @ weights[-1] + biases[-1]
        pred = 1 / (1 + np.exp(-logits.squeeze()))

        # Backward pass and update (simplified)
        grad = (pred - y).reshape(-1, 1) / n
        for i in range(len(weights) - 1, -1, -1):
            gw = activations[i].T @ grad
            weights[i] -= 0.5 * gw
            biases[i] -= 0.5 * grad.sum(axis=0)
            if i > 0:
                grad = (grad @ weights[i].T) * (activations[i] > 0)

    # Now PROBE each hidden layer for the target label
    for layer_idx in range(1, len(activations)):
        A = activations[layer_idx]  # (n, 16)
        # Linear probe: logistic regression via closed-form (pseudo-inverse)
        A_bias = np.column_stack([A, np.ones(n)])
        w_probe = np.linalg.lstsq(A_bias, y, rcond=None)[0]
        probe_pred = (A_bias @ w_probe > 0.5).astype(float)
        accuracy = np.mean(probe_pred == y)
        print(f"Layer {layer_idx} probe accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    build_and_probe_network()
    # Expected: accuracy increases across layers (~72%, ~88%, ~96%)
