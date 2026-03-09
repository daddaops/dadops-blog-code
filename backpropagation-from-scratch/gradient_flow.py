"""Vanishing gradient demonstration — measure gradient magnitude at layer 1
across different depths, activations, and with/without residual connections."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def measure_gradient_flow(n_layers, activation='sigmoid', residual=False):
    """Measure how much gradient reaches layer 1 in an n-layer network."""
    np.random.seed(42)
    dim = 16
    x = np.random.randn(1, dim)

    # Forward pass — store activations and pre-activations
    weights = []
    pre_acts = []
    acts = [x]
    for i in range(n_layers):
        W = np.random.randn(dim, dim) * (2.0 / dim) ** 0.5  # Kaiming init
        weights.append(W)
        z = acts[-1] @ W
        pre_acts.append(z)
        if activation == 'sigmoid':
            a = sigmoid(z)
        else:
            a = np.maximum(0, z)   # ReLU
        if residual and i > 0:
            a = a + acts[-1]       # skip connection
        acts.append(a)

    # Backward pass — track gradient magnitude
    grad = np.ones_like(acts[-1])  # dL/d(output) = 1
    for i in range(n_layers - 1, -1, -1):
        if residual and i > 0:
            grad_skip = grad.copy()  # residual path
        if activation == 'sigmoid':
            s = sigmoid(pre_acts[i])
            local = s * (1 - s)
        else:
            local = (pre_acts[i] > 0).astype(float)
        grad = grad * local @ weights[i].T
        if residual and i > 0:
            grad = grad + grad_skip  # add skip gradient

    return np.mean(np.abs(grad))

# Compare across depths
print(f"{'Layers':>8} {'Sigmoid':>12} {'ReLU':>12} {'Sig+Resid':>12}")
print("-" * 48)
for n in [2, 5, 10, 15, 20]:
    g_sig = measure_gradient_flow(n, 'sigmoid', False)
    g_relu = measure_gradient_flow(n, 'relu', False)
    g_res = measure_gradient_flow(n, 'sigmoid', True)
    print(f"{n:>8} {g_sig:>12.2e} {g_relu:>12.2e} {g_res:>12.2e}")
