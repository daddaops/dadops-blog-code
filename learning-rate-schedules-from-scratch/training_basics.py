import numpy as np

# --- Tiny MLP for classification (2 hidden layers, ReLU, softmax) ---
def init_mlp(dims, seed=42):
    rng = np.random.RandomState(seed)
    params = []
    for i in range(len(dims) - 1):
        scale = np.sqrt(2.0 / dims[i])          # He initialization
        W = rng.randn(dims[i], dims[i + 1]) * scale
        b = np.zeros(dims[i + 1])
        params.append((W, b))
    return params

def forward(params, X):
    acts = [X]
    for W, b in params[:-1]:                     # hidden layers: ReLU
        X = np.maximum(0, X @ W + b)
        acts.append(X)
    W, b = params[-1]                            # output layer: softmax
    logits = X @ W + b
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    acts.append(probs)
    return acts

def cross_entropy(probs, y):
    n = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-12)
    return -log_p.mean()

def backward_and_update(params, acts, y, lr):
    n = len(y)
    grad = acts[-1].copy()
    grad[np.arange(n), y] -= 1                   # softmax-CE shortcut
    grad /= n
    for i in reversed(range(len(params))):
        W, b = params[i]
        dW = acts[i].T @ grad
        db = grad.sum(axis=0)
        W -= lr * dW                              # SGD update with LR
        b -= lr * db
        if i > 0:
            grad = grad @ W.T * (acts[i] > 0)    # ReLU derivative

# --- Generate spiral dataset (3 classes) ---
def make_spirals(n_per_class=100, noise=0.25, seed=0):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in range(3):
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.randn(n_per_class) * noise
        r = np.linspace(0.2, 1.0, n_per_class)
        X.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)

X, y = make_spirals()

if __name__ == "__main__":
    # --- Train with three constant LRs ---
    for lr, label in [(1.0, "too high"), (0.0001, "too low"), (0.01, "okay-ish")]:
        params = init_mlp([2, 64, 64, 3])
        losses = []
        for epoch in range(300):
            acts = forward(params, X)
            losses.append(cross_entropy(acts[-1], y))
            backward_and_update(params, acts, y, lr)
        print(f"LR={lr:<8} ({label:<9}): final loss = {losses[-1]:.4f}")
