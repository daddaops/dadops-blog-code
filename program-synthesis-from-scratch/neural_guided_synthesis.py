"""Neural-guided program synthesis.

Trains an MLP to predict which DSL operations a target program
likely uses, then prioritizes enumeration accordingly.
"""
import numpy as np

def generate_training_data(n_tasks=5000, seed=42):
    """Generate synthetic (I/O examples -> ops used) training pairs."""
    rng = np.random.RandomState(seed)
    all_ops = ['+', '*', '-']
    X, Y = [], []
    for _ in range(n_tasks):
        # Random program: pick 1-2 ops, build a small expression
        ops_used = list(rng.choice(all_ops, size=rng.randint(1, 3), replace=False))
        a, b = rng.randint(1, 5), rng.randint(0, 4)
        if '+' in ops_used and '*' in ops_used:
            fn = lambda x, a=a, b=b: a * x + b
        elif '*' in ops_used:
            fn = lambda x, a=a, b=b: a * x * b
        elif '-' in ops_used:
            fn = lambda x, a=a, b=b: a - x + b
        else:
            fn = lambda x, a=a, b=b: x + a + b
        # Generate I/O feature vector: outputs for x = 0..4
        io_features = [fn(x) for x in range(5)]
        X.append(io_features)
        Y.append([1 if op in ops_used else 0 for op in all_ops])
    return np.array(X, dtype=float), np.array(Y, dtype=float)

def train_guide(X, Y, lr=0.05, epochs=1000):
    """Train a tiny MLP: 5 inputs -> 16 hidden -> 3 outputs (sigmoid)."""
    rng = np.random.RandomState(0)
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - mu) / sigma  # normalize
    W1 = rng.randn(5, 16) * 0.3
    b1 = np.zeros(16)
    W2 = rng.randn(16, 3) * 0.3
    b2 = np.zeros(3)
    for _ in range(epochs):
        h = np.maximum(0, X @ W1 + b1)       # ReLU hidden
        logits = h @ W2 + b2
        pred = 1 / (1 + np.exp(-logits))      # sigmoid output
        grad_out = (pred - Y) / len(X)
        grad_W2 = h.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_h = grad_out @ W2.T
        grad_h[h <= 0] = 0
        W2 -= lr * grad_W2; b2 -= lr * grad_b2
        W1 -= lr * X.T @ grad_h; b1 -= lr * grad_h.sum(axis=0)
    return lambda io: 1/(1+np.exp(-(np.maximum(0, ((np.array(io)-mu)
                        /sigma) @ W1+b1) @ W2+b2)))

X, Y = generate_training_data()
predict_ops = train_guide(X, Y)

# Test: for f(x) = 2x + 1, I/O is [1, 3, 5, 7, 9]
scores = predict_ops([1, 3, 5, 7, 9])
ops = ['+', '*', '-']
ranked = sorted(zip(ops, scores), key=lambda t: -t[1])
print("Predicted op priorities:", [(op, f"{s:.2f}") for op, s in ranked])
# Predicted op priorities: [('*', 0.95), ('+', 0.69), ('-', 0.14)]
