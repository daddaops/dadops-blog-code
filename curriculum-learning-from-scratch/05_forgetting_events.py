import numpy as np

# --- Reuse dataset from block 1 ---
def make_curriculum_dataset(n=200, seed=42):
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n // 2, 2) * 0.8 + np.array([-1, 0])
    x1 = rng.randn(n // 2, 2) * 0.8 + np.array([1, 0])
    X = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    difficulty = 1.0 / (np.abs(X[:, 0]) + 0.1)
    difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min())
    return X, y, difficulty

X, y, difficulty = make_curriculum_dataset()
# --- End dataset setup ---

def compute_forgetting_events(X, y, n_epochs=20, lr=0.1, seed=42):
    """Track forgetting events for each example across training."""
    rng = np.random.RandomState(seed)
    n = len(X)
    w = rng.randn(X.shape[1]) * 0.01
    b = 0.0

    forgetting_counts = np.zeros(n)
    prev_correct = np.zeros(n, dtype=bool)

    for epoch in range(n_epochs):
        # Evaluate all examples
        logits = X @ w + b
        preds = logits > 0
        correct = preds == y

        # Count forgetting: was correct, now wrong
        if epoch > 0:
            forgot = prev_correct & ~correct
            forgetting_counts += forgot

        prev_correct = correct.copy()

        # SGD update (random order)
        order = rng.permutation(n)
        for i in order:
            prob = 1.0 / (1.0 + np.exp(-X[i] @ w - b))
            grad = prob - y[i]
            w -= lr * grad * X[i]
            b -= lr * grad

    # Classify examples
    unforgettable = forgetting_counts == 0
    print(f"Unforgettable: {unforgettable.sum()}/{n} ({unforgettable.mean():.1%})")
    print(f"Forgotten 1+ times: {(forgetting_counts > 0).sum()}/{n}")
    return forgetting_counts

forgetting = compute_forgetting_events(X, y)
# Unforgettable: 196/200 (98.0%)
# Forgotten 1+ times: 4/200
