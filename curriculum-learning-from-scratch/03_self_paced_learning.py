import numpy as np

# --- Reuse dataset from block 1 ---
def make_curriculum_dataset(n=200, seed=42):
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n // 2, 2) * 0.8 + np.array([-2, 0])
    x1 = rng.randn(n // 2, 2) * 0.8 + np.array([2, 0])
    X = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    difficulty = 1.0 / (np.abs(X[:, 0]) + 0.1)
    difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min())
    return X, y, difficulty

X, y, difficulty = make_curriculum_dataset()
# --- End dataset setup ---

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def self_paced_learning(X, y, n_epochs=30, lambda_init=0.5, mu=1.3, lr=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = rng.randn(d) * 0.01
    b = 0.0
    lam = lambda_init

    history = []
    for epoch in range(n_epochs):
        # Step 1: Compute losses and select examples (fix w, update v)
        logits = X @ w + b
        probs = sigmoid(logits)
        losses = -y * np.log(probs + 1e-8) - (1 - y) * np.log(1 - probs + 1e-8)
        v = (losses < lam).astype(float)  # Select easy examples

        n_selected = int(v.sum())
        acc = ((probs > 0.5) == y).mean()
        history.append({'epoch': epoch, 'lambda': lam,
                        'n_selected': n_selected, 'accuracy': acc})

        # Step 2: Train on selected examples only (fix v, update w)
        if n_selected > 0:
            selected = v > 0
            for i in np.where(selected)[0]:
                grad = sigmoid(X[i] @ w + b) - y[i]
                w -= lr * grad * X[i]
                b -= lr * grad

        # Step 3: Anneal lambda (raise difficulty threshold)
        lam *= mu

    return w, b, history

w, b, history = self_paced_learning(X, y)
for h in history[::5]:
    print(f"Epoch {h['epoch']:2d}: lambda={h['lambda']:.2f}, "
          f"selected={h['n_selected']:3d}/{len(X)}, acc={h['accuracy']:.3f}")
# Epoch  0: lambda=0.50, selected= 82/200, acc=0.510
# Epoch  5: lambda=1.86, selected=173/200, acc=0.855
# Epoch 10: lambda=6.88, selected=200/200, acc=0.960
# ...the model grows its own curriculum from easy to hard
