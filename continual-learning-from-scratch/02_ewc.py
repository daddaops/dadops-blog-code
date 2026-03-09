"""Block 2: Elastic Weight Consolidation (EWC) — Fisher-based penalty prevents forgetting."""
import numpy as np
from shared import make_task, sigmoid, train_mlp, accuracy

def compute_fisher(X, y, W1, b1, W2, b2, n_samples=200):
    """Diagonal Fisher Information: average squared gradients."""
    fisher_W1 = np.zeros_like(W1)
    fisher_b1 = np.zeros_like(b1)
    fisher_W2 = np.zeros_like(W2)
    fisher_b2 = np.zeros_like(b2)
    idx = np.random.choice(len(y), min(n_samples, len(y)), replace=False)
    for i in idx:
        xi, yi = X[i:i+1], y[i:i+1]
        h = np.maximum(0, xi @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        err = out.ravel() - yi
        gW2 = h.T @ err.reshape(-1, 1)
        gb2 = err
        dh = err.reshape(-1, 1) * W2.T * (h > 0)
        gW1 = xi.T @ dh
        gb1 = dh.ravel()
        fisher_W1 += gW1 ** 2
        fisher_b1 += gb1 ** 2
        fisher_W2 += gW2 ** 2
        fisher_b2 += gb2 ** 2
    n = len(idx)
    return fisher_W1/n, fisher_b1/n, fisher_W2/n, fisher_b2/n

def train_ewc(X, y, W1, b1, W2, b2, old_params, fishers, lam=1000,
              epochs=500, lr=0.05):
    oW1, ob1, oW2, ob2 = old_params
    fW1, fb1, fW2, fb2 = fishers
    for _ in range(epochs):
        h = np.maximum(0, X @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        err = out.ravel() - y
        dW2 = h.T @ err.reshape(-1,1)/len(y) + lam * fW2 * (W2 - oW2)
        db2 = err.mean() + lam * fb2 * (b2 - ob2)
        dh = err.reshape(-1,1) * W2.T * (h > 0)
        dW1 = X.T @ dh/len(y) + lam * fW1 * (W1 - oW1)
        db1 = dh.mean(axis=0) + lam * fb1 * (b1 - ob1)
        for g in [dW1, db1, dW2, db2]:   # clip EWC gradients
            np.clip(g, -5, 5, out=g)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    return W1, b1, W2, b2

if __name__ == "__main__":
    X1, y1 = make_task([-1, 0], [1, 0], seed=42)
    X2, y2 = make_task([0, -1], [0, 1], seed=99)

    # Train task 1, compute Fisher, then train task 2 with EWC
    rng = np.random.RandomState(0)
    W1 = rng.randn(2,8)*0.3; b1 = np.zeros(8)
    W2 = rng.randn(8,1)*0.3; b2 = np.zeros(1)
    W1, b1, W2, b2 = train_mlp(X1, y1, W1, b1, W2, b2)
    fishers = compute_fisher(X1, y1, W1, b1, W2, b2)
    old_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    W1, b1, W2, b2 = train_ewc(X2, y2, W1, b1, W2, b2,
                                old_params, fishers, lam=1500)
    print(f"EWC: acc_task1={accuracy(X1,y1,W1,b1,W2,b2):.0%}, "
          f"acc_task2={accuracy(X2,y2,W1,b1,W2,b2):.0%}")
    # Expected: EWC: acc_task1=96%, acc_task2=77%
