"""Block 4: PackNet — prune and freeze weights per task for zero-forgetting."""
import numpy as np
from shared import make_task, sigmoid, accuracy

def train_packnet(X, y, W1, b1, W2, b2, mask_W1, mask_b1, mask_W2, mask_b2,
                  epochs=200, lr=0.05):
    """Train only the parameters where mask == 1 (unfrozen)."""
    for _ in range(epochs):
        h = np.maximum(0, X @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        err = out.ravel() - y
        dW2 = h.T @ err.reshape(-1,1) / len(y)
        db2 = err.mean()
        dh = err.reshape(-1,1) * W2.T * (h > 0)
        dW1 = X.T @ dh / len(y)
        db1 = dh.mean(axis=0)
        W1 -= lr * dW1 * mask_W1   # only update unfrozen weights
        b1 -= lr * db1 * mask_b1
        W2 -= lr * dW2 * mask_W2
        b2 -= lr * db2 * mask_b2
    return W1, b1, W2, b2

def prune_and_freeze(W, keep_ratio=0.30):
    """Keep top keep_ratio weights by magnitude, return frozen mask."""
    flat = np.abs(W).ravel()
    if len(flat) == 0:
        return np.ones_like(W), np.zeros_like(W)
    threshold = np.percentile(flat[flat > 0], (1 - keep_ratio) * 100)
    frozen = (np.abs(W) >= threshold).astype(float)
    W *= frozen                    # zero out pruned weights
    free = 1.0 - frozen           # mask for next task
    return frozen, free

if __name__ == "__main__":
    X1, y1 = make_task([-2, -2], [2, 2], seed=42)
    X2, y2 = make_task([-2, 2], [2, -2], seed=99)

    # Task 1: train full network, then prune to 30%
    rng = np.random.RandomState(0)
    W1 = rng.randn(2, 16)*0.3; b1 = np.zeros(16)
    W2 = rng.randn(16, 1)*0.3; b2 = np.zeros(1)
    ones_W1 = np.ones_like(W1); ones_b1 = np.ones_like(b1)
    ones_W2 = np.ones_like(W2); ones_b2 = np.ones_like(b2)

    W1,b1,W2,b2 = train_packnet(X1,y1,W1,b1,W2,b2,
                                 ones_W1,ones_b1,ones_W2,ones_b2)
    frozen_W1, free_W1 = prune_and_freeze(W1, keep_ratio=0.30)
    frozen_W2, free_W2 = prune_and_freeze(W2, keep_ratio=0.30)
    free_b1 = np.ones_like(b1); free_b2 = np.ones_like(b2)

    print(f"After Task 1 + prune: acc={accuracy(X1,y1,W1,b1,W2,b2):.0%}, "
          f"free params: {free_W1.sum() + free_W2.sum():.0f}/{W1.size + W2.size}")

    # Task 2: train only the freed weights
    W1,b1,W2,b2 = train_packnet(X2,y2,W1,b1,W2,b2,
                                 free_W1,free_b1,free_W2,free_b2)
    print(f"After Task 2: acc_task1={accuracy(X1,y1,W1,b1,W2,b2):.0%}, "
          f"acc_task2={accuracy(X2,y2,W1,b1,W2,b2):.0%}")
    # Expected:
    # After Task 1 + prune: acc=97%, free params: 37/48
    # After Task 2: acc_task1=97%, acc_task2=90%
