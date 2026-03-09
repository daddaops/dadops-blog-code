"""Block 5: Learning without Forgetting (LwF) — self-distillation from snapshot."""
import numpy as np
from shared import make_task, sigmoid, train_mlp, accuracy

def train_lwf(X_new, y_new, W1, b1, W2, b2, snap_W1, snap_b1,
              snap_W2, snap_b2, alpha=1.0, epochs=200, lr=0.05):
    """Learning without Forgetting: distill from past-self snapshot."""
    for _ in range(epochs):
        # Current model forward pass
        h = np.maximum(0, X_new @ W1 + b1)
        out = sigmoid(h @ W2 + b2).ravel()
        # Snapshot (old model) predictions on the SAME new-task data
        h_old = np.maximum(0, X_new @ snap_W1 + snap_b1)
        out_old = sigmoid(h_old @ snap_W2 + snap_b2).ravel()
        # Hard loss (new task) + distillation loss (match old predictions)
        hard_err = out - y_new
        # Soft distillation: MSE between current and snapshot outputs
        soft_err = out - out_old
        err = hard_err + alpha * soft_err
        # Backprop combined error
        dW2 = h.T @ err.reshape(-1,1) / len(y_new)
        db2 = err.mean()
        dh = err.reshape(-1,1) * W2.T * (h > 0)
        dW1 = X_new.T @ dh / len(y_new)
        db1 = dh.mean(axis=0)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    return W1, b1, W2, b2

if __name__ == "__main__":
    X1, y1 = make_task([-2, -2], [2, 2], seed=42)
    X2, y2 = make_task([-2, 2], [2, -2], seed=99)

    # Train with LwF
    rng = np.random.RandomState(0)
    W1 = rng.randn(2,8)*0.3; b1 = np.zeros(8)
    W2 = rng.randn(8,1)*0.3; b2 = np.zeros(1)
    W1, b1, W2, b2 = train_mlp(X1, y1, W1, b1, W2, b2)
    snap = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    W1, b1, W2, b2 = train_lwf(X2, y2, W1, b1, W2, b2, *snap, alpha=2.0)
    print(f"LwF: acc_task1={accuracy(X1,y1,W1,b1,W2,b2):.0%}, "
          f"acc_task2={accuracy(X2,y2,W1,b1,W2,b2):.0%}")
    # Expected: LwF: acc_task1=82%, acc_task2=88%
