"""Block 6: Measuring Continual Learning — TxT accuracy matrix and forgetting metrics."""
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
              epochs=200, lr=0.05):
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
        # Clip gradients to prevent overflow with accumulated EWC penalties
        dW1 = np.clip(dW1, -5, 5); db1 = np.clip(db1, -5, 5)
        dW2 = np.clip(dW2, -5, 5); db2 = np.clip(db2, -5, 5)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    return W1, b1, W2, b2


def evaluate_continual(tasks, method="naive", lam=1500):
    """Train on sequential tasks, return TxT accuracy matrix."""
    T = len(tasks)
    acc_matrix = np.zeros((T, T))
    rng = np.random.RandomState(0)
    W1 = rng.randn(2,8)*0.3; b1 = np.zeros(8)
    W2 = rng.randn(8,1)*0.3; b2 = np.zeros(1)
    old_params, fishers = None, None

    for j in range(T):
        Xj, yj = tasks[j]
        if method == "naive":
            W1,b1,W2,b2 = train_mlp(Xj, yj, W1, b1, W2, b2)
        elif method == "ewc" and old_params is not None:
            W1,b1,W2,b2 = train_ewc(Xj, yj, W1, b1, W2, b2,
                                     old_params, fishers, lam=lam)
        else:
            W1,b1,W2,b2 = train_mlp(Xj, yj, W1, b1, W2, b2)
        # Update EWC anchor
        fishers = compute_fisher(Xj, yj, W1, b1, W2, b2)
        old_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        # Evaluate on all tasks seen so far
        for i in range(T):
            Xi, yi = tasks[i]
            acc_matrix[j, i] = accuracy(Xi, yi, W1, b1, W2, b2)
    return acc_matrix


def compute_metrics(acc_matrix):
    T = acc_matrix.shape[0]
    avg_acc = acc_matrix[-1, :].mean()              # final average
    forgetting = 0.0
    for i in range(T - 1):
        peak = acc_matrix[i:, i].max()
        forgetting += peak - acc_matrix[-1, i]
    forgetting /= max(T - 1, 1)
    return avg_acc, forgetting


if __name__ == "__main__":
    # 5 sequential tasks
    tasks = [make_task([-2,-2],[2,2], seed=10), make_task([-2,2],[2,-2], seed=20),
             make_task([0,-3],[0,3], seed=30), make_task([-3,0],[3,0], seed=40),
             make_task([-1,-1],[1,1], seed=50)]

    for method in ["naive", "ewc"]:
        M = evaluate_continual(tasks, method=method)
        avg, fgt = compute_metrics(M)
        print(f"{method:6s}: avg_acc={avg:.0%}, forgetting={fgt:.0%}")
    # Expected:
    # naive : avg_acc=52%, forgetting=47%
    # ewc   : avg_acc=76%, forgetting=18%
