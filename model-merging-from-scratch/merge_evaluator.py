"""Merge evaluator: SLERP, DARE, and comparison across methods.

Implements spherical interpolation (SLERP), DARE random dropout,
and an evaluation harness comparing LERP, Task Arithmetic, TIES,
and DARE-TIES on multi-task data.
"""
import numpy as np
from ties_merge import ties_merge

def slerp(v1, v2, t):
    """Spherical linear interpolation between two vectors."""
    v1_n = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_n = v2 / (np.linalg.norm(v2) + 1e-10)
    omega = np.arccos(np.clip(np.dot(v1_n, v2_n), -1.0, 1.0))
    if omega < 1e-6:
        return (1 - t) * v1 + t * v2  # fallback to LERP for near-parallel
    return (np.sin((1-t)*omega)/np.sin(omega)) * v1 + (np.sin(t*omega)/np.sin(omega)) * v2

def dare_drop(tv, p=0.9, seed=None):
    """DARE: randomly zero elements, rescale survivors."""
    rng = np.random.RandomState(seed)
    mask = rng.binomial(1, 1-p, size=tv.shape).astype(float)
    return mask * tv / max(1-p, 1e-10)

def score_model(W, test_data):
    results = []
    for X, y in test_data:
        pred = (1 / (1 + np.exp(-X @ W))).flatten()
        acc = np.mean((pred > 0.5).astype(float) == y)
        results.append(acc)
    return results

def evaluate_merge(pretrained, task_vectors, test_data, method='lerp', **kwargs):
    """Evaluate a merging method. Returns per-task accuracies."""
    alpha = kwargs.get('alpha', 0.7)
    tvs = task_vectors

    if method == 'lerp':
        merged_tv = np.mean(tvs, axis=0)
    elif method == 'slerp' and len(tvs) == 2:
        # SLERP operates on full weights, not task vectors
        W_merged = slerp(pretrained + tvs[0], pretrained + tvs[1], 0.5)
        return score_model(W_merged, test_data)
    elif method == 'task_arith':
        merged_tv = np.sum(tvs, axis=0)
    elif method == 'ties':
        merged_tv = ties_merge([tv.flatten() for tv in tvs], trim_pct=kwargs.get('trim', 0.8))
        merged_tv = merged_tv.reshape(pretrained.shape)
    elif method == 'dare_ties':
        dared = [dare_drop(tv.flatten(), p=kwargs.get('drop', 0.9), seed=i) for i, tv in enumerate(tvs)]
        merged_tv = ties_merge(dared, trim_pct=kwargs.get('trim', 0.8))
        merged_tv = merged_tv.reshape(pretrained.shape)
    else:
        merged_tv = np.mean(tvs, axis=0)

    W_merged = pretrained + alpha * merged_tv
    return score_model(W_merged, test_data)


if __name__ == "__main__":
    # Build test data using same setup as task_arithmetic.py
    def sigmoid(z):
        return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

    def make_task_data(n, task_id):
        X = np.random.randn(n, 4)
        if task_id == 0:
            y = (X[:, 0] + X[:, 1] > 0).astype(float)
        elif task_id == 1:
            y = (X[:, 2] - X[:, 3] > 0.5).astype(float)
        else:
            y = (np.sin(X[:, 0]) + X[:, 2] > 0).astype(float)
        return X, y

    def train(X, y, W_init, lr=0.2, steps=500):
        W = W_init.copy()
        for _ in range(steps):
            pred = sigmoid(X @ W)
            grad = X.T @ (pred - y.reshape(-1, 1)) / len(X)
            W -= lr * grad
        return W

    np.random.seed(7)
    W_pretrained = np.random.randn(4, 1) * 0.3
    datasets = [make_task_data(300, i) for i in range(3)]
    W_ft = [train(X, y, W_pretrained) for X, y in datasets]
    task_vecs = np.array([W_ft[i] - W_pretrained for i in range(3)])

    # Generate test data
    np.random.seed(99)
    test_data = [make_task_data(200, i) for i in range(3)]

    # Compare methods
    print(f"{'Method':<14} {'Task0':>7} {'Task1':>7} {'Task2':>7} {'Combined':>10}")
    print("-" * 50)
    for method in ['lerp', 'task_arith', 'ties', 'dare_ties']:
        accs = evaluate_merge(W_pretrained, task_vecs, test_data, method=method)
        combined = np.mean(accs)
        print(f"{method:<14} {accs[0]:>6.1%} {accs[1]:>6.1%} {accs[2]:>6.1%} {combined:>9.1%}")
