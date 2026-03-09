"""Gradual pruning with cubic sparsity schedule.

Prunes during training rather than after, giving the network
time to adapt to each round of weight removal.
"""
import numpy as np
from helpers import make_spiral_data, softmax, train_mlp, evaluate, magnitude_prune

def cubic_sparsity(step, total_steps, target, start_step=0):
    """Cubic sparsity schedule: slow start, fast middle, gentle finish."""
    if step < start_step:
        return 0.0
    progress = min(1.0, (step - start_step) / max(1, total_steps - start_step))
    return target * (1 - (1 - progress) ** 3)

def train_gradual_pruning(X, y, target_sparsity=0.9, lr=0.05, epochs=400,
                          prune_start=50, prune_freq=10):
    """Train with gradual magnitude pruning on a cubic schedule."""
    np.random.seed(42)
    dims = [2, 64, 32, 3]
    W, b = [], []
    for i in range(len(dims) - 1):
        W.append(np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i]))
        b.append(np.zeros(dims[i+1]))
    mask = [np.ones_like(w) for w in W]
    prune_end = int(epochs * 0.8)
    n_classes = y.max() + 1
    for epoch in range(epochs):
        for i in range(len(W)):
            W[i] *= mask[i]
        if epoch >= prune_start and epoch <= prune_end and epoch % prune_freq == 0:
            current_sparsity = cubic_sparsity(epoch, prune_end, target_sparsity, prune_start)
            all_w = np.concatenate([(w * m).flatten() for w, m in zip(W, mask)])
            alive = np.abs(all_w[all_w != 0])
            if len(alive) > 0:
                threshold = np.percentile(alive, current_sparsity * 100)
                for i in range(len(mask)):
                    mask[i] = (np.abs(W[i]) >= threshold).astype(float)
        h = X
        activations = [h]
        for i in range(len(W) - 1):
            h = np.maximum(0, h @ W[i] + b[i])
            activations.append(h)
        logits = h @ W[-1] + b[-1]
        probs = softmax(logits)
        dz = probs.copy()
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1
        dz -= one_hot
        dz /= len(y)
        for i in range(len(W) - 1, -1, -1):
            dW = activations[i].T @ dz
            db = dz.sum(axis=0)
            if i > 0:
                dz = (dz @ W[i].T) * (activations[i] > 0)
            W[i] -= lr * (dW * mask[i])
            b[i] -= lr * db
    return W, b, mask

X, y = make_spiral_data()

W_dense, b_dense = train_mlp(X, y)
W_oneshot = magnitude_prune(W_dense, 0.9)
acc_oneshot = evaluate(W_oneshot, b_dense, X, y)

W_gradual, b_gradual, _ = train_gradual_pruning(X, y, target_sparsity=0.9)
acc_gradual = evaluate(W_gradual, b_gradual, X, y)

print(f"At 90% sparsity:")
print(f"  One-shot pruning:  {acc_oneshot:.1%}")
print(f"  Gradual pruning:   {acc_gradual:.1%}")
print(f"  Dense baseline:    {evaluate(W_dense, b_dense, X, y):.1%}")
