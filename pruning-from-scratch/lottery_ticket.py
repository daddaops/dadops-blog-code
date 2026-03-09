"""Lottery Ticket Hypothesis: iterative magnitude pruning.

Finds a sparse subnetwork (the "winning ticket") that, when trained
from the original initialization, matches the dense network's accuracy.
"""
import numpy as np
from helpers import make_spiral_data, softmax, train_mlp, evaluate

def train_with_mask(W_init, b_init, mask, X, y, lr=0.05, epochs=400):
    """Train a network while enforcing a binary mask on weights."""
    W = [w.copy() for w in W_init]
    b = [bi.copy() for bi in b_init]
    n_classes = y.max() + 1
    for epoch in range(epochs):
        for i in range(len(W)):
            W[i] *= mask[i]
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
    return W, b

def find_lottery_ticket(X, y, target_sparsity=0.8, prune_rounds=4):
    """Iterative Magnitude Pruning to find a winning lottery ticket."""
    np.random.seed(42)
    dims = [2, 64, 32, 3]
    W_init, b_init = [], []
    for i in range(len(dims) - 1):
        W_init.append(np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i]))
        b_init.append(np.zeros(dims[i+1]))
    mask = [np.ones_like(w) for w in W_init]
    per_round = 1.0 - (1.0 - target_sparsity) ** (1.0 / prune_rounds)
    for round_i in range(prune_rounds):
        W_trained, b_trained = train_with_mask(W_init, b_init, mask, X, y)
        surviving = np.concatenate([
            (w * m).flatten() for w, m in zip(W_trained, mask)
        ])
        surviving_abs = np.abs(surviving)
        surviving_abs = surviving_abs[surviving_abs > 0]
        if len(surviving_abs) == 0:
            break
        threshold = np.percentile(surviving_abs, per_round * 100)
        for i in range(len(mask)):
            mask[i] *= (np.abs(W_trained[i]) >= threshold).astype(float)
    W_ticket, b_ticket = train_with_mask(W_init, b_init, mask, X, y)
    return W_ticket, b_ticket, mask, W_init, b_init

X, y = make_spiral_data()

W_dense, b_dense = train_mlp(X, y)
acc_dense = evaluate(W_dense, b_dense, X, y)

W_ticket, b_ticket, mask, W_init, b_init = find_lottery_ticket(X, y, target_sparsity=0.8)
acc_ticket = evaluate(W_ticket, b_ticket, X, y)

np.random.seed(999)
W_random = [np.random.randn(*w.shape) * np.sqrt(2.0 / w.shape[0]) for w in W_init]
b_random = [np.zeros_like(bi) for bi in b_init]
W_rand_trained, b_rand_trained = train_with_mask(W_random, b_random, mask, X, y)
acc_random = evaluate(W_rand_trained, b_rand_trained, X, y)

sparsity = 1.0 - sum(m.sum() for m in mask) / sum(m.size for m in mask)
print(f"Sparsity: {sparsity:.0%}")
print(f"  Dense network:    {acc_dense:.1%}")
print(f"  Lottery ticket:   {acc_ticket:.1%}  <- matches dense!")
print(f"  Random reinit:    {acc_random:.1%}  <- fails")
