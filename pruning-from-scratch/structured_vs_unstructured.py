"""Structured vs unstructured pruning comparison.

Unstructured: zeros individual weights (same shape, sparse).
Structured: removes entire neurons (smaller shape, dense).
"""
import numpy as np
from helpers import make_spiral_data, train_mlp, evaluate

def unstructured_prune(W, sparsity):
    """Zero individual weights by magnitude — sparse but same shape."""
    all_w = np.concatenate([w.flatten() for w in W])
    threshold = np.percentile(np.abs(all_w), sparsity * 100)
    return [np.where(np.abs(w) >= threshold, w, 0.0) for w in W]

def structured_prune(W, b, sparsity):
    """Remove entire neurons — actually shrinks the matrices."""
    W_new, b_new = list(W), list(b)
    for layer in range(len(W) - 1):
        n_neurons = W_new[layer].shape[1]
        n_keep = max(1, int(n_neurons * (1 - sparsity)))
        neuron_scores = np.linalg.norm(W_new[layer], axis=0)
        keep_idx = np.argsort(neuron_scores)[-n_keep:]
        keep_idx.sort()
        W_new[layer] = W_new[layer][:, keep_idx]
        b_new[layer] = b_new[layer][keep_idx]
        W_new[layer + 1] = W_new[layer + 1][keep_idx, :]
    return W_new, b_new

def count_params(W):
    return sum(np.count_nonzero(w) for w in W)

def count_dense_params(W):
    return sum(w.size for w in W)

X, y = make_spiral_data()
W, b = train_mlp(X, y)

W_unst = unstructured_prune(W, 0.5)
W_stru, b_stru = structured_prune(W, b, 0.5)

print("Method        | Dense shape params | Non-zero | Accuracy")
print("--------------|-------------------|----------|--------")
print(f"Original      | {count_dense_params(W):17d} | {count_params(W):8d} | {evaluate(W, b, X, y):.1%}")
print(f"Unstructured  | {count_dense_params(W_unst):17d} | {count_params(W_unst):8d} | {evaluate(W_unst, b, X, y):.1%}")
print(f"Structured    | {count_dense_params(W_stru):17d} | {count_params(W_stru):8d} | {evaluate(W_stru, b_stru, X, y):.1%}")
