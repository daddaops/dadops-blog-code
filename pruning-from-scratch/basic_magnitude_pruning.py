"""Global magnitude pruning at various sparsity levels.

Trains a 2->64->32->3 MLP on spiral data, then prunes
weights by magnitude and measures accuracy degradation.
"""
import numpy as np
from helpers import make_spiral_data, train_mlp, evaluate, magnitude_prune

X, y = make_spiral_data()
W, b = train_mlp(X, y)
base_acc = evaluate(W, b, X, y)
total_params = sum(w.size for w in W)

print(f"Dense network: {total_params} params, accuracy: {base_acc:.1%}")
print(f"\nSparsity | Non-zero | Accuracy")
print(f"---------|----------|--------")
for s in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    W_pruned = magnitude_prune(W, s)
    acc = evaluate(W_pruned, b, X, y)
    nnz = sum(np.count_nonzero(w) for w in W_pruned)
    print(f"  {s:4.0%}   |  {nnz:5d}   | {acc:.1%}")
