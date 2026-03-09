"""WANDA pruning: weight magnitude x activation norm.

Prunes weights based on both their magnitude AND how active
their input features are on calibration data.
"""
import numpy as np
from helpers import make_spiral_data, train_mlp, evaluate

def wanda_prune(W, X_calib, sparsity):
    """Wanda-style pruning: |weight| x ||activation||_2.

    W: weight matrix (in_features x out_features)
    X_calib: calibration activations (n_samples x in_features)
    """
    activation_norms = np.linalg.norm(X_calib, axis=0)
    scores = np.abs(W) * activation_norms[:, None]
    W_pruned = W.copy()
    for row in range(W.shape[1]):
        row_scores = scores[:, row]
        threshold = np.percentile(row_scores, sparsity * 100)
        W_pruned[:, row] = np.where(row_scores >= threshold, W[:, row], 0.0)
    return W_pruned

# Demo: compare magnitude vs WANDA pruning on the first layer
X, y = make_spiral_data()
W, b = train_mlp(X, y)

# Magnitude pruning on first layer
threshold = np.percentile(np.abs(W[0]), 50 * 100 / 100)
W0_mag = np.where(np.abs(W[0]) >= np.percentile(np.abs(W[0]), 50), W[0], 0.0)

# WANDA pruning on first layer using training data as calibration
W0_wanda = wanda_prune(W[0], X, 0.5)

W_mag = [W0_mag] + W[1:]
W_wan = [W0_wanda] + W[1:]

print(f"First-layer pruning at 50% sparsity:")
print(f"  Magnitude:  {np.count_nonzero(W0_mag)} / {W0_mag.size} non-zero, acc={evaluate(W_mag, b, X, y):.1%}")
print(f"  WANDA:      {np.count_nonzero(W0_wanda)} / {W0_wanda.size} non-zero, acc={evaluate(W_wan, b, X, y):.1%}")
