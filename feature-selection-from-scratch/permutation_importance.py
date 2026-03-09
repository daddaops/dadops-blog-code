"""
Permutation Importance

Model-agnostic feature importance by measuring accuracy drop
when each feature is shuffled.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np
from dataset import make_dataset, sigmoid

X, y, names, n = make_dataset()

# Train/validation split (same as wrapper_methods.py)
idx = np.arange(n)
np.random.shuffle(idx)
Xtr, ytr = X[idx[:140]], y[idx[:140]]
Xval, yval = X[idx[140:]], y[idx[140:]]


def permutation_importance(X_val, y_val, predict_fn, n_repeats=20):
    """Permutation importance: accuracy drop when each feature is shuffled."""
    base_acc = (predict_fn(X_val) == y_val).mean()
    importances = np.zeros(X_val.shape[1])

    rng = np.random.RandomState(0)
    for j in range(X_val.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_acc = (predict_fn(X_perm) == y_val).mean()
            drops.append(base_acc - perm_acc)
        importances[j] = np.mean(drops)
    return importances


# Train a logistic regression on all features
w_full = np.zeros(10)
b_full = 0.0
for _ in range(500):
    p = sigmoid(Xtr @ w_full + b_full)
    w_full -= 0.1 * (Xtr.T @ (p - ytr) / len(ytr))
    b_full -= 0.1 * (p - ytr).mean()

predict_fn = lambda Xv: (sigmoid(Xv @ w_full + b_full) > 0.5).astype(int)

perm_imp = permutation_importance(Xval, yval, predict_fn)

print("Permutation importance ranking:")
for idx in np.argsort(-perm_imp):
    marker = " *" if "info" in names[idx] else ""
    print(f"  {names[idx]:<10s} drop = {perm_imp[idx]:+.3f}{marker}")
