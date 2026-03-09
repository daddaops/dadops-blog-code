"""
Mondrian Conformal Prediction — per-group coverage for imbalanced classes.
From: https://dadops.co/blog/conformal-prediction-from-scratch/
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # Imbalanced 3-class problem: 70% / 20% / 10% split
    np.random.seed(42)
    n_samples = [700, 200, 100]
    centers = np.array([[0, 2], [-2, -1], [2, -1]])
    X = np.vstack([centers[c] + np.random.randn(n, 2) * 0.7 for c, n in enumerate(n_samples)])
    y = np.concatenate([np.full(n, c) for c, n in enumerate(n_samples)])

    # Shuffle and split
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    X_tr, X_cal, X_te = X[:600], X[600:800], X[800:]
    y_tr, y_cal, y_te = y[:600], y[600:800], y[800:]

    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0)
    clf.fit(X_tr, y_tr)
    alpha = 0.10

    # Standard conformal: one threshold for all classes
    cal_probs = clf.predict_proba(X_cal)
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(cal_scores)
    q_global = np.quantile(cal_scores, np.ceil((1-alpha)*(n+1))/n, method="higher")

    # Mondrian conformal: separate threshold per class
    q_per_class = {}
    for c in range(3):
        mask = y_cal == c
        sc = cal_scores[mask]
        nc = len(sc)
        q_per_class[c] = np.quantile(sc, np.ceil((1-alpha)*(nc+1))/nc, method="higher")

    # Evaluate per-class coverage
    te_probs = clf.predict_proba(X_te)
    for c in range(3):
        mask = y_te == c
        if mask.sum() == 0:
            continue
        # Standard: same threshold
        std_cov = np.mean(1 - te_probs[mask, c] <= q_global)
        # Mondrian: per-class threshold
        mon_cov = np.mean(1 - te_probs[mask, c] <= q_per_class[c])
        print(f"Class {c} (n={mask.sum()}): Standard={std_cov:.0%}, Mondrian={mon_cov:.0%}")
