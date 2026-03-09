"""
Coverage-Efficiency Tradeoff — sweeping alpha to see prediction set behavior.
From: https://dadops.co/blog/conformal-prediction-from-scratch/
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # Build a full conformal pipeline with selective prediction
    np.random.seed(42)
    centers = np.array([[0, 2], [-2, -1], [2, -1]])
    X = np.vstack([c + np.random.randn(400, 2) * 0.9 for c in centers])
    y = np.concatenate([np.full(400, c) for c in range(3)])
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5)

    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0)
    clf.fit(X_tr, y_tr)

    # Sweep alpha to see the coverage-efficiency tradeoff
    for alpha in [0.01, 0.05, 0.10, 0.20, 0.50]:
        cal_probs = clf.predict_proba(X_cal)
        scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
        n = len(scores)
        q = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n, method="higher")

        te_probs = clf.predict_proba(X_te)
        sets = []
        for i in range(len(X_te)):
            pset = [c for c in range(3) if 1 - te_probs[i, c] <= q]
            sets.append(pset)

        coverage = np.mean([y_te[i] in sets[i] for i in range(len(y_te))])
        avg_size = np.mean([len(s) for s in sets])
        singletons = sum(len(s) == 1 for s in sets) / len(sets)
        print(f"alpha={alpha:.2f}: coverage={coverage:.0%}, "
              f"avg_size={avg_size:.2f}, singletons={singletons:.0%}")
