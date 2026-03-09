"""
Adaptive Prediction Sets (APS) — smarter sets that adapt to difficulty.
From: https://dadops.co/blog/conformal-prediction-from-scratch/
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # Same data setup as before
    np.random.seed(42)
    centers = np.array([[0, 2], [-2, -1], [2, -1]])
    X = np.vstack([c + np.random.randn(300, 2) * 0.8 for c in centers])
    y = np.concatenate([np.full(300, c) for c in range(3)])
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5)

    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0)
    clf.fit(X_tr, y_tr)
    alpha = 0.10

    # APS calibration scores
    cal_probs = clf.predict_proba(X_cal)
    cal_scores_aps = []
    for i in range(len(y_cal)):
        sorted_idx = np.argsort(-cal_probs[i])  # descending probability
        cumsum = np.cumsum(cal_probs[i, sorted_idx])
        rank = np.where(sorted_idx == y_cal[i])[0][0]
        cal_scores_aps.append(cumsum[rank])  # cumulative prob up to true class

    cal_scores_aps = np.array(cal_scores_aps)
    n = len(cal_scores_aps)
    q_aps = np.quantile(cal_scores_aps, np.ceil((1-alpha)*(n+1))/n, method="higher")

    # APS prediction sets
    te_probs = clf.predict_proba(X_te)
    sets_aps = []
    for i in range(len(X_te)):
        sorted_idx = np.argsort(-te_probs[i])
        cumsum = np.cumsum(te_probs[i, sorted_idx])
        # Include classes until cumulative probability exceeds threshold
        k = np.searchsorted(cumsum, q_aps) + 1
        sets_aps.append(sorted_idx[:k].tolist())

    covered = sum(y_te[i] in sets_aps[i] for i in range(len(y_te)))
    print(f"APS coverage: {covered/len(y_te):.1%}")
    print(f"APS avg set size: {np.mean([len(s) for s in sets_aps]):.2f}")
    sizes = [len(s) for s in sets_aps]
    print(f"  Singletons: {sizes.count(1)}, Pairs: {sizes.count(2)}, Triples: {sizes.count(3)}")
