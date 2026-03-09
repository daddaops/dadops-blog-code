"""
Split Conformal Prediction — the core algorithm for classification.
From: https://dadops.co/blog/conformal-prediction-from-scratch/
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # Generate synthetic 2D classification data (3 classes)
    np.random.seed(42)
    centers = np.array([[0, 2], [-2, -1], [2, -1]])
    X, y = [], []
    for c in range(3):
        pts = centers[c] + np.random.randn(300, 2) * 0.8
        X.append(pts)
        y.append(np.full(300, c))
    X, y = np.vstack(X), np.concatenate(y)

    # Split: 60% train, 20% calibration, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Train any classifier
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0)
    clf.fit(X_train, y_train)

    # Step 1: Compute calibration scores
    cal_probs = clf.predict_proba(X_cal)
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]

    # Step 2: Find conformal quantile
    alpha = 0.10
    n = len(cal_scores)
    q_hat = np.quantile(cal_scores, np.ceil((1 - alpha) * (n + 1)) / n,
                        method="higher")

    # Step 3: Build prediction sets for test data
    test_probs = clf.predict_proba(X_test)
    prediction_sets = []
    for i in range(len(X_test)):
        pset = [c for c in range(3) if 1 - test_probs[i, c] <= q_hat]
        prediction_sets.append(pset)

    # Verify coverage
    covered = sum(y_test[i] in prediction_sets[i] for i in range(len(y_test)))
    print(f"Target coverage: {1-alpha:.0%}")
    print(f"Empirical coverage: {covered/len(y_test):.1%}")
    print(f"Average set size: {np.mean([len(s) for s in prediction_sets]):.2f}")
    print(f"Singleton sets: {sum(len(s)==1 for s in prediction_sets)}/{len(y_test)}")
