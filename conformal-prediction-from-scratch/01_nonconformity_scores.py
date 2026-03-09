"""
Nonconformity Scores — computing calibration scores from softmax outputs.
From: https://dadops.co/blog/conformal-prediction-from-scratch/
"""
import numpy as np

if __name__ == "__main__":
    # Simulate a 5-class classifier's softmax outputs on 500 calibration points
    np.random.seed(42)
    n_cal = 500
    n_classes = 5
    true_labels = np.random.randint(0, n_classes, n_cal)

    # Generate fake softmax outputs (model is decent but not perfect)
    logits = np.random.randn(n_cal, n_classes)
    logits[np.arange(n_cal), true_labels] += 2.0  # boost true class
    softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # Nonconformity scores: 1 - P(true class)
    scores = 1 - softmax[np.arange(n_cal), true_labels]

    print(f"Score statistics:")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  90th percentile: {np.quantile(scores, 0.9):.3f}")
    print(f"  95th percentile: {np.quantile(scores, 0.95):.3f}")

    # The quantile threshold for alpha=0.1 (90% coverage)
    alpha = 0.1
    q_hat = np.quantile(scores, np.ceil((1 - alpha) * (n_cal + 1)) / n_cal,
                        method="higher")
    print(f"\nConformal quantile (alpha={alpha}): {q_hat:.3f}")
    print(f"  Include label y if 1 - f(x)_y <= {q_hat:.3f}")
    print(f"  Equivalently: include if f(x)_y >= {1 - q_hat:.3f}")
