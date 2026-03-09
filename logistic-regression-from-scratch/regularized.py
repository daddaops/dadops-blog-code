import numpy as np
from logistic_core import sigmoid

class RegularizedLogisticRegression:
    def __init__(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def fit(self, X, y, lr=0.1, epochs=300, lam=0.1, penalty="l2"):
        for epoch in range(epochs):
            p = self.predict_proba(X)
            grad_w = X.T @ (p - y) / len(y)
            grad_b = np.mean(p - y)
            # Add regularization gradient (not applied to bias)
            if penalty == "l2":
                grad_w += 2 * lam * self.w
            elif penalty == "l1":
                grad_w += lam * np.sign(self.w)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

if __name__ == "__main__":
    # High-dimensional example: 100 features, only 5 are relevant
    np.random.seed(7)
    n_samples, n_features, n_relevant = 80, 100, 5
    X_train = np.random.randn(n_samples, n_features)
    true_w = np.zeros(n_features)
    true_w[:n_relevant] = np.random.randn(n_relevant) * 3
    y_train = (sigmoid(X_train @ true_w) > 0.5).astype(float)

    # L1 finds the sparse solution
    model_l1 = RegularizedLogisticRegression(n_features)
    model_l1.fit(X_train, y_train, lr=0.05, epochs=500, lam=0.05, penalty="l1")
    nonzero = np.sum(np.abs(model_l1.w) > 0.01)
    print(f"L1: {nonzero} non-zero weights (true: {n_relevant})")
    print(f"Top features: {np.argsort(np.abs(model_l1.w))[-5:]}")
