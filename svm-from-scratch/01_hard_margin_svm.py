import numpy as np

class HardMarginSVM:
    """SVM trained via sub-gradient descent on the hinge loss."""

    def __init__(self, C=1000.0, lr=0.001, epochs=1000):
        self.C = C          # penalty for margin violations
        self.lr = lr         # learning rate
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                margin = y[i] * (X[i] @ self.w + self.b)

                if margin < 1:
                    # point violates margin: hinge loss gradient
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b -= self.lr * (-self.C * y[i])
                else:
                    # point is safely classified: only regularization gradient
                    self.w -= self.lr * self.w

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def support_vectors(self, X, y):
        margins = y * self.decision_function(X)
        # support vectors lie near the margin boundary (margin ~ 1)
        return X[margins <= 1.01]


if __name__ == "__main__":
    # Generate linearly separable data
    np.random.seed(42)
    X_pos = np.random.randn(30, 2) + np.array([2, 2])
    X_neg = np.random.randn(30, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*30 + [-1]*30)

    svm = HardMarginSVM(C=1000, lr=0.0005, epochs=500)
    svm.fit(X, y)
    print(f"Weight vector: [{svm.w[0]:.3f}, {svm.w[1]:.3f}]")
    print(f"Bias: {svm.b:.3f}")
    print(f"Support vectors found: {len(svm.support_vectors(X, y))}")
