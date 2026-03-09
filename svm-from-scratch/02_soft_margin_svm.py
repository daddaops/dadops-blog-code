import numpy as np


class SoftMarginSVM:
    """SVM with tunable C for the bias-variance tradeoff."""

    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            for i in range(n_samples):
                margin = y[i] * (X[i] @ self.w + self.b)

                if margin < 1:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b -= self.lr * (-self.C * y[i])
                else:
                    self.w -= self.lr * self.w

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


if __name__ == "__main__":
    # Add noise: move some points close to the boundary
    np.random.seed(42)
    X_pos = np.random.randn(50, 2) + np.array([1.5, 1.5])
    X_neg = np.random.randn(50, 2) + np.array([-1.5, -1.5])
    X_noisy = np.vstack([X_pos, X_neg])
    y_noisy = np.array([1]*50 + [-1]*50)

    for C_val in [100.0, 1.0, 0.01]:
        svm = SoftMarginSVM(C=C_val, lr=0.0005, epochs=500)
        svm.fit(X_noisy, y_noisy)
        preds = svm.predict(X_noisy)
        acc = np.mean(preds == y_noisy) * 100
        margin_width = 2.0 / (np.linalg.norm(svm.w) + 1e-8)
        print(f"C={C_val:>6}: accuracy={acc:.0f}%, margin width={margin_width:.2f}")
