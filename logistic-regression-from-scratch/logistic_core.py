import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class LogisticRegression:
    def __init__(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def loss(self, X, y):
        p = self.predict_proba(X)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y, lr=0.1, epochs=200):
        for epoch in range(epochs):
            p = self.predict_proba(X)
            residual = p - y  # the beautiful gradient
            grad_w = X.T @ residual / len(y)
            grad_b = np.mean(residual)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

if __name__ == "__main__":
    # Generate a 2D dataset
    np.random.seed(42)
    X_pos = np.random.randn(50, 2) + [2, 2]
    X_neg = np.random.randn(50, 2) + [-1, -1]
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*50 + [0]*50, dtype=float)

    model = LogisticRegression(n_features=2)
    print(f"Before training: loss = {model.loss(X, y):.4f}")
    model.fit(X, y, lr=0.5, epochs=300)
    print(f"After training:  loss = {model.loss(X, y):.4f}")
    print(f"Accuracy: {np.mean(model.predict(X) == y):.1%}")
    print(f"Weights: {model.w}, Bias: {model.b:.3f}")
