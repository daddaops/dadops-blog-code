import numpy as np

def softmax(Z):
    Z_shifted = Z - Z.max(axis=1, keepdims=True)  # numerical stability
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, n_features, n_classes):
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def fit(self, X, y_onehot, lr=0.1, epochs=300):
        n = len(X)
        for epoch in range(epochs):
            probs = self.predict_proba(X)
            residual = probs - y_onehot  # same elegant gradient
            self.W -= lr * (X.T @ residual) / n
            self.b -= lr * residual.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

if __name__ == "__main__":
    # Three-class dataset
    np.random.seed(99)
    centers = [[-2, 0], [2, 0], [0, 3]]
    X_parts, y_parts = [], []
    for i, c in enumerate(centers):
        X_parts.append(np.random.randn(40, 2) * 0.8 + c)
        y_parts.append(np.full(40, i))
    X_multi = np.vstack(X_parts)
    y_multi = np.concatenate(y_parts)
    y_onehot = np.eye(3)[y_multi]

    model_mc = SoftmaxRegression(n_features=2, n_classes=3)
    model_mc.fit(X_multi, y_onehot, lr=0.3, epochs=500)
    accuracy = np.mean(model_mc.predict(X_multi) == y_multi)
    print(f"3-class accuracy: {accuracy:.1%}")
    print(f"Class probabilities for [0, 1.5]:")
    print(f"  {model_mc.predict_proba(np.array([[0, 1.5]]))[0].round(3)}")
