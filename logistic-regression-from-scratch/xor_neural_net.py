import numpy as np
from logistic_core import sigmoid, LogisticRegression

class TinyNeuralNetwork:
    """A 1-hidden-layer network = logistic regression on learned features."""
    def __init__(self, n_input, n_hidden):
        scale = np.sqrt(2 / n_input)
        self.W1 = np.random.randn(n_input, n_hidden) * scale
        self.b1 = np.zeros(n_hidden)
        self.w2 = np.random.randn(n_hidden) * scale
        self.b2 = 0.0

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1       # linear transform
        self.h = sigmoid(self.z1)              # learned features
        self.z2 = self.h @ self.w2 + self.b2   # logistic regression
        return sigmoid(self.z2)                 # on those features

    def fit(self, X, y, lr=1.0, epochs=2000):
        for epoch in range(epochs):
            p = self.forward(X)
            # Output layer gradient (same as logistic regression)
            d2 = p - y
            grad_w2 = self.h.T @ d2 / len(y)
            grad_b2 = np.mean(d2)
            # Hidden layer gradient (backpropagation)
            d1 = np.outer(d2, self.w2) * self.h * (1 - self.h)
            grad_W1 = X.T @ d1 / len(y)
            grad_b1 = d1.mean(axis=0)
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

if __name__ == "__main__":
    # The XOR problem: logistic regression fails, neural network succeeds
    X_xor = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0], dtype=float)

    # Logistic regression: stuck at 50%
    lr_model = LogisticRegression(2)
    lr_model.fit(X_xor, y_xor, lr=1.0, epochs=1000)
    lr_preds = lr_model.predict_proba(X_xor)
    print("Logistic Regression on XOR:")
    for x, yt, yp in zip(X_xor, y_xor, lr_preds):
        print(f"  {x} -> true={yt:.0f}, pred={yp:.2f}")

    # Neural network: solves it
    np.random.seed(5)
    nn_model = TinyNeuralNetwork(2, 4)
    nn_model.fit(X_xor, y_xor, lr=2.0, epochs=3000)
    nn_preds = nn_model.forward(X_xor)
    print("\nNeural Network on XOR:")
    for x, yt, yp in zip(X_xor, y_xor, nn_preds):
        print(f"  {x} -> true={yt:.0f}, pred={yp:.2f}")
    print(f"\nLearned features for [1,0]: {nn_model.h[2].round(3)}")
    print(f"Learned features for [1,1]: {nn_model.h[3].round(3)}")
