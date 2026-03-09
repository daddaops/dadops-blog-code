"""Full MLP forward + backward — 3-layer network trained on XOR
with manual backprop and numerical gradient verification."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class ManualMLP:
    """3-layer MLP with fully manual forward and backward passes."""

    def __init__(self):
        np.random.seed(3)
        # Layer 1: 2 -> 4 (hidden)
        self.W1 = np.random.randn(2, 4) * 0.5
        self.b1 = np.zeros((1, 4))
        # Layer 2: 4 -> 4 (hidden)
        self.W2 = np.random.randn(4, 4) * 0.5
        self.b2 = np.zeros((1, 4))
        # Layer 3: 4 -> 1 (output)
        self.W3 = np.random.randn(4, 1) * 0.5
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        """Forward pass — cache everything for backward."""
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)          # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)          # ReLU
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = sigmoid(self.z3)                 # Sigmoid output
        return self.a3

    def loss(self, y_pred, y_true):
        """Binary cross-entropy loss."""
        eps = 1e-12
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )

    def backward(self, y_true):
        """Backward pass — chain rule, layer by layer, right to left."""
        m = y_true.shape[0]

        # Step 1: Loss gradient -> sigmoid output
        # d(BCE)/d(a3) combined with sigmoid gives:
        dz3 = (self.a3 - y_true) / m

        # Step 2: Layer 3 weight gradients
        self.dW3 = self.a2.T @ dz3
        self.db3 = np.sum(dz3, axis=0, keepdims=True)

        # Step 3: Propagate to layer 2 activation
        da2 = dz3 @ self.W3.T

        # Step 4: Through ReLU (binary gate)
        dz2 = da2 * (self.z2 > 0).astype(float)

        # Step 5: Layer 2 weight gradients
        self.dW2 = self.a1.T @ dz2
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        # Step 6: Propagate to layer 1 activation
        da1 = dz2 @ self.W2.T

        # Step 7: Through ReLU
        dz1 = da1 * (self.z1 > 0).astype(float)

        # Step 8: Layer 1 weight gradients
        self.dW1 = self.X.T @ dz1
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def step(self, lr=0.5):
        """Gradient descent update."""
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W3 -= lr * self.dW3
        self.b3 -= lr * self.db3

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

net = ManualMLP()

# Verify gradients numerically before training
y_pred = net.forward(X)
L = net.loss(y_pred, y)
net.backward(y)

def loss_for_param(param_name, net, X, y):
    """Compute loss with a given parameter."""
    y_pred = net.forward(X)
    return net.loss(y_pred, y)

# Check W1 gradient numerically
eps = 1e-5
dW1_num = np.zeros_like(net.W1)
for i in range(net.W1.shape[0]):
    for j in range(net.W1.shape[1]):
        old = net.W1[i, j]
        net.W1[i, j] = old + eps
        lp = loss_for_param('W1', net, X, y)
        net.W1[i, j] = old - eps
        lm = loss_for_param('W1', net, X, y)
        net.W1[i, j] = old
        dW1_num[i, j] = (lp - lm) / (2 * eps)

max_diff = np.max(np.abs(net.dW1 - dW1_num))
print(f"W1 gradient check: max diff = {max_diff:.2e}")

# Train on XOR
net = ManualMLP()
for epoch in range(2000):
    y_pred = net.forward(X)
    L = net.loss(y_pred, y)
    net.backward(y)
    net.step(lr=1.0)
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}: loss={L:.4f}")

# Final predictions
y_pred = net.forward(X)
print(f"\nFinal predictions:")
for i in range(4):
    print(f"  {X[i]} -> {y_pred[i,0]:.4f} (target: {y[i,0]:.0f})")
