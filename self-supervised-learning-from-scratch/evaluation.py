"""Linear probing and k-NN evaluation of SSL features."""
import numpy as np

# Synthetic data: 3 classes with clear cluster structure in 2D
np.random.seed(42)
n_per_class = 50
centers = np.array([[2, 2], [-2, 2], [0, -2.5]])
X = np.vstack([c + np.random.randn(n_per_class, 2) * 0.6 for c in centers])
y = np.repeat([0, 1, 2], n_per_class)

# Simulate a self-supervised encoder: 2D input -> 8D representations
W_enc = np.random.randn(2, 8) * 0.5
b_enc = np.random.randn(8) * 0.1
features = np.tanh(X @ W_enc + b_enc)

# Train/test split
perm = np.random.permutation(150)
X_tr, X_te = features[perm[:100]], features[perm[100:]]
y_tr, y_te = y[perm[:100]], y[perm[100:]]

# METHOD 1: Linear Probing -- freeze encoder, train linear head
def linear_probe(X_tr, y_tr, X_te, y_te, n_cls=3, lr=0.1, epochs=200):
    W = np.zeros((X_tr.shape[1], n_cls))
    for _ in range(epochs):
        logits = X_tr @ W
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        grad = X_tr.T @ (probs - np.eye(n_cls)[y_tr]) / len(y_tr)
        W -= lr * grad
    return np.mean(np.argmax(X_te @ W, axis=1) == y_te)

# METHOD 2: k-NN -- classify by majority vote of nearest neighbors
def knn_eval(X_tr, y_tr, X_te, y_te, k=5):
    correct = 0
    for i in range(len(X_te)):
        dists = np.linalg.norm(X_tr - X_te[i], axis=1)
        nearest_labels = y_tr[np.argsort(dists)[:k]]
        pred = np.argmax(np.bincount(nearest_labels, minlength=3))
        correct += (pred == y_te[i])
    return correct / len(y_te)

lp_acc = linear_probe(X_tr, y_tr, X_te, y_te)
knn_acc = knn_eval(X_tr, y_tr, X_te, y_te)

print(f"Linear probe accuracy: {lp_acc:.1%}")
print(f"k-NN (k=5) accuracy:   {knn_acc:.1%}")
print(f"\nThe encoder never saw labels during pre-training.")
print(f"High accuracy = SSL features captured the true class structure!")
