"""
FGSM on a multi-class classifier: attack success rate vs epsilon.

Trains a 64->32->3 MLP on synthetic 8x8 "image" patterns
(horizontal bar, vertical bar, diagonal), then measures how
attack success scales with epsilon.

Requires: numpy

From: https://dadops.dev/blog/adversarial-examples-from-scratch/
"""

import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    np.random.seed(42)
    n_per_class, d = 100, 64  # 8x8 "images"
    X, y = [], []
    for i in range(n_per_class):
        img = np.random.randn(d) * 0.1
        img[24:40] += 1.0               # rows 3-4 bright (horizontal bar)
        X.append(img); y.append(0)

        img = np.random.randn(d) * 0.1
        img[3::8] += 1.0                # column 3 bright (vertical bar)
        X.append(img); y.append(1)

        img = np.random.randn(d) * 0.1
        for k in range(8): img[k*8 + k] += 1.0  # diagonal
        X.append(img); y.append(2)

    X, y = np.array(X), np.array(y)

    # Train 64 -> 32 -> 3 MLP
    np.random.seed(7)
    W1 = np.random.randn(64, 32) * np.sqrt(2 / 64)
    b1 = np.zeros(32)
    W2 = np.random.randn(32, 3) * np.sqrt(2 / 32)
    b2 = np.zeros(3)

    for _ in range(500):
        h = np.maximum(0, X @ W1 + b1)
        probs = softmax(h @ W2 + b2)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1
        dz2 = (probs - one_hot) / len(X)
        W2 -= 1.0 * h.T @ dz2;   b2 -= 1.0 * dz2.sum(axis=0)
        dh = dz2 @ W2.T;          dh[h == 0] = 0
        W1 -= 1.0 * X.T @ dh;     b1 -= 1.0 * dh.sum(axis=0)

    def fgsm(x, label, eps):
        z1 = x.reshape(1, -1) @ W1 + b1
        h = np.maximum(0, z1)
        probs = softmax(h @ W2 + b2)
        one_hot = np.zeros((1, 3));  one_hot[0, label] = 1
        dz2 = probs - one_hot
        dh = dz2 @ W2.T;  dh[h == 0] = 0
        dx = (dh @ W1.T).flatten()           # gradient w.r.t. input
        return x + eps * np.sign(dx)         # FGSM perturbation

    # Attack success rate vs epsilon
    for eps in [0.01, 0.05, 0.1, 0.2, 0.3]:
        flipped = 0
        for i in range(len(X)):
            x_adv = fgsm(X[i], y[i], eps)
            pred = softmax((np.maximum(0, x_adv.reshape(1,-1) @ W1 + b1) @ W2 + b2)).argmax()
            if pred != y[i]:
                flipped += 1
        print(f"eps={eps:.2f}: {flipped}/{len(X)} flipped ({100*flipped/len(X):.0f}% attack success)")
