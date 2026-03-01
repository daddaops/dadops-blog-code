"""
Adversarial training: fighting fire with fire.

Compares a standard-trained MLP vs an adversarially-trained MLP.
The adversarially-trained model sacrifices some clean accuracy
for dramatically improved robustness against FGSM attacks.

Requires: numpy

From: https://dadops.dev/blog/adversarial-examples-from-scratch/
"""

import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    # Build the same dataset as fgsm_multiclass.py
    np.random.seed(42)
    n_per_class, d = 100, 64
    X, y = [], []
    for i in range(n_per_class):
        img = np.random.randn(d) * 0.1
        img[24:40] += 1.0
        X.append(img); y.append(0)
        img = np.random.randn(d) * 0.1
        img[3::8] += 1.0
        X.append(img); y.append(1)
        img = np.random.randn(d) * 0.1
        for k in range(8): img[k*8 + k] += 1.0
        X.append(img); y.append(2)
    X, y = np.array(X), np.array(y)

    def train_classifier(X, y, epochs=500, adversarial=False, eps=0.15, lr=1.0):
        """Train a 64->32->3 MLP, optionally with adversarial training."""
        np.random.seed(7)
        W1 = np.random.randn(64, 32) * np.sqrt(2 / 64)
        b1 = np.zeros(32)
        W2 = np.random.randn(32, 3) * np.sqrt(2 / 32)
        b2 = np.zeros(3)

        for _ in range(epochs):
            X_batch = X.copy()

            if adversarial:
                # Inner max: FGSM on current batch
                h = np.maximum(0, X @ W1 + b1)
                probs = softmax(h @ W2 + b2)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(y)), y] = 1
                dz = probs - one_hot
                dh = dz @ W2.T;  dh[h == 0] = 0
                dx = dh @ W1.T
                X_batch = X + eps * np.sign(dx)    # adversarial inputs

            # Outer min: standard training on (possibly adversarial) inputs
            h = np.maximum(0, X_batch @ W1 + b1)
            probs = softmax(h @ W2 + b2)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(y)), y] = 1
            dz = (probs - one_hot) / len(X)
            W2 -= lr * h.T @ dz;   b2 -= lr * dz.sum(axis=0)
            dh = dz @ W2.T;          dh[h == 0] = 0
            W1 -= lr * X_batch.T @ dh;  b1 -= lr * dh.sum(axis=0)

        return W1, b1, W2, b2

    def accuracy(W1, b1, W2, b2, X, y, eps=None):
        h = np.maximum(0, X @ W1 + b1)
        probs = softmax(h @ W2 + b2)
        clean = (probs.argmax(axis=1) == y).mean()
        if eps is None:
            return clean, 0.0
        # FGSM attack
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1
        dz = probs - one_hot
        dh = dz @ W2.T;  dh[h == 0] = 0
        dx = dh @ W1.T
        X_adv = X + eps * np.sign(dx)
        h_a = np.maximum(0, X_adv @ W1 + b1)
        robust = (softmax(h_a @ W2 + b2).argmax(axis=1) == y).mean()
        return clean, robust

    W1s, b1s, W2s, b2s = train_classifier(X, y, adversarial=False)
    W1r, b1r, W2r, b2r = train_classifier(X, y, adversarial=True, eps=0.3, lr=0.3, epochs=1000)

    split = int(0.7 * len(X))
    X_te, y_te = X[split:], y[split:]

    print(f"{'Model':>12s} | {'Clean':>7s} | {'eps=0.15':>8s} | {'eps=0.30':>8s} | {'eps=0.50':>8s}")
    print("-" * 60)
    for name, w1, b1, w2, b2 in [("Standard", W1s, b1s, W2s, b2s),
                                   ("Adversarial", W1r, b1r, W2r, b2r)]:
        clean, _ = accuracy(w1, b1, w2, b2, X_te, y_te)
        results = [f"{clean:.1%}"]
        for eps_eval in [0.15, 0.30, 0.50]:
            _, robust = accuracy(w1, b1, w2, b2, X_te, y_te, eps=eps_eval)
            results.append(f"{robust:.1%}")
        print(f"{name:>12s} | {' | '.join(f'{r:>7s}' for r in results)}")
