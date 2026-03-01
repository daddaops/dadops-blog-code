"""
PGD (Projected Gradient Descent): iterated FGSM attack.

Trains the same multi-class MLP as fgsm_multiclass.py, then compares
single-step FGSM vs multi-step PGD attack success rates.

Requires: numpy

From: https://dadops.dev/blog/adversarial-examples-from-scratch/
"""

import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    # Build the same dataset and model as fgsm_multiclass.py
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
        dx = (dh @ W1.T).flatten()
        return x + eps * np.sign(dx)

    def pgd_attack(x, label, eps, steps=20, step_size=None):
        """Projected Gradient Descent: iterated FGSM with projection."""
        if step_size is None:
            step_size = eps / 5

        # Start from random point in epsilon-ball
        x_adv = x + np.random.uniform(-eps, eps, x.shape)
        x_adv = np.clip(x_adv, x - eps, x + eps)
        losses = []

        for _ in range(steps):
            z1 = x_adv.reshape(1, -1) @ W1 + b1
            h = np.maximum(0, z1)
            logits = h @ W2 + b2
            probs = softmax(logits).flatten()
            losses.append(-np.log(probs[label] + 1e-10))

            one_hot = np.zeros((1, 3));  one_hot[0, label] = 1
            dz2 = probs.reshape(1, -1) - one_hot
            dh = dz2 @ W2.T;  dh[h == 0] = 0
            dx = (dh @ W1.T).flatten()

            x_adv = x_adv + step_size * np.sign(dx)      # FGSM step
            x_adv = np.clip(x_adv, x - eps, x + eps)     # project onto ball

        return x_adv, losses

    # Compare FGSM vs PGD at the same epsilon
    np.random.seed(42)
    eps = 0.15
    fgsm_flips, pgd_flips = 0, 0
    n_test = 50
    for i in range(n_test):
        idx = i * 6
        x_adv_f = fgsm(X[idx], y[idx], eps)
        if softmax((np.maximum(0, x_adv_f.reshape(1,-1) @ W1 + b1) @ W2 + b2)).argmax() != y[idx]:
            fgsm_flips += 1

        x_adv_p, _ = pgd_attack(X[idx], y[idx], eps)
        if softmax((np.maximum(0, x_adv_p.reshape(1,-1) @ W1 + b1) @ W2 + b2)).argmax() != y[idx]:
            pgd_flips += 1

    print(f"Attack success rate at eps={eps}:")
    print(f"  FGSM (1 step):   {fgsm_flips}/{n_test} = {100*fgsm_flips/n_test:.0f}%")
    print(f"  PGD  (20 steps): {pgd_flips}/{n_test} = {100*pgd_flips/n_test:.0f}%")

    # Show loss climbing during PGD
    _, losses = pgd_attack(X[0], y[0], eps=0.15, steps=20)
    print(f"\nPGD loss trajectory: {losses[0]:.3f} -> {losses[-1]:.3f} (grew {losses[-1]/max(losses[0],1e-6):.1f}x)")
