"""
FGSM on a binary classifier: the basic adversarial attack.

Trains a 2-layer MLP on 2D ring data, then uses FGSM to craft
an adversarial perturbation that flips a class-1 point to class-0.

Requires: numpy

From: https://dadops.dev/blog/adversarial-examples-from-scratch/
"""

import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    t = np.random.uniform(0, 2 * np.pi, n)
    r0 = 1.0 + np.random.randn(n // 2) * 0.15   # inner ring -> class 0
    r1 = 2.5 + np.random.randn(n // 2) * 0.15   # outer ring -> class 1
    X = np.vstack([np.c_[r0 * np.cos(t[:n//2]), r0 * np.sin(t[:n//2])],
                   np.c_[r1 * np.cos(t[n//2:]), r1 * np.sin(t[n//2:])]])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    relu = lambda z: np.maximum(0, z)
    sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Train a 2 -> 16 -> 1 MLP
    np.random.seed(7)
    W1 = np.random.randn(2, 16) * 0.5
    b1 = np.zeros(16)
    W2 = np.random.randn(16, 1) * 0.5
    b2 = np.zeros(1)

    for _ in range(2000):
        h = relu(X @ W1 + b1)
        p = sigmoid(h @ W2 + b2).ravel()
        err = (p - y).reshape(-1, 1)
        dW2 = h.T @ err / n;  db2 = err.mean()
        dh = err @ W2.T;      dh[h == 0] = 0
        dW1 = X.T @ dh / n;   db1 = dh.mean(axis=0)
        W1 -= 0.5 * dW1;  b1 -= 0.5 * db1
        W2 -= 0.5 * dW2;  b2 -= 0.5 * db2

    # Pick a class-1 point (outer ring)
    x_test = np.array([[2.3, 0.5]])
    h_t = relu(x_test @ W1 + b1)
    conf = sigmoid(h_t @ W2 + b2).item()
    print(f"Original: class 1, confidence = {conf:.3f}")

    # Compute gradient of loss w.r.t. INPUT via manual backprop
    z1 = x_test @ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    p = sigmoid(z2).item()

    dp = -1.0 / (p + 1e-10)           # dL/dp for L = -log(p)
    dz2 = dp * p * (1 - p)            # sigmoid backward
    dh1 = dz2 * W2.T                  # linear backward
    dz1 = dh1 * (z1 > 0)              # ReLU backward
    dx = dz1 @ W1.T                   # input gradient

    # FGSM: perturb in the direction that INCREASES loss
    epsilon = 0.5
    x_adv = x_test + epsilon * np.sign(dx)

    h_a = relu(x_adv @ W1 + b1)
    conf_adv = sigmoid(h_a @ W2 + b2).item()
    pred_adv = 1 if conf_adv > 0.5 else 0
    print(f"Adversarial: class {pred_adv}, confidence = {max(conf_adv, 1-conf_adv):.3f}")
    print(f"Perturbation: ({x_test[0,0]:.1f}, {x_test[0,1]:.1f}) -> ({x_adv[0,0]:.2f}, {x_adv[0,1]:.2f})")
    print(f"L_inf norm of perturbation: {np.abs(x_adv - x_test).max():.2f}")
