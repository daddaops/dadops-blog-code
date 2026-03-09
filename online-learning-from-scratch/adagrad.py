"""AdaGrad: per-coordinate adaptive learning rates for sparse data.

Compares uniform OGD vs AdaGrad on sparse online regression.
"""
import numpy as np

np.random.seed(2026)
T, d = 1000, 20
w_true = np.zeros(d)
w_true[0], w_true[5], w_true[15] = 2.0, -1.5, 1.0

# Online squared-loss regression with sparse features
w_uniform, w_ada = np.zeros(d), np.zeros(d)
G = np.zeros(d)  # AdaGrad accumulator
loss_uniform, loss_ada = 0.0, 0.0

for t in range(1, T + 1):
    x = np.zeros(d)
    active = np.random.choice(d, 3, replace=False)
    x[active] = np.random.randn(3)
    y = w_true @ x + 0.1 * np.random.randn()

    # Uniform-eta OGD
    pred_u = w_uniform @ x
    loss_uniform += (pred_u - y) ** 2
    grad = 2 * (pred_u - y) * x
    w_uniform -= (0.1 / np.sqrt(t)) * grad

    # AdaGrad: per-coordinate adaptive eta
    pred_a = w_ada @ x
    loss_ada += (pred_a - y) ** 2
    grad_a = 2 * (pred_a - y) * x
    G += grad_a ** 2
    w_ada -= 0.5 * grad_a / (np.sqrt(G) + 1e-8)

print(f"Uniform OGD loss: {loss_uniform:.1f}")
print(f"AdaGrad loss:     {loss_ada:.1f}")
print(f"Improvement:      {(1 - loss_ada / loss_uniform) * 100:.1f}%")
top = np.argsort(np.abs(w_ada))[::-1][:5]
for idx in top:
    print(f"  w[{idx:2d}] = {w_ada[idx]:+.3f}  (true: {w_true[idx]:+.1f})")
# AdaGrad excels on sparse problems with rare features
