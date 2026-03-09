"""Online-to-Batch conversion with Polyak-Ruppert averaging.

Shows averaging OGD iterates gives better generalization.
"""
import numpy as np

np.random.seed(99)
n, d = 500, 3
X = np.random.randn(n, d)
w_true = np.array([1.0, -0.5, 0.3])
y = np.sign(X @ w_true + 0.2 * np.random.randn(n))

# Online GD on logistic loss, one sample at a time
w = np.zeros(d)
all_w = [w.copy()]

for t in range(n):
    margin = y[t] * (w @ X[t])
    sig = 1.0 / (1.0 + np.exp(np.clip(margin, -30, 30)))
    grad = -y[t] * X[t] * sig
    eta = 1.0 / np.sqrt(t + 1)
    w = w - eta * grad
    all_w.append(w.copy())

# Online-to-batch: Polyak-Ruppert averaging
w_avg = np.mean(all_w, axis=0)

acc_last = np.mean(np.sign(X @ w) == y)
acc_avg = np.mean(np.sign(X @ w_avg) == y)
acc_opt = np.mean(np.sign(X @ w_true) == y)

print(f"Last iterate accuracy:  {acc_last:.3f}")
print(f"Averaged (Polyak):      {acc_avg:.3f}")
print(f"True weights:           {acc_opt:.3f}")
# Averaging smooths out SGD oscillations -> better generalization
