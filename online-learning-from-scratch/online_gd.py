"""Online Gradient Descent for online linear regression.

Data arrives one point at a time, OGD converges to true weights.
"""
import numpy as np

np.random.seed(7)
d, T = 5, 1000
w_star = np.random.randn(d)  # unknown optimal weights

w = np.zeros(d)
cumulative_loss, best_loss = 0.0, 0.0

for t in range(1, T + 1):
    # Adversary reveals data point
    x = np.random.randn(d)
    y = w_star @ x + 0.1 * np.random.randn()

    # Learner's prediction and loss
    pred = w @ x
    loss = (pred - y) ** 2
    cumulative_loss += loss
    best_loss += (w_star @ x - y) ** 2

    # OGD update: gradient of squared loss
    grad = 2 * (pred - y) * x
    eta = 1.0 / np.sqrt(t)
    w -= eta * grad

regret = cumulative_loss - best_loss
print(f"Cumulative loss (OGD): {cumulative_loss:.1f}")
print(f"Best-in-hindsight:     {best_loss:.1f}")
print(f"Regret:                {regret:.1f}")
print(f"Regret / sqrt(T):      {regret / np.sqrt(T):.2f}")
print(f"||w - w*||:            {np.linalg.norm(w - w_star):.4f}")
# Regret scales as O(sqrt(T)), w converges to w_star
