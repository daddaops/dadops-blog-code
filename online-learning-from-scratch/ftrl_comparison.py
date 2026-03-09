"""FTRL comparison: MW (entropic) vs OGD (L2) on same problem.

Shows both are instances of Follow the Regularized Leader.
"""
import numpy as np

np.random.seed(42)
T, N = 300, 6
losses = np.random.uniform(0.3, 0.7, (T, N))
losses[:, 2] = np.random.uniform(0.0, 0.2, T)  # expert 2 best

eta = np.sqrt(2 * np.log(N) / T)

# FTRL with entropic regularizer -> Multiplicative Weights
w_mw = np.ones(N) / N
mw_cum = 0.0

# FTRL with L2 regularizer -> Online Gradient Descent
w_ogd = np.ones(N) / N
ogd_cum = 0.0

for t in range(T):
    # Play distributions, suffer expected loss
    mw_cum += w_mw @ losses[t]
    ogd_cum += w_ogd @ losses[t]

    # MW update (entropic regularizer)
    w_mw *= np.exp(-eta * losses[t])
    w_mw /= w_mw.sum()

    # OGD update (L2 regularizer)
    w_ogd -= eta * losses[t]
    w_ogd = np.maximum(w_ogd, 0)
    if w_ogd.sum() > 0:
        w_ogd /= w_ogd.sum()
    else:
        w_ogd = np.ones(N) / N

best = min(losses[:, i].sum() for i in range(N))
bound = np.sqrt(2 * T * np.log(N))
print(f"MW (entropic FTRL)  regret: {mw_cum - best:.1f}")
print(f"OGD (L2 FTRL)       regret: {ogd_cum - best:.1f}")
print(f"Theoretical bound:          {bound:.1f}")
print(f"\nMW final weights:  {np.round(w_mw, 3)}")
print(f"OGD final weights: {np.round(w_ogd, 3)}")
# Both achieve sublinear regret; MW adapts faster on simplex
