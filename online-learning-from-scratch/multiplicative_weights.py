"""Multiplicative Weights Update for adversarial expert advice.

Handles continuous losses in [0,1] with shifting best expert.
"""
import numpy as np

np.random.seed(123)
T, N = 500, 5
eta = np.sqrt(2 * np.log(N) / T)  # optimal learning rate

# Adversarial losses: each expert is "best" for 100 rounds
losses = np.random.uniform(0.3, 0.7, (T, N))
for phase in range(5):
    s, e = phase * 100, (phase + 1) * 100
    losses[s:e, phase] = np.random.uniform(0.0, 0.2, 100)

# Multiplicative Weights Update
weights = np.ones(N) / N
cumulative_algo = 0.0

for t in range(T):
    # Expected loss under current distribution
    cumulative_algo += weights @ losses[t]

    # Exponential update + renormalize
    weights *= np.exp(-eta * losses[t])
    weights /= weights.sum()

best_expert = min(losses[:, i].sum() for i in range(N))
regret = cumulative_algo - best_expert
bound = np.sqrt(2 * T * np.log(N))  # (ln N)/eta + eta*T/2

print(f"Optimal eta: {eta:.4f}")
print(f"Algorithm loss: {cumulative_algo:.1f}")
print(f"Best expert: {best_expert:.1f}")
print(f"Regret: {regret:.1f}")
print(f"Theory bound: {bound:.1f}")
# Actual regret is well within the sqrt(2 T ln N) bound!
