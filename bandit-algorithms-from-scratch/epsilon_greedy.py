"""Epsilon-greedy bandit strategy with optional decay schedule.

Compares fixed epsilon=0.1, fixed epsilon=0.01, and decaying epsilon=1/sqrt(t).
"""
import numpy as np
from bandit_testbed import bernoulli_bandit, pull_arm


def run_epsilon_greedy(probs, T=5000, epsilon=0.1, decay=False, seed=0):
    """Epsilon-greedy with optional decay schedule."""
    K = len(probs)
    rng = np.random.RandomState(seed)
    counts = np.zeros(K)
    values = np.zeros(K)
    regret_history = np.zeros(T)
    cumulative_regret = 0
    best_mean = probs.max()

    for t in range(T):
        eps = 1.0 / np.sqrt(t + 1) if decay else epsilon

        if rng.random() < eps:
            arm = rng.randint(K)
        else:
            arm = np.argmax(values)

        reward = pull_arm(probs, arm, rng)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        cumulative_regret += best_mean - probs[arm]
        regret_history[t] = cumulative_regret

    return regret_history, counts


if __name__ == "__main__":
    probs = bernoulli_bandit(K=5, seed=42)

    reg_fixed_10, _ = run_epsilon_greedy(probs, T=5000, epsilon=0.1, seed=0)
    reg_fixed_01, _ = run_epsilon_greedy(probs, T=5000, epsilon=0.01, seed=0)
    reg_decay, _    = run_epsilon_greedy(probs, T=5000, decay=True, seed=0)

    print(f"eps=0.10 fixed  -> final regret: {reg_fixed_10[-1]:.1f}")
    print(f"eps=0.01 fixed  -> final regret: {reg_fixed_01[-1]:.1f}")
    print(f"eps=1/sqrt(t)   -> final regret: {reg_decay[-1]:.1f}")
