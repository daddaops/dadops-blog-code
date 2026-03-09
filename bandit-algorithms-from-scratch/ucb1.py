"""UCB1 — Upper Confidence Bound bandit algorithm.

Selects arms by maximizing estimated reward + confidence bonus.
"""
import numpy as np
from bandit_testbed import bernoulli_bandit, pull_arm


def run_ucb1(probs, T=5000, c=np.sqrt(2), seed=0):
    """UCB1: always pick the arm with highest upper confidence bound."""
    K = len(probs)
    rng = np.random.RandomState(seed)
    counts = np.zeros(K)
    values = np.zeros(K)
    regret_history = np.zeros(T)
    cumulative_regret = 0
    best_mean = probs.max()

    # Pull each arm once
    for arm in range(K):
        reward = pull_arm(probs, arm, rng)
        counts[arm] = 1
        values[arm] = reward
        cumulative_regret += best_mean - probs[arm]
        regret_history[arm] = cumulative_regret

    for t in range(K, T):
        ucb_values = values + c * np.sqrt(np.log(t) / counts)
        arm = np.argmax(ucb_values)
        reward = pull_arm(probs, arm, rng)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        cumulative_regret += best_mean - probs[arm]
        regret_history[t] = cumulative_regret

    return regret_history, counts


if __name__ == "__main__":
    probs = bernoulli_bandit(K=5, seed=42)
    reg_ucb, counts_ucb = run_ucb1(probs, T=5000)
    print(f"UCB1 final regret: {reg_ucb[-1]:.1f}")
    print(f"Pull distribution: {counts_ucb.astype(int)}")
    print(f"Best arm gets {int(counts_ucb[np.argmax(probs)])} of 5000 pulls")
