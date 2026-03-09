"""Bandit testbed — Bernoulli bandit environment and pure greedy agent.

Demonstrates how a pure greedy strategy locks onto a suboptimal arm
because it never explores.
"""
import numpy as np


def bernoulli_bandit(K, seed=42):
    """Create a K-armed bandit with random Bernoulli reward probabilities."""
    rng = np.random.RandomState(seed)
    probs = rng.uniform(0.1, 0.9, size=K)
    return probs


def pull_arm(probs, arm, rng):
    """Pull an arm and get a Bernoulli reward (0 or 1)."""
    return 1 if rng.random() < probs[arm] else 0


def run_greedy(probs, T=1000, seed=0):
    """Pure greedy: always pick the arm with highest observed average."""
    K = len(probs)
    rng = np.random.RandomState(seed)
    counts = np.zeros(K)
    values = np.zeros(K)
    total_reward = 0

    # Pull each arm once to initialize
    for arm in range(K):
        reward = pull_arm(probs, arm, rng)
        counts[arm] = 1
        values[arm] = reward
        total_reward += reward

    # Then always exploit
    for t in range(K, T):
        arm = np.argmax(values)
        reward = pull_arm(probs, arm, rng)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        total_reward += reward

    best_arm = np.argmax(probs)
    regret = T * probs[best_arm] - total_reward
    return counts, values, regret, best_arm


if __name__ == "__main__":
    probs = bernoulli_bandit(K=5, seed=42)
    print(f"True arm probabilities: {probs.round(3)}")
    print(f"Best arm: {np.argmax(probs)} (p={probs.max():.3f})\n")

    counts, values, regret, best = run_greedy(probs, T=2000)
    print(f"Greedy chose arm {np.argmax(counts)} most often ({int(counts.max())} pulls)")
    print(f"Pull distribution: {counts.astype(int)}")
    print(f"Cumulative regret: {regret:.1f}")
