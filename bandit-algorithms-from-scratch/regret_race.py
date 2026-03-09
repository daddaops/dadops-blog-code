"""Regret race — compare all four bandit algorithms on the same instance.

Runs greedy, epsilon-greedy, UCB1, and Thompson Sampling for 10,000 steps
and compares cumulative pseudo-regret, averaged over multiple seeds.
"""
import numpy as np
from bandit_testbed import bernoulli_bandit, pull_arm
from epsilon_greedy import run_epsilon_greedy
from ucb1 import run_ucb1
from thompson_sampling import run_thompson


def run_greedy_pseudoregret(probs, T=10000, seed=0):
    """Pure greedy returning pseudo-regret (same metric as other algorithms)."""
    K = len(probs)
    rng = np.random.RandomState(seed)
    counts = np.zeros(K)
    values = np.zeros(K)
    best_mean = probs.max()
    cumulative_regret = 0

    # Pull each arm once
    for arm in range(K):
        reward = pull_arm(probs, arm, rng)
        counts[arm] = 1
        values[arm] = reward
        cumulative_regret += best_mean - probs[arm]

    # Then always exploit
    for t in range(K, T):
        arm = np.argmax(values)
        reward = pull_arm(probs, arm, rng)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        cumulative_regret += best_mean - probs[arm]

    return cumulative_regret


if __name__ == "__main__":
    probs = bernoulli_bandit(K=5, seed=42)
    T = 10000
    n_runs = 20

    # Average over multiple seeds for stable results
    greedy_regrets = []
    eps_regrets = []
    ucb_regrets = []
    ts_regrets = []

    for s in range(n_runs):
        greedy_regrets.append(run_greedy_pseudoregret(probs, T=T, seed=s))
        reg_eps, _ = run_epsilon_greedy(probs, T=T, epsilon=0.1, seed=s)
        eps_regrets.append(reg_eps[-1])
        reg_ucb, _ = run_ucb1(probs, T=T, seed=s)
        ucb_regrets.append(reg_ucb[-1])
        reg_ts, _, _, _ = run_thompson(probs, T=T, seed=s)
        ts_regrets.append(reg_ts[-1])

    print(f"Cumulative pseudo-regret after {T:,} steps (avg over {n_runs} runs):")
    print(f"  Greedy:           {np.mean(greedy_regrets):.1f} (std={np.std(greedy_regrets):.1f})")
    print(f"  Epsilon-greedy:   {np.mean(eps_regrets):.1f} (std={np.std(eps_regrets):.1f})")
    print(f"  UCB1:             {np.mean(ucb_regrets):.1f} (std={np.std(ucb_regrets):.1f})")
    print(f"  Thompson:         {np.mean(ts_regrets):.1f} (std={np.std(ts_regrets):.1f})")
    print(f"\nAsymptotic rates:")
    print(f"  Greedy:           O(T)          -- linear, catastrophic")
    print(f"  Eps-greedy:       O(eps * T)    -- linear, proportional to eps")
    print(f"  UCB1:             O(sqrt(KT logT)) -- sublinear, logarithmic")
    print(f"  Thompson:         Lai-Robbins   -- optimal, can't do better")
