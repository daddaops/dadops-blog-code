"""Thompson Sampling with Beta posteriors for Bernoulli bandits.

Maintains a Beta(alpha, beta) posterior for each arm and samples
to decide which arm to pull.
"""
import numpy as np
from bandit_testbed import bernoulli_bandit, pull_arm


def run_thompson(probs, T=5000, seed=0):
    """Thompson Sampling with Beta posteriors for Bernoulli bandits."""
    K = len(probs)
    rng = np.random.RandomState(seed)
    alpha = np.ones(K)  # Beta prior: starts at Beta(1,1) = uniform
    beta_param = np.ones(K)
    counts = np.zeros(K)
    regret_history = np.zeros(T)
    cumulative_regret = 0
    best_mean = probs.max()

    for t in range(T):
        # Sample from each arm's posterior
        samples = rng.beta(alpha, beta_param)
        arm = np.argmax(samples)

        # Pull arm and observe reward
        reward = pull_arm(probs, arm, rng)

        # Update posterior
        alpha[arm] += reward
        beta_param[arm] += 1 - reward
        counts[arm] += 1

        cumulative_regret += best_mean - probs[arm]
        regret_history[t] = cumulative_regret

    return regret_history, counts, alpha, beta_param


if __name__ == "__main__":
    probs = bernoulli_bandit(K=5, seed=42)
    reg_ts, counts_ts, final_alpha, final_beta = run_thompson(probs, T=5000)
    print(f"Thompson Sampling final regret: {reg_ts[-1]:.1f}")
    print(f"Pull distribution: {counts_ts.astype(int)}")
    print(f"\nPosterior parameters (alpha, beta) per arm:")
    for i in range(len(probs)):
        mean_est = final_alpha[i] / (final_alpha[i] + final_beta[i])
        print(f"  Arm {i}: Beta({final_alpha[i]:.0f}, {final_beta[i]:.0f})"
              f" -> mean={mean_est:.3f} (true={probs[i]:.3f})")
