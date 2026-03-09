"""LinUCB — contextual bandit with linear reward model.

Implements LinUCB for personalized article recommendation with
UCB-style exploration bonus based on ridge regression uncertainty.
"""
import numpy as np


def bernoulli_bandit(K, seed=42):
    """Create a K-armed bandit with random Bernoulli reward probabilities."""
    rng = np.random.RandomState(seed)
    probs = rng.uniform(0.1, 0.9, size=K)
    return probs


def run_linucb(n_articles, d_features, T=1000, alpha=1.0, seed=0):
    """LinUCB for contextual bandits with linear reward model."""
    rng = np.random.RandomState(seed)

    # True (hidden) article reward weights
    true_theta = rng.randn(n_articles, d_features) * 0.5

    # Per-arm models: A = d x d matrix, b = d vector
    A = [np.eye(d_features) for _ in range(n_articles)]
    b = [np.zeros(d_features) for _ in range(n_articles)]

    total_reward = 0
    optimal_reward = 0

    for t in range(T):
        # Observe context (user features)
        context = rng.randn(d_features)
        context = context / np.linalg.norm(context)

        # Compute UCB for each article
        ucb_scores = np.zeros(n_articles)
        for arm in range(n_articles):
            A_inv = np.linalg.inv(A[arm])
            theta_hat = A_inv @ b[arm]
            bonus = alpha * np.sqrt(context @ A_inv @ context)
            ucb_scores[arm] = theta_hat @ context + bonus

        chosen = np.argmax(ucb_scores)

        # Observe reward (linear + noise)
        true_rewards = [true_theta[a] @ context for a in range(n_articles)]
        noise = rng.randn() * 0.1
        reward = true_rewards[chosen] + noise

        # Update chosen arm's model
        A[chosen] += np.outer(context, context)
        b[chosen] += reward * context

        total_reward += reward
        optimal_reward += max(true_rewards)

    regret = optimal_reward - total_reward
    print(f"LinUCB after {T} rounds:")
    print(f"  Total reward:   {total_reward:.1f}")
    print(f"  Optimal reward: {optimal_reward:.1f}")
    print(f"  Regret:         {regret:.1f}")
    print(f"  Avg regret/step: {regret/T:.4f}")


if __name__ == "__main__":
    run_linucb(n_articles=5, d_features=4, T=2000, alpha=1.0)
