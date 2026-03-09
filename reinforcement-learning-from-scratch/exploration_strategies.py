"""Exploration strategies: epsilon-greedy, softmax, UCB on a multi-armed bandit."""
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    """Random action with prob epsilon, greedy otherwise."""
    if np.random.random() < epsilon:
        return np.random.randint(len(Q[state]))
    return np.argmax(Q[state])

def softmax_explore(Q, state, tau=1.0):
    """Sample action proportionally to exp(Q/tau)."""
    q = Q[state]
    q_shifted = q - np.max(q)  # numerical stability
    probs = np.exp(q_shifted / tau) / np.sum(np.exp(q_shifted / tau))
    return np.random.choice(len(q), p=probs)

def ucb_select(Q, N, state, t, c=2.0):
    """Select action maximizing Q + exploration bonus."""
    n = N[state]
    # Avoid division by zero for untried actions (infinite bonus)
    bonus = c * np.sqrt(np.log(t + 1) / (n + 1e-8))
    return np.argmax(Q[state] + bonus)

# --- Multi-armed bandit comparison ---
n_arms = 5
true_rewards = np.random.randn(n_arms) * 0.5 + 1.0  # hidden means
n_pulls = 1000

for strategy_name, select_fn in [("e-greedy", "eg"),
                                  ("softmax", "sm"),
                                  ("UCB", "ucb")]:
    Q_est = np.zeros(n_arms)
    N_count = np.zeros(n_arms)
    total_reward = 0

    for t in range(n_pulls):
        if select_fn == "eg":
            arm = epsilon_greedy(Q_est, slice(None), epsilon=0.1)
        elif select_fn == "sm":
            arm = softmax_explore(Q_est, slice(None), tau=0.5)
        else:
            arm = ucb_select(Q_est, N_count, slice(None), t)

        reward = np.random.randn() * 0.3 + true_rewards[arm]
        N_count[arm] += 1
        Q_est[arm] += (reward - Q_est[arm]) / N_count[arm]  # running mean
        total_reward += reward

    best_possible = true_rewards.max() * n_pulls
    regret = best_possible - total_reward
    print(f"{strategy_name:>10}: reward={total_reward:.1f}, regret={regret:.1f}")
