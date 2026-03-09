"""PPO clipped surrogate objective."""
import numpy as np

def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    """PPO's clipped surrogate objective.

    log_probs_new: log pi_theta(a|s) for each token/action
    log_probs_old: log pi_old(a|s) from data collection
    advantages:    A_t = G_t - V(s_t)
    """
    # Policy ratio: how much did the policy change?
    ratio = np.exp(log_probs_new - log_probs_old)

    # Unclipped objective: ratio * advantage
    unclipped = ratio * advantages

    # Clipped objective: keep ratio in [1-eps, 1+eps]
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages

    # Take the minimum — pessimistic bound prevents overconfident updates
    loss = np.minimum(unclipped, clipped).mean()
    return loss

# Smoke test with synthetic data
np.random.seed(42)
n = 100
log_probs_old = np.random.randn(n) * 0.5 - 1.0
log_probs_new = log_probs_old + np.random.randn(n) * 0.1  # small policy change
advantages = np.random.randn(n)

loss = ppo_clipped_loss(log_probs_new, log_probs_old, advantages)
print(f"PPO clipped loss: {loss:.4f}")

# Show that clipping prevents large updates
log_probs_big = log_probs_old + np.random.randn(n) * 2.0  # large change
loss_big = ppo_clipped_loss(log_probs_big, log_probs_old, advantages)
print(f"PPO loss with large update: {loss_big:.4f}")
print(f"Clipping effect: {'active' if abs(loss_big) < abs(loss) * 5 else 'minimal'}")
