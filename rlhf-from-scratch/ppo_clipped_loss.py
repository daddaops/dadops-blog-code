"""PPO clipped surrogate objective for RLHF."""
import numpy as np


def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    """
    PPO clipped surrogate objective.

    log_probs_new: log pi_theta(a|s) for each token    — (batch,)
    log_probs_old: log pi_old(a|s) from data collection — (batch,)
    advantages:    A_t = reward - value_baseline         — (batch,)
    epsilon:       clipping range (default 0.2)
    """
    # Policy ratio: how much has the policy changed?
    ratio = np.exp(log_probs_new - log_probs_old)  # pi_new / pi_old

    # Unclipped objective
    surr1 = ratio * advantages

    # Clipped objective — limit how much the ratio can change
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    surr2 = clipped_ratio * advantages

    # Take the MINIMUM — the conservative (pessimistic) estimate
    # This prevents catastrophically large policy updates
    loss = -np.minimum(surr1, surr2).mean()
    return loss


# Verify the worked example from the blog:
# ratio = 1.5, A_t = 0.3
# clipped_ratio = clip(1.5, 0.8, 1.2) = 1.2
# surr1 = 1.5 * 0.3 = 0.45
# surr2 = 1.2 * 0.3 = 0.36
# min(0.45, 0.36) = 0.36

# Construct log_probs to give ratio = 1.5: exp(new - old) = 1.5 => new - old = log(1.5)
log_old = np.array([-1.0])
log_new = np.array([-1.0 + np.log(1.5)])
advantages = np.array([0.3])

ratio = np.exp(log_new - log_old)
clipped = np.clip(ratio, 0.8, 1.2)
surr1 = ratio * advantages
surr2 = clipped * advantages

print(f"ratio = {ratio[0]:.1f}")
print(f"clipped_ratio = {clipped[0]:.1f}")
print(f"surr1 = {ratio[0]:.1f} * {advantages[0]:.1f} = {surr1[0]:.2f}")
print(f"surr2 = {clipped[0]:.1f} * {advantages[0]:.1f} = {surr2[0]:.2f}")
print(f"min({surr1[0]:.2f}, {surr2[0]:.2f}) = {min(surr1[0], surr2[0]):.2f}")
