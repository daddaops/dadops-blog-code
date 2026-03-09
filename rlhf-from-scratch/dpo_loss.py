"""Direct Preference Optimization (DPO) loss."""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dpo_loss(log_pi_chosen, log_pi_rejected, log_ref_chosen, log_ref_rejected, beta=0.1):
    """
    DPO loss — collapses reward modeling + RL into a single supervised loss.

    beta: controls divergence from reference.
          Small beta = trust preferences more, allow more divergence.
          Large beta = trust reference more, stay closer to SFT.
    """
    # Log-probability RATIOS — the implicit reward
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # DPO loss — push chosen ratio above rejected ratio
    logit = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -np.log(sigmoid(logit))
    return loss


# Smoke test: when policy assigns higher probability to chosen vs reference,
# and lower probability to rejected vs reference, loss should be small
log_pi_c, log_ref_c = -2.0, -3.0      # policy likes chosen more than ref does
log_pi_r, log_ref_r = -4.0, -3.0      # policy likes rejected less than ref does

loss = dpo_loss(log_pi_c, log_pi_r, log_ref_c, log_ref_r, beta=0.1)
print(f"DPO loss (correct preference): {loss:.4f}")

# When policy prefers the rejected response — loss should be large
loss_wrong = dpo_loss(log_pi_r, log_pi_c, log_ref_c, log_ref_r, beta=0.1)
print(f"DPO loss (wrong preference):   {loss_wrong:.4f}")

# Verify implicit reward formula: r_hat = beta * log(pi/ref)
r_chosen = 0.1 * (log_pi_c - log_ref_c)
r_rejected = 0.1 * (log_pi_r - log_ref_r)
print(f"\nImplicit rewards: chosen={r_chosen:.3f}, rejected={r_rejected:.3f}")
print(f"Reward gap: {r_chosen - r_rejected:.3f} (positive = correct preference)")
