"""Uncertainty Weighting (Kendall et al. 2018).

Automatically learns task weights by treating per-task homoscedastic
uncertainty as a trainable parameter. High-loss tasks get downweighted.
"""
import numpy as np


class UncertaintyWeighting:
    """Kendall et al. 2018: learn task weights from homoscedastic uncertainty."""
    def __init__(self, n_tasks=2):
        # log(sigma^2) for each task, initialized to 0 (sigma=1)
        self.log_vars = np.zeros(n_tasks)

    def weighted_loss(self, task_losses):
        """Compute uncertainty-weighted total loss.
        task_losses: list of scalar loss values [L_1, L_2, ...]
        """
        total = 0.0
        for k, L_k in enumerate(task_losses):
            precision = np.exp(-self.log_vars[k])   # 1/sigma^2
            total += 0.5 * precision * L_k + 0.5 * self.log_vars[k]
        return total

    def grad_log_vars(self, task_losses):
        """Gradient of total loss w.r.t. log_vars (for updating weights)."""
        grads = np.zeros_like(self.log_vars)
        for k, L_k in enumerate(task_losses):
            precision = np.exp(-self.log_vars[k])
            grads[k] = -0.5 * precision * L_k + 0.5   # dL/d(log_var_k)
        return grads

    def update(self, task_losses, lr=0.01):
        grads = self.grad_log_vars(task_losses)
        self.log_vars -= lr * grads

    def effective_weights(self):
        return 0.5 * np.exp(-self.log_vars)


if __name__ == "__main__":
    uw = UncertaintyWeighting(n_tasks=2)

    # Simulate: regression loss ~200, classification loss ~0.7
    # Over many steps, weights should auto-balance
    print("Uncertainty Weighting convergence:")
    print(f"{'Step':>6} {'L_reg':>8} {'L_cls':>8} {'w_reg':>8} {'w_cls':>8} {'Weighted':>10}")
    for step in range(300):
        loss_reg = 200.0 * np.exp(-0.005 * step)  # regression improves
        loss_cls = 0.7 - 0.001 * step              # classification barely moves
        loss_cls = max(loss_cls, 0.4)

        uw.update([loss_reg, loss_cls], lr=0.01)
        weights = uw.effective_weights()
        weighted = uw.weighted_loss([loss_reg, loss_cls])

        if step % 50 == 0:
            print(f"{step:>6} {loss_reg:>8.3f} {loss_cls:>8.3f} "
                  f"{weights[0]:>8.4f} {weights[1]:>8.4f} {weighted:>10.3f}")

    print(f"\nFinal effective weights: [{weights[0]:.3f}, {weights[1]:.3f}]")
    print(f"Ratio: {weights[1]/weights[0]:.1f}x — classification upweighted")
