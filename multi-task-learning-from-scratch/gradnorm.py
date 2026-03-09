"""GradNorm: balance training rates via gradient norm matching.

Adjusts task weights so that all tasks have similar relative
training rates, measured by loss ratios and gradient magnitudes.
"""
import numpy as np


class GradNorm:
    """GradNorm: balance training rates via gradient norm matching."""
    def __init__(self, n_tasks=2, alpha=1.5):
        self.weights = np.ones(n_tasks)       # task weights w_k
        self.alpha = alpha                     # rate balancing strength
        self.initial_losses = None             # L_k(0) for relative rates

    def update_weights(self, task_losses, task_grad_norms, lr=0.025):
        """Adjust weights to balance gradient norms across tasks.
        task_losses: current loss per task [L_1, L_2]
        task_grad_norms: ||grad_W(w_k * L_k)||_2 per task
        """
        if self.initial_losses is None:
            self.initial_losses = np.array(task_losses, dtype=float)
            return
        # Relative inverse training rate: how fast is each task learning?
        loss_ratios = np.array(task_losses) / (self.initial_losses + 1e-8)
        avg_ratio = np.mean(loss_ratios)
        rel_rates = loss_ratios / (avg_ratio + 1e-8)
        # Target gradient norm per task
        avg_gnorm = np.mean(task_grad_norms)
        target_gnorms = avg_gnorm * (rel_rates ** self.alpha)
        # Gradient of |G_k - target_k| w.r.t. w_k
        for k in range(len(self.weights)):
            grad_w = (task_grad_norms[k] - target_gnorms[k])
            self.weights[k] -= lr * grad_w
        # Renormalize so weights sum to n_tasks
        self.weights = len(self.weights) * self.weights / (np.sum(self.weights) + 1e-8)
        self.weights = np.clip(self.weights, 0.1, 10.0)


if __name__ == "__main__":
    gn = GradNorm(n_tasks=2, alpha=1.5)

    print("GradNorm weight evolution:")
    print(f"{'Step':>6} {'L_reg':>8} {'L_cls':>8} {'gnorm_r':>8} {'gnorm_c':>8} {'w_reg':>6} {'w_cls':>6}")

    for step in range(20):
        # Simulate: regression trains fast, classification is stuck
        loss_reg = 200.0 * np.exp(-0.1 * step)
        loss_cls = 0.7 * np.exp(-0.01 * step)
        gnorm_reg = 15.0 * np.exp(-0.05 * step)
        gnorm_cls = 0.8 * np.exp(-0.02 * step)

        gn.update_weights([loss_reg, loss_cls], [gnorm_reg, gnorm_cls])
        print(f"{step:>6} {loss_reg:>8.2f} {loss_cls:>8.3f} "
              f"{gnorm_reg:>8.2f} {gnorm_cls:>8.3f} "
              f"{gn.weights[0]:>6.3f} {gn.weights[1]:>6.3f}")

    print(f"\nFinal weights: [{gn.weights[0]:.3f}, {gn.weights[1]:.3f}]")
    if gn.weights[1] > gn.weights[0]:
        print("Classification upweighted — GradNorm boosts the slow learner")
