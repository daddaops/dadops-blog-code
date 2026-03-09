"""One-shot NAS with a supernet.

Trains shared weights across randomly sampled architectures,
then ranks operations by their supernet performance.
"""
import numpy as np
from search_space import OPERATIONS


class Supernet:
    """A supernet with shared weights for all candidate architectures."""
    def __init__(self, n_edges, n_ops, rng):
        self.n_edges = n_edges
        self.n_ops = n_ops
        # Shared weights: one parameter per (edge, operation) pair
        self.weights = rng.normal(0, 0.1, (n_edges, n_ops))
        # Track how often each weight is updated
        self.update_counts = np.zeros((n_edges, n_ops))

    def forward(self, arch_indices, x=1.0):
        """Forward pass through a specific architecture (subgraph)."""
        output = x
        for edge, op_idx in enumerate(arch_indices):
            output = output * self.weights[edge, op_idx]
        return output

    def train_step(self, arch_indices, target, lr=0.01):
        """Train only the weights used by this architecture."""
        pred = self.forward(arch_indices)
        loss = (pred - target) ** 2

        # Gradient for each active weight
        for edge, op_idx in enumerate(arch_indices):
            grad = 2 * (pred - target)
            # Chain rule: d(loss)/d(w_i) = grad * product of other weights
            for e2, o2 in enumerate(arch_indices):
                if e2 != edge:
                    grad *= self.weights[e2, o2]
            self.weights[edge, op_idx] -= lr * np.clip(grad, -1, 1)
            self.update_counts[edge, op_idx] += 1
        return loss


def train_supernet(n_edges, n_ops, n_steps, rng):
    """Train supernet by sampling random architectures each step."""
    supernet = Supernet(n_edges, n_ops, rng)
    target = 0.5  # target output

    for step in range(n_steps):
        # Sample random architecture
        arch = [rng.integers(0, n_ops) for _ in range(n_edges)]
        supernet.train_step(arch, target, lr=0.005)

    # Evaluate all single-edge architectures by shared weights
    rankings = {}
    for op_idx in range(n_ops):
        arch = [op_idx] * n_edges  # same op everywhere
        score = abs(supernet.forward(arch) - target)
        rankings[OPERATIONS[op_idx]] = 1.0 - min(score, 1.0)

    return supernet, rankings


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    supernet, rankings = train_supernet(n_edges=4, n_ops=5, n_steps=2000, rng=rng)
    print("Supernet rankings (higher = closer to target):")
    for op, score in sorted(rankings.items(), key=lambda x: -x[1]):
        print(f"  {op:<10s} {score:.3f}")
    print(f"\nTotal weight updates: {int(supernet.update_counts.sum())}")
