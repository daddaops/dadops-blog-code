"""DARTS: Differentiable Architecture Search.

Uses bilevel optimization with softmax-weighted operation mixtures.
Architecture parameters converge to select optimal operations.
"""
import numpy as np
from search_space import OPERATIONS


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def darts_search(n_edges, n_ops, steps, lr_w, lr_alpha, rng):
    """Simplified DARTS: bilevel optimization of weights and arch params."""
    # Architecture parameters: one logit vector per edge
    alphas = [np.zeros(n_ops) for _ in range(n_edges)]

    # Simulated network weights (one weight per edge-operation pair)
    weights = [[rng.normal(0, 0.1) for _ in range(n_ops)]
               for _ in range(n_edges)]

    # Target: conv3x3 (idx 0) and skip (idx 3) are the best operations
    target_ops = [0, 3, 0, 0, 3, 0]  # best op index per edge

    history = []
    for step in range(steps):
        # Step 1: Update weights on "training loss"
        for e in range(n_edges):
            probs = softmax(alphas[e])
            for i in range(n_ops):
                grad = 2 * probs[i] * (sum(probs[j] * weights[e][j]
                       for j in range(n_ops)) - 1.0)
                weights[e][i] -= lr_w * grad

        # Step 2: Update architecture params on "validation loss"
        for e in range(n_edges):
            probs = softmax(alphas[e])
            for i in range(n_ops):
                grad_a = probs[i] * (1.0 if i != target_ops[e] else
                         (1.0 - 1.0 / (probs[i] + 1e-8)))
                alphas[e][i] -= lr_alpha * grad_a

        # Record which operation is selected per edge
        selected = [OPERATIONS[np.argmax(a)] for a in alphas]
        history.append(selected)

    # Final architecture: argmax of alpha per edge
    final = [OPERATIONS[np.argmax(a)] for a in alphas]
    final_probs = [softmax(a) for a in alphas]
    return final, final_probs, history


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    arch, probs, hist = darts_search(
        n_edges=6, n_ops=5, steps=200, lr_w=0.01, lr_alpha=0.05, rng=rng)
    print("Final architecture:", arch)
    print("Edge 0 probs:", {OPERATIONS[i]: f"{p:.3f}"
          for i, p in enumerate(probs[0])})
