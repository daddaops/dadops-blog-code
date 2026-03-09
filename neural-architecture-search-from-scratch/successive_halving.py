"""Successive halving random search for NAS.

Progressively eliminates the bottom half of candidates while
increasing training budget for survivors.
"""
import numpy as np
from search_space import sample_architecture, synthetic_fitness, OPERATIONS


def successive_halving(n_initial, max_budget, rng):
    """Random search with successive halving."""
    archs = [sample_architecture(rng) for _ in range(n_initial)]
    budget = max_budget // (int(np.log2(n_initial)) + 1)  # per-round budget

    round_num = 0
    while len(archs) > 1:
        # Evaluate all surviving architectures at current budget
        scores = [synthetic_fitness(a, budget * (round_num + 1), rng)
                  for a in archs]
        # Keep the top half
        ranked = sorted(zip(scores, archs), reverse=True)
        archs = [a for _, a in ranked[:len(archs) // 2]]
        print(f"Round {round_num}: {len(ranked)} archs evaluated, "
              f"best={ranked[0][0]:.3f}, kept top {len(archs)}")
        round_num += 1

    return archs[0]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    best = successive_halving(n_initial=16, max_budget=100, rng=rng)
    print(f"Winner: {[(op1, op2) for ((_,op1),(_,op2)) in best]}")
