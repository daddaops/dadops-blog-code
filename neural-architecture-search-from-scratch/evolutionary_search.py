"""Regularized evolutionary search for NAS.

Uses mutation (operation change or input rewiring) and tournament
selection with age-based regularization.
"""
import numpy as np
from copy import deepcopy
from search_space import sample_architecture, synthetic_fitness, OPERATIONS


def mutate(arch, rng):
    """Apply one random mutation to an architecture."""
    child = deepcopy(arch)
    node_idx = rng.integers(0, len(child))
    branch = rng.integers(0, 2)  # mutate first or second input
    mutation_type = str(rng.choice(['op', 'input']))

    if mutation_type == 'op':
        new_op = str(rng.choice(OPERATIONS))
        inp, _ = child[node_idx][branch]
        child[node_idx] = list(child[node_idx])
        child[node_idx][branch] = (inp, new_op)
        child[node_idx] = tuple(child[node_idx])
    else:
        num_inputs = node_idx + 2
        new_inp = rng.integers(0, num_inputs)
        _, op = child[node_idx][branch]
        child[node_idx] = list(child[node_idx])
        child[node_idx][branch] = (new_inp, op)
        child[node_idx] = tuple(child[node_idx])
    return child


def evolutionary_search(pop_size, generations, tournament_k, rng):
    """Regularized evolution for architecture search."""
    # Initialize population with random architectures
    population = []
    for _ in range(pop_size):
        arch = sample_architecture(rng)
        fitness = synthetic_fitness(arch, max_epochs=50, rng=rng)
        population.append((arch, fitness))

    history = [max(f for _, f in population)]

    for gen in range(generations):
        # Tournament selection: pick best of K random members
        candidates = [population[i] for i in
                      rng.choice(len(population), size=tournament_k, replace=False)]
        parent_arch, _ = max(candidates, key=lambda x: x[1])

        # Mutate parent to create child
        child_arch = mutate(parent_arch, rng)
        child_fitness = synthetic_fitness(child_arch, max_epochs=50, rng=rng)

        # Add child, remove oldest (regularized evolution)
        population.append((child_arch, child_fitness))
        population.pop(0)  # remove oldest

        best = max(f for _, f in population)
        history.append(best)

    return max(population, key=lambda x: x[1]), history


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    (best_arch, best_fit), history = evolutionary_search(
        pop_size=20, generations=100, tournament_k=5, rng=rng)
    print(f"Best fitness: {best_fit:.3f}")
    print(f"Improvement: {history[0]:.3f} -> {history[-1]:.3f}")
