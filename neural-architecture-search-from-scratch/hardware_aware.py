"""Hardware-aware NAS with multi-objective optimization.

Balances accuracy against latency using evolutionary search
with Pareto analysis.
"""
import numpy as np
from search_space import sample_architecture, OPERATIONS
from evolutionary_search import mutate


def hardware_aware_search(pop_size, generations, target_latency, rng):
    """Evolutionary NAS with multi-objective: accuracy + latency."""
    # Latency cost per operation (milliseconds, simulated)
    op_latency = {'conv3x3': 2.5, 'conv5x5': 5.0, 'maxpool': 1.0,
                  'skip': 0.1, 'none': 0.0}
    # Accuracy contribution per operation
    op_quality = {'conv3x3': 0.85, 'conv5x5': 0.80, 'maxpool': 0.35,
                  'skip': 0.50, 'none': 0.05}

    def arch_latency(arch):
        return sum(op_latency[op] for node in arch for (_, op) in node)

    def arch_accuracy(arch):
        quality = sum(op_quality[op] for node in arch for (_, op) in node)
        return quality / (2 * len(arch)) + rng.normal(0, 0.02)

    def reward(arch):
        acc = arch_accuracy(arch)
        lat = arch_latency(arch)
        # Penalize if latency exceeds target
        lat_penalty = (lat / target_latency) ** (-0.07) if lat > 0 else 1.0
        return acc * lat_penalty

    # Evolution with multi-objective reward
    population = [(sample_architecture(rng), None) for _ in range(pop_size)]
    population = [(a, reward(a)) for a, _ in population]

    for gen in range(generations):
        # Tournament selection
        idxs = rng.choice(len(population), size=5, replace=False)
        parent = max([population[i] for i in idxs], key=lambda x: x[1])[0]
        child = mutate(parent, rng)
        child_reward = reward(child)
        population.append((child, child_reward))
        population.pop(0)

    # Final Pareto analysis
    all_results = [(arch_accuracy(a), arch_latency(a), a) for a, _ in population]
    all_results.sort(key=lambda x: x[1])  # sort by latency

    print(f"{'Accuracy':>10s} {'Latency':>10s}  Architecture ops")
    for acc, lat, arch in all_results[:5]:
        ops = [op for node in arch for (_, op) in node]
        print(f"{acc:10.3f} {lat:8.1f}ms  {ops}")

    return all_results


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    results = hardware_aware_search(
        pop_size=20, generations=80, target_latency=15.0, rng=rng)
