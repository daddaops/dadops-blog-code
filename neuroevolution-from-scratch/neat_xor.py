"""NEAT XOR evolution.

Evolves network topology to solve XOR, starting from minimal
3-input-to-1-output networks and discovering hidden nodes.
Note: This simplified NEAT omits speciation, which real NEAT uses
to protect topological innovations. Without it, convergence is slower.
"""
import numpy as np
import random
from neat_genome import Genome, ConnectionGene


def feed_forward(genome, inputs):
    """Evaluate a NEAT genome with proper topological ordering."""
    values = {0: inputs[0], 1: inputs[1], 2: 1.0}  # x1, x2, bias
    active = [c for c in genome.connections if c.enabled]

    # Compute depth of each node (longest path from inputs)
    depth = {0: 0, 1: 0, 2: 0}
    changed = True
    while changed:
        changed = False
        for c in active:
            if c.in_node in depth:
                d = depth[c.in_node] + 1
                if c.out_node not in depth or depth[c.out_node] < d:
                    depth[c.out_node] = d
                    changed = True

    # Process non-input nodes in depth order, sigmoid per node
    ordered = sorted(
        [n for n in genome.node_genes if n not in (0, 1, 2)],
        key=lambda n: depth.get(n, 1)
    )
    for node in ordered:
        total = sum(values.get(c.in_node, 0) * c.weight
                    for c in active if c.out_node == node)
        values[node] = 1 / (1 + np.exp(-np.clip(total, -50, 50)))

    return values.get(3, 0.5)  # output node = 3

def evolve_xor(pop_size=150, generations=200):
    xor_data = [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)]

    # Minimal starting topology: 3 inputs -> 1 output
    population = []
    for _ in range(pop_size):
        g = Genome([0, 1, 2, 3])
        g.add_connection(0, 3)
        g.add_connection(1, 3)
        g.add_connection(2, 3)
        g.mutate_weights()
        population.append(g)

    for gen in range(generations):
        fitnesses = []
        for genome in population:
            error = sum((feed_forward(genome, x) - y)**2
                        for x, y in xor_data)
            fitnesses.append(4.0 - error)

        best = max(fitnesses)
        avg_nodes = np.mean([len(g.node_genes) for g in population])
        if gen % 25 == 0 or gen == generations - 1 or best > 3.9:
            print(f"Gen {gen:3d} | Best: {best:.3f} | Nodes: {avg_nodes:.1f}")

        if best > 3.9:
            winner = population[np.argmax(fitnesses)]
            print(f"Solved! Network has {len(winner.node_genes)} nodes")
            return winner

        # Selection + reproduction (simplified, with elitism)
        ranked = sorted(range(pop_size), key=lambda i: fitnesses[i])
        elite = population[ranked[-1]]
        next_pop = [Genome(list(elite.node_genes), [
            ConnectionGene(c.in_node, c.out_node, c.weight,
                           c.enabled, c.innovation)
            for c in elite.connections])]
        while len(next_pop) < pop_size:
            p = population[ranked[np.random.randint(pop_size//2, pop_size)]]
            child = Genome(list(p.node_genes), [
                ConnectionGene(c.in_node, c.out_node, c.weight,
                               c.enabled, c.innovation)
                for c in p.connections])
            child.mutate_weights()
            if random.random() < 0.1:
                child.add_connection(
                    random.choice(child.node_genes[:3]),
                    random.choice(child.node_genes[3:]))
            if random.random() < 0.05:
                child.add_node()
            next_pop.append(child)
        population = next_pop

    print(f"Best fitness after {generations} generations: {max(fitnesses):.3f}")
    return population[np.argmax(fitnesses)]


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    evolve_xor()
