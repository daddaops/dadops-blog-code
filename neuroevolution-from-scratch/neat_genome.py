"""NEAT genome representation.

Core data structures: ConnectionGene, Genome with add_connection,
add_node, mutate_weights, crossover, and compatibility distance.
"""
import random
from dataclasses import dataclass, field

innovation_counter = 0

@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0

@dataclass
class Genome:
    node_genes: list
    connections: list = field(default_factory=list)

    def add_connection(self, in_node, out_node):
        global innovation_counter
        innovation_counter += 1
        self.connections.append(ConnectionGene(
            in_node, out_node,
            weight=random.gauss(0, 1),
            innovation=innovation_counter
        ))

    def add_node(self):
        """Split a random enabled connection with a new neuron."""
        global innovation_counter
        enabled = [c for c in self.connections if c.enabled]
        if not enabled:
            return
        conn = random.choice(enabled)
        conn.enabled = False
        new_id = max(self.node_genes) + 1
        self.node_genes.append(new_id)
        # in -> new (weight 1.0), new -> out (original weight)
        innovation_counter += 1
        self.connections.append(ConnectionGene(
            conn.in_node, new_id, 1.0,
            innovation=innovation_counter))
        innovation_counter += 1
        self.connections.append(ConnectionGene(
            new_id, conn.out_node, conn.weight,
            innovation=innovation_counter))

    def mutate_weights(self, rate=0.8, sigma=0.2):
        for c in self.connections:
            if random.random() < rate:
                c.weight += random.gauss(0, sigma)
            else:
                c.weight = random.gauss(0, 1)

def crossover(p1, p2, fit1, fit2):
    """Align genes by innovation number, inherit from fitter parent."""
    genes1 = {c.innovation: c for c in p1.connections}
    genes2 = {c.innovation: c for c in p2.connections}
    child_conns = []
    for innov in sorted(set(genes1) | set(genes2)):
        if innov in genes1 and innov in genes2:
            chosen = random.choice([genes1[innov], genes2[innov]])
        elif innov in genes1:
            chosen = genes1[innov] if fit1 >= fit2 else None
        else:
            chosen = genes2[innov] if fit2 >= fit1 else None
        if chosen:
            child_conns.append(ConnectionGene(
                chosen.in_node, chosen.out_node,
                chosen.weight, chosen.enabled, chosen.innovation))
    nodes = set(p1.node_genes)
    for c in child_conns:
        nodes.update([c.in_node, c.out_node])
    return Genome(sorted(nodes), child_conns)

def compatibility(g1, g2, c1=1.0, c2=0.4):
    """Distance metric for speciation."""
    genes1 = {c.innovation: c for c in g1.connections}
    genes2 = {c.innovation: c for c in g2.connections}
    matching = set(genes1) & set(genes2)
    disjoint = len(set(genes1) ^ set(genes2))
    avg_w = (sum(abs(genes1[i].weight - genes2[i].weight)
             for i in matching) / max(len(matching), 1))
    N = max(len(g1.connections), len(g2.connections), 1)
    return c1 * disjoint / N + c2 * avg_w


if __name__ == "__main__":
    random.seed(42)
    # Create two genomes and demonstrate operations
    g1 = Genome([0, 1, 2, 3])
    g1.add_connection(0, 3)
    g1.add_connection(1, 3)
    g1.add_connection(2, 3)

    g2 = Genome([0, 1, 2, 3])
    g2.add_connection(0, 3)
    g2.add_connection(1, 3)
    g2.add_connection(2, 3)

    # Add a hidden node to g1
    g1.add_node()
    print(f"g1 nodes: {g1.node_genes}")
    print(f"g1 connections: {len(g1.connections)} "
          f"({sum(c.enabled for c in g1.connections)} enabled)")

    # Crossover
    child = crossover(g1, g2, fit1=3.5, fit2=2.8)
    print(f"Child nodes: {child.node_genes}")
    print(f"Child connections: {len(child.connections)}")

    # Compatibility distance
    dist = compatibility(g1, g2)
    print(f"Compatibility distance: {dist:.3f}")
