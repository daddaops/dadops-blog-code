"""Genetic operators: selection, crossover, and mutation."""
import copy
import numpy as np
from node import Node
from tree_gen import FUNCTIONS, random_terminal, grow_tree, init_population


def get_random_node(tree, rng):
    """Return a random node and its parent (None if root)."""
    nodes = []
    def collect(node, parent, idx):
        nodes.append((node, parent, idx))
        for i, c in enumerate(node.children):
            collect(c, node, i)
    collect(tree, None, 0)
    return nodes[rng.randint(len(nodes))]


def tournament_select(population, fitnesses, k, rng):
    idxs = rng.choice(len(population), size=k, replace=False)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] < fitnesses[best]:
            best = i
    return copy.deepcopy(population[best])


def subtree_crossover(p1, p2, rng, max_depth=8):
    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    n1, par1, idx1 = get_random_node(c1, rng)
    n2, par2, idx2 = get_random_node(c2, rng)
    if par1 is None:
        c1 = n2
    else:
        par1.children[idx1] = n2
    if c1.depth() > max_depth:
        return copy.deepcopy(p1)  # reject if too deep
    return c1


def point_mutation(tree, rng):
    tree = copy.deepcopy(tree)
    node, parent, idx = get_random_node(tree, rng)
    if not node.children:
        new = random_terminal(rng)
        if parent is None:
            return new
        parent.children[idx] = new
    else:
        arity = len(node.children)
        candidates = [f for f, a in FUNCTIONS.items() if a == arity]
        node.value = rng.choice(candidates)
    return tree


def hoist_mutation(tree, rng):
    tree = copy.deepcopy(tree)
    node, parent, idx = get_random_node(tree, rng)
    if node.children:
        child_idx = rng.randint(len(node.children))
        if parent is None:
            return node.children[child_idx]
        parent.children[idx] = node.children[child_idx]
    return tree


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    pop = init_population(10, 4, rng)

    # Demo crossover
    p1, p2 = pop[0], pop[1]
    print("Parent 1:", p1)
    print("Parent 2:", p2)
    child = subtree_crossover(p1, p2, rng)
    print("Crossover child:", child)

    # Demo point mutation
    mutant = point_mutation(p1, rng)
    print("\nOriginal:", p1)
    print("Point mutation:", mutant)

    # Demo hoist mutation
    hoisted = hoist_mutation(pop[3], rng)
    print("\nOriginal:", pop[3])
    print("Hoist mutation:", hoisted)

    # Demo tournament selection
    fitnesses = [rng.random() for _ in pop]
    winner = tournament_select(pop, fitnesses, 3, rng)
    print("\nTournament winner:", winner)
