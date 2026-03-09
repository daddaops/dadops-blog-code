"""Random tree generation with Ramped Half-and-Half initialization."""
import numpy as np
from node import Node

FUNCTIONS = {'+': 2, '-': 2, '*': 2, '/': 2, 'sin': 1}
TERMINALS = ['x']  # plus ephemeral random constants


def random_terminal(rng):
    if rng.random() < 0.5:
        return Node('x')
    return Node(round(rng.uniform(-3, 3), 2))


def grow_tree(depth, max_depth, rng, full=False):
    """Recursively grow a random expression tree."""
    if depth == max_depth or (not full and depth > 0 and rng.random() < 0.3):
        return random_terminal(rng)
    func = rng.choice(list(FUNCTIONS.keys()))
    arity = FUNCTIONS[func]
    children = [grow_tree(depth + 1, max_depth, rng, full) for _ in range(arity)]
    return Node(func, children)


def init_population(pop_size, max_depth, rng):
    """Ramped Half-and-Half initialization."""
    population = []
    depths = range(2, max_depth + 1)
    for i in range(pop_size):
        d = depths[i % len(depths)]
        use_full = (i // len(depths)) % 2 == 0
        population.append(grow_tree(0, d, rng, full=use_full))
    return population


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    population = init_population(20, 5, rng)
    print(f"Generated {len(population)} trees:")
    for i, tree in enumerate(population):
        print(f"  [{i:2d}] size={tree.size():2d}  depth={tree.depth()}  expr={tree}")
