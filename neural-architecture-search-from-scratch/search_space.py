"""Cell-based architecture search space.

Defines operations, random architecture sampling, and synthetic
fitness evaluation used across all NAS methods.
"""
import numpy as np

# Cell-based search space: a DAG with N intermediate nodes
# Each node picks 2 inputs and an operation for each input
OPERATIONS = ['conv3x3', 'conv5x5', 'maxpool', 'skip', 'none']
NUM_NODES = 4  # intermediate nodes in one cell

def sample_architecture(rng):
    """Sample a random cell architecture."""
    cell = []
    for node_idx in range(NUM_NODES):
        # Each node picks 2 inputs from {input_0, input_1, prev_nodes...}
        num_inputs = node_idx + 2  # node 0 can pick from 2, node 1 from 3, etc.
        input_1 = rng.integers(0, num_inputs)
        input_2 = rng.integers(0, num_inputs)
        op_1 = str(rng.choice(OPERATIONS))
        op_2 = str(rng.choice(OPERATIONS))
        cell.append(((input_1, op_1), (input_2, op_2)))
    return cell

def synthetic_fitness(arch, max_epochs, rng):
    """Simulate training: architecture quality + noise that decreases with epochs."""
    # True quality: sum of operation 'goodness' scores
    op_scores = {'conv3x3': 0.8, 'conv5x5': 0.7, 'maxpool': 0.3,
                 'skip': 0.5, 'none': 0.0}
    quality = sum(op_scores[op] for node in arch for (_, op) in node)
    quality = quality / (2 * len(arch))  # normalize to [0, 1]
    # Noisy estimate: noise shrinks with more training
    noise = rng.normal(0, 0.15 / np.sqrt(max_epochs))
    return np.clip(quality + noise, 0, 1)


if __name__ == "__main__":
    # How big is this search space?
    total = 1
    for node_idx in range(NUM_NODES):
        num_inputs = node_idx + 2
        choices_per_node = (num_inputs * len(OPERATIONS)) ** 2
        total *= choices_per_node
        print(f"Node {node_idx}: {num_inputs} inputs x {len(OPERATIONS)} ops "
              f"= {choices_per_node} combos")
    print(f"\nTotal search space: {total:,} architectures")
