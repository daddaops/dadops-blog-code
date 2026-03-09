import numpy as np
from hopfield_basic import hopfield_store, hopfield_recall

def hopfield_energy(W, x):
    """Compute energy E = -0.5 * x^T W x."""
    return -0.5 * x @ W @ x

def recall_with_energy(W, state, max_steps=100):
    """Track energy at each sweep during recall."""
    N = len(state)
    x = state.copy()
    energies = [hopfield_energy(W, x)]
    for step in range(max_steps):
        changed = False
        order = np.random.permutation(N)
        for i in order:
            h_i = W[i] @ x
            new_val = 1 if h_i >= 0 else -1
            if new_val != x[i]:
                x[i] = new_val
                changed = True
        energies.append(hopfield_energy(W, x))
        if not changed:
            break
    return x, energies

if __name__ == "__main__":
    # Store 3 patterns
    np.random.seed(42)
    patterns = [np.random.choice([-1, 1], size=64) for _ in range(3)]
    W = hopfield_store(patterns)

    # Recall from 40% corruption to see energy descent
    np.random.seed(7)
    corrupted_heavy = patterns[0].copy()
    flip_idx = np.random.choice(64, size=26, replace=False)
    corrupted_heavy[flip_idx] *= -1

    recalled, energies = recall_with_energy(W, corrupted_heavy)
    for i, e in enumerate(energies):
        print(f"Sweep {i}: energy = {e:.2f}")
    # Sweep 0: energy = -16.00
    # Sweep 1: energy = -482.00
    # Sweep 2: energy = -482.00
