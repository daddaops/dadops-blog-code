import numpy as np

def hopfield_store(patterns):
    """Store patterns via Hebbian learning (outer product rule)."""
    N = patterns[0].shape[0]
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    W /= len(patterns)
    np.fill_diagonal(W, 0)  # no self-connections
    return W

def hopfield_recall(W, state, max_steps=1000):
    """Asynchronous update until convergence."""
    N = len(state)
    x = state.copy()
    for step in range(max_steps):
        changed = False
        order = np.random.permutation(N)
        for i in order:
            h_i = W[i] @ x
            new_val = 1 if h_i >= 0 else -1
            if new_val != x[i]:
                x[i] = new_val
                changed = True
        if not changed:
            return x, step + 1
    return x, max_steps

if __name__ == "__main__":
    # Store 3 patterns of length 64 (imagine 8x8 images)
    np.random.seed(42)
    patterns = [np.random.choice([-1, 1], size=64) for _ in range(3)]
    W = hopfield_store(patterns)

    # Corrupt pattern 0 by flipping 25% of bits
    corrupted = patterns[0].copy()
    flip_idx = np.random.choice(64, size=16, replace=False)
    corrupted[flip_idx] *= -1

    recalled, steps = hopfield_recall(W, corrupted)
    accuracy = np.mean(recalled == patterns[0])
    print(f"Flipped 16/64 bits, recalled in {steps} sweeps")
    print(f"Recall accuracy: {accuracy:.1%}")
    # Flipped 16/64 bits, recalled in 2 sweeps
    # Recall accuracy: 100.0%
