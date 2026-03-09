"""
Hopfield Network

Implements a Hopfield network with Hebbian learning and pattern recall.
Demonstrates energy minimization as associative memory.

Blog post: https://dadops.dev/blog/energy-based-models-from-scratch/
"""
import numpy as np

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def store(self, patterns):
        """Store patterns using Hebbian learning."""
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            self.W += np.outer(p, p) / self.n
        np.fill_diagonal(self.W, 0)  # no self-connections

    def energy(self, state):
        return -0.5 * state @ self.W @ state

    def recall(self, probe, max_steps=20):
        """Run async dynamics until convergence."""
        state = probe.copy()
        for step in range(max_steps):
            changed = False
            order = np.random.permutation(self.n)
            for i in order:
                h_i = self.W[i] @ state
                new_si = 1 if h_i >= 0 else -1
                if new_si != state[i]:
                    state[i] = new_si
                    changed = True
            if not changed:
                break
        return state

# Store 3 patterns in a 64-neuron network (8x8 grid)
np.random.seed(42)
net = HopfieldNetwork(64)
p1 = np.ones(64) * -1; p1[[3,4,11,12,19,20,27,28,35,36,43,44]] = 1  # vertical bar
p2 = np.ones(64) * -1; p2[[24,25,26,27,28,29,30,31]] = 1             # horizontal bar
p3 = np.ones(64) * -1; p3[[0,9,18,27,36,45,54,63]] = 1               # diagonal
net.store([p1, p2, p3])

# Corrupt p1 with 30% noise and recall
corrupted = p1.copy()
flip_idx = np.random.choice(64, size=19, replace=False)
corrupted[flip_idx] *= -1
recovered = net.recall(corrupted)
accuracy = np.mean(recovered == p1)
print(f"Recovery accuracy: {accuracy:.1%}")  # typically 95-100%
