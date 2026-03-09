"""
InfoNCE Loss

Contrastive learning as energy-based softmax. The similarity function
is the negative energy, and the loss minimizes energy of positive pairs
while maximizing energy of negative pairs.

Blog post: https://dadops.dev/blog/energy-based-models-from-scratch/
"""
import numpy as np

def infonce_loss(query, keys_pos, keys_neg, temperature=0.1):
    """InfoNCE loss as energy-based softmax.

    The similarity function is the negative energy:
    sim(q, k) = -E(q, k) = q . k / tau
    """
    # Energy of positive pair (should be LOW = high similarity)
    energy_pos = -np.dot(query, keys_pos) / temperature

    # Energies of negative pairs (should be HIGH = low similarity)
    energies_neg = -np.dot(keys_neg, query) / temperature

    # InfoNCE = -log softmax over all energies
    # = -log [exp(-E_pos) / (exp(-E_pos) + sum exp(-E_neg))]
    all_energies = np.concatenate([[energy_pos], energies_neg])
    log_Z = np.log(np.sum(np.exp(-all_energies)))  # log partition function
    loss = energy_pos + log_Z  # = -log p(positive)

    return loss

# Example: query matches keys_pos, not keys_neg
np.random.seed(42)
query = np.array([1.0, 0.0, 0.0])
keys_pos = np.array([0.9, 0.1, 0.0])
keys_neg = np.random.randn(5, 3)

loss = infonce_loss(query, keys_pos, keys_neg)
print(f"InfoNCE loss: {loss:.3f}")
# Low loss when positive pair has much lower energy than negatives
