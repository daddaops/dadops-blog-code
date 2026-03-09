"""Weighted Majority Algorithm for online expert advice.

Tracks 8 experts with varying accuracy over 200 rounds,
penalizing wrong experts multiplicatively.
"""
import numpy as np

np.random.seed(42)
T, N = 200, 8
accuracies = [0.85, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45]

# Ground truth and expert predictions
labels = np.random.randint(0, 2, T)
expert_preds = np.zeros((T, N), dtype=int)
for i, acc in enumerate(accuracies):
    correct = np.random.random(T) < acc
    expert_preds[:, i] = np.where(correct, labels, 1 - labels)

# Weighted Majority Algorithm
weights = np.ones(N)
epsilon = 0.3
algo_loss = 0

for t in range(T):
    # Weighted majority vote
    vote = np.zeros(2)
    for c in range(2):
        vote[c] = weights[expert_preds[t] == c].sum()
    pred = np.argmax(vote)

    # Update: penalize wrong experts
    algo_loss += int(pred != labels[t])
    wrong = expert_preds[t] != labels[t]
    weights[wrong] *= (1 - epsilon)

best = min(np.sum(expert_preds[:, i] != labels) for i in range(N))
print(f"Algorithm mistakes: {algo_loss}")
print(f"Best expert mistakes: {best}")
print(f"Regret: {algo_loss - best}")
print(f"Per-round regret: {(algo_loss - best) / T:.4f}")
# Algorithm: ~42 mistakes, Best expert: ~30, Regret: ~12
