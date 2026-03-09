import numpy as np

def exponential_mechanism(scores, epsilon, sensitivity):
    """Sample a candidate proportional to exp(eps * score / (2*sensitivity)).
    Returns the index of the selected candidate."""
    log_probs = (epsilon * np.array(scores)) / (2 * sensitivity)
    log_probs -= np.max(log_probs)  # numerical stability
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return np.random.choice(len(scores), p=probs)

# Private mode: find the most common color without revealing individuals
np.random.seed(42)
candidates = ["red", "blue", "green", "yellow"]
true_counts = [5, 4, 3, 2]  # 14 people surveyed

# Run 1000 trials to see the selection distribution
results = np.zeros(4)
for _ in range(1000):
    idx = exponential_mechanism(true_counts, epsilon=1.0, sensitivity=1)
    results[idx] += 1

print("True counts:", dict(zip(candidates, true_counts)))
print(f"\nSelection frequency (epsilon=1.0, 1000 trials):")
for i, c in enumerate(candidates):
    print(f"  {c:>6}: {results[i]/10:.1f}%")
