import numpy as np
from hopfield_basic import hopfield_store, hopfield_recall

def capacity_experiment(N, max_patterns, trials=10):
    """Measure recall accuracy as pattern count increases."""
    ratios, accuracies = [], []
    for P in range(1, max_patterns + 1):
        acc_sum = 0
        for t in range(trials):
            np.random.seed(t * 1000 + P)
            pats = [np.random.choice([-1, 1], size=N) for _ in range(P)]
            W = hopfield_store(pats)
            # Test recall of each pattern from 10% corruption
            for pat in pats:
                corrupted = pat.copy()
                flips = np.random.choice(N, size=max(1, N // 10), replace=False)
                corrupted[flips] *= -1
                recalled, _ = hopfield_recall(W, corrupted, max_steps=50)
                acc_sum += np.mean(recalled == pat)
        avg_acc = acc_sum / (trials * P)
        ratios.append(P / N)
        accuracies.append(avg_acc)
    return ratios, accuracies

if __name__ == "__main__":
    ratios, accuracies = capacity_experiment(N=100, max_patterns=25, trials=5)
    for r, a in zip(ratios[::3], accuracies[::3]):
        print(f"P/N = {r:.2f}: accuracy = {a:.3f}")
    # P/N = 0.01: accuracy = 1.000
    # P/N = 0.04: accuracy = 1.000
    # P/N = 0.07: accuracy = 1.000
    # P/N = 0.10: accuracy = 0.995
    # P/N = 0.13: accuracy = 0.996
    # P/N = 0.16: accuracy = 0.960
    # P/N = 0.19: accuracy = 0.895
    # P/N = 0.22: accuracy = 0.855
    # P/N = 0.25: accuracy = 0.784
