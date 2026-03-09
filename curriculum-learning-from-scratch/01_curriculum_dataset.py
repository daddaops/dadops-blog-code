import numpy as np

def make_curriculum_dataset(n=200, seed=42):
    rng = np.random.RandomState(seed)

    # Class 0: cluster centered at (-1, 0)
    x0 = rng.randn(n // 2, 2) * 0.8 + np.array([-1, 0])
    # Class 1: cluster centered at (+1, 0)
    x1 = rng.randn(n // 2, 2) * 0.8 + np.array([1, 0])

    X = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # Difficulty = inverse distance to decision boundary (x=0 line)
    difficulty = 1.0 / (np.abs(X[:, 0]) + 0.1)

    # Normalize to [0, 1] range
    difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min())

    return X, y, difficulty

X, y, difficulty = make_curriculum_dataset()
easy_mask = difficulty < 0.3   # 70% of examples are "easy"
hard_mask = difficulty >= 0.3  # 30% are near the boundary

print(f"Easy examples: {easy_mask.sum()}, Hard examples: {hard_mask.sum()}")
# Easy examples: 185, Hard examples: 15
