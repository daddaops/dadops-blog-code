import numpy as np

# --- Reuse dataset from block 1 ---
def make_curriculum_dataset(n=200, seed=42):
    rng = np.random.RandomState(seed)
    x0 = rng.randn(n // 2, 2) * 0.8 + np.array([-2, 0])
    x1 = rng.randn(n // 2, 2) * 0.8 + np.array([2, 0])
    X = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    difficulty = 1.0 / (np.abs(X[:, 0]) + 0.1)
    difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min())
    return X, y, difficulty

X, y, difficulty = make_curriculum_dataset()
# --- End dataset setup ---

def compute_difficulty_metrics(X, y, n_epochs=10, seed=42):
    rng = np.random.RandomState(seed)
    n = len(X)

    # Simple linear model: w^T x + b
    w = rng.randn(2) * 0.01
    b = 0.0
    lr = 0.05

    losses_over_time = np.zeros((n_epochs, n))
    forgetting_counts = np.zeros(n)
    prev_correct = np.zeros(n, dtype=bool)

    for epoch in range(n_epochs):
        for i in range(n):
            logit = X[i] @ w + b
            prob = 1.0 / (1.0 + np.exp(-logit))
            loss = -y[i] * np.log(prob + 1e-8) - (1 - y[i]) * np.log(1 - prob + 1e-8)
            losses_over_time[epoch, i] = loss

            # Track forgetting events
            correct = (prob > 0.5) == y[i]
            if epoch > 0 and prev_correct[i] and not correct:
                forgetting_counts[i] += 1
            prev_correct[i] = correct

            # SGD update
            grad = prob - y[i]
            w -= lr * grad * X[i]
            b -= lr * grad

    avg_loss = losses_over_time.mean(axis=0)        # Loss-based difficulty
    margin_dist = np.abs(X[:, 0])                    # Margin-based difficulty
    return avg_loss, 1.0 / (margin_dist + 0.1), forgetting_counts

avg_loss, margin_diff, forgetting = compute_difficulty_metrics(X, y)
# All three metrics correlate: examples near the boundary
# have high avg_loss, high margin_diff, and more forgetting events
