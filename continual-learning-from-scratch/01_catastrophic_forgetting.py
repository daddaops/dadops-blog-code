"""Block 1: Catastrophic Forgetting — train on two sequential tasks and watch task 1 collapse."""
import numpy as np
from shared import make_task, sigmoid, train_mlp, accuracy

if __name__ == "__main__":
    # Two tasks with different cluster locations
    X1, y1 = make_task([-2, -2], [2, 2], seed=42)   # diagonal clusters
    X2, y2 = make_task([-2, 2], [2, -2], seed=99)    # anti-diagonal

    rng = np.random.RandomState(0)
    W1 = rng.randn(2, 8) * 0.3; b1 = np.zeros(8)
    W2 = rng.randn(8, 1) * 0.3; b2 = np.zeros(1)

    W1, b1, W2, b2 = train_mlp(X1, y1, W1, b1, W2, b2)
    print(f"After Task 1: acc_task1={accuracy(X1, y1, W1, b1, W2, b2):.0%}")

    W1, b1, W2, b2 = train_mlp(X2, y2, W1, b1, W2, b2)
    print(f"After Task 2: acc_task1={accuracy(X1, y1, W1, b1, W2, b2):.0%}, "
          f"acc_task2={accuracy(X2, y2, W1, b1, W2, b2):.0%}")
    # Expected:
    # After Task 1: acc_task1=99%
    # After Task 2: acc_task1=50%, acc_task2=99%
