import numpy as np
from collections import Counter

def gini_impurity(y):
    """Gini = 1 - sum(p_i^2) for each class."""
    counts = Counter(y)
    n = len(y)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(y):
    """H = -sum(p_i * log2(p_i)) for each class."""
    counts = Counter(y)
    n = len(y)
    if n == 0:
        return 0.0
    return sum(-(c / n) * np.log2(c / n)
               for c in counts.values() if c > 0)

def information_gain(y_parent, y_left, y_right, criterion=gini_impurity):
    """Weighted impurity reduction from a split."""
    n = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)
    return criterion(y_parent) - (
        n_l / n * criterion(y_left) + n_r / n * criterion(y_right)
    )

# A node with 30 cats and 10 dogs
y = [0]*30 + [1]*10
print(f"Gini:    {gini_impurity(y):.4f}")   # 0.3750
print(f"Entropy: {entropy(y):.4f}")          # 0.8113

# Split into [28 cats, 2 dogs] | [2 cats, 8 dogs]
y_left  = [0]*28 + [1]*2
y_right = [0]*2  + [1]*8
print(f"IG (Gini):    {information_gain(y, y_left, y_right):.4f}")
print(f"IG (Entropy): {information_gain(y, y_left, y_right, entropy):.4f}")
