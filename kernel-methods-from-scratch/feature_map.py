import numpy as np
from itertools import combinations_with_replacement
from collections import Counter
from math import comb, factorial

# Generate XOR-like data: label = sign(x1 * x2)
rng = np.random.RandomState(42)
X = rng.randn(200, 2)
y = np.sign(X[:, 0] * X[:, 1])  # +1 in Q1/Q3, -1 in Q2/Q4

# Explicit degree-2 polynomial feature map
# [x1, x2] -> [x1^2, sqrt(2)*x1*x2, x2^2, sqrt(2)*x1, sqrt(2)*x2, 1]
def poly_features(X, degree=2):
    n, d = X.shape
    features = []
    for deg in range(degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for idx in combo:
                col *= X[:, idx]
            # Multinomial coefficient for proper inner product
            counts = Counter(combo)
            coeff = np.sqrt(factorial(deg) /
                    np.prod([factorial(c) for c in counts.values()]))
            features.append(col * coeff)
    return np.column_stack(features)

phi = poly_features(X, degree=2)  # 200 x 6

# Linear classifier in lifted space: ridge regression
lam = 0.01
w = np.linalg.solve(phi.T @ phi + lam * np.eye(phi.shape[1]), phi.T @ y)
preds = np.sign(phi @ w)
print(f"Accuracy in lifted space: {np.mean(preds == y):.1%}")  # ~100%

# Dimensionality explosion: 10 input features at increasing degrees
for deg in [2, 3, 5, 10]:
    print(f"  Degree {deg}: {comb(10 + deg, deg):,} dimensions")
print(f"  Degree 10 on 50 features: {comb(50 + 10, 10):,} dimensions")
