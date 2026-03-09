import numpy as np
from one_nn import euclidean_distance

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def minkowski_distance(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1.0 / p)

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))

def cosine_distance(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - dot / (norm + 1e-10)

# Compare metrics on the same two points
a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
print(f"Euclidean:  {euclidean_distance(a, b):.3f}")   # 1.414
print(f"Manhattan:  {manhattan_distance(a, b):.3f}")    # 2.000
print(f"Minkowski3: {minkowski_distance(a, b, p=3):.3f}")  # 1.260
print(f"Chebyshev:  {chebyshev_distance(a, b):.3f}")   # 1.000
print(f"Cosine:     {cosine_distance(a, b):.3f}")      # 1.000

# The feature scaling problem
# Feature 1: house size in sqft [500, 5000]
# Feature 2: number of bedrooms [1, 6]
houses = np.array([[2000, 3], [2100, 5], [500, 3]], dtype=float)

# Without scaling: sqft dominates everything
d_unscaled = euclidean_distance(houses[0], houses[1])  # ~100 (sqft diff)
d_unscaled2 = euclidean_distance(houses[0], houses[2])  # ~1500 (sqft diff)
print(f"Unscaled: d(house0, house1)={d_unscaled:.0f}")
print(f"Unscaled: d(house0, house2)={d_unscaled2:.0f}")
# House 1 is "closer" — bedrooms are invisible

# With standardization: both features contribute equally
mean = houses.mean(axis=0)
std = houses.std(axis=0)
houses_scaled = (houses - mean) / std

d_scaled = euclidean_distance(houses_scaled[0], houses_scaled[1])
d_scaled2 = euclidean_distance(houses_scaled[0], houses_scaled[2])
print(f"Scaled:   d(house0, house1)={d_scaled:.2f}")
print(f"Scaled:   d(house0, house2)={d_scaled2:.2f}")
# Now bedrooms matter — the "nearest" neighbor may change
