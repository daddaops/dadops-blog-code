import numpy as np
import time

# Two random vectors
rng = np.random.RandomState(0)
x = rng.randn(500)
y = rng.randn(500)

# Method 1: Explicit degree-2 features, then dot product
from itertools import combinations_with_replacement
from collections import Counter
from math import factorial

def explicit_poly_dot(x, y, degree=2):
    """Compute dot product in polynomial feature space explicitly."""
    features_x, features_y = [], []
    d = len(x)
    for deg in range(degree + 1):
        for combo in combinations_with_replacement(range(d), deg):
            val_x = val_y = 1.0
            for idx in combo:
                val_x *= x[idx]
                val_y *= y[idx]
            counts = Counter(combo)
            coeff = factorial(deg)
            for c in counts.values():
                coeff //= factorial(c)
            features_x.append(val_x * np.sqrt(coeff))
            features_y.append(val_y * np.sqrt(coeff))
    return np.dot(features_x, features_y)

# Method 2: Polynomial kernel (one line!)
def poly_kernel(x, y, degree=2, c=1.0):
    return (np.dot(x, y) + c) ** degree

# Verify they match
explicit = explicit_poly_dot(x, y, degree=2)
kernel = poly_kernel(x, y, degree=2)
print(f"Explicit:  {explicit:.6f}")
print(f"Kernel:    {kernel:.6f}")
print(f"Match:     {abs(explicit - kernel) < 1e-6}")

# Timing comparison
t0 = time.time()
for _ in range(10):
    explicit_poly_dot(x, y, degree=2)
t_explicit = (time.time() - t0) / 10

t0 = time.time()
for _ in range(10000):
    poly_kernel(x, y, degree=2)
t_kernel = (time.time() - t0) / 10000

print(f"\nExplicit: {t_explicit*1000:.2f} ms | Kernel: {t_kernel*1000:.4f} ms")
print(f"Kernel is {t_explicit/t_kernel:.0f}x faster")
