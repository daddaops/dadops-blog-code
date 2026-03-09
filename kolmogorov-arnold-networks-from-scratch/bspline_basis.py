import numpy as np

def bspline_basis(x, knots, i, k):
    """Cox-de Boor recursion for B-spline basis function B_{i,k}(x)."""
    if k == 0:
        return np.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)

    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]

    term1 = 0.0 if denom1 == 0 else (x - knots[i]) / denom1 * bspline_basis(x, knots, i, k - 1)
    term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - x) / denom2 * bspline_basis(x, knots, i + 1, k - 1)
    return term1 + term2

# Build a cubic B-spline (k=3) on G=5 grid intervals
G, k = 5, 3
interior_knots = np.linspace(0, 1, G + 1)
knots = np.concatenate([np.zeros(k), interior_knots, np.ones(k)])  # augmented knot vector
n_basis = G + k  # 8 basis functions

x = np.linspace(0, 0.999, 200)  # avoid right endpoint for open knot vector

# Evaluate all basis functions
basis_vals = np.array([bspline_basis(x, knots, i, k) for i in range(n_basis)])
print(f"Grid intervals: {G}, Degree: {k}, Basis functions: {n_basis}")
print(f"Partition of unity check (should be ~1.0): {basis_vals.sum(axis=0).mean():.4f}")

# Approximate sin(2*pi*x) using learned coefficients
target = np.sin(2 * np.pi * x)
coeffs = np.linalg.lstsq(basis_vals.T, target, rcond=None)[0]  # least-squares fit
approx = coeffs @ basis_vals
print(f"Max approximation error for sin(2*pi*x): {np.max(np.abs(target - approx)):.4f}")
# Output: Max approximation error for sin(2*pi*x): 0.0046  (improves with larger G)
