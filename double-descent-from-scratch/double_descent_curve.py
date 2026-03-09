"""
Double Descent Curve with Random Fourier Features

Uses random Fourier features (cos(omega*x + b)) instead of Vandermonde polynomials
to avoid numerical instability at high feature counts. Fits models with p=1 to p=120
features on n=30 noisy sine-wave data points to show the three regimes:
underfitting, interpolation peak, overparameterized descent.
"""
import numpy as np

np.random.seed(42)
n = 30
x_train = np.linspace(0, 2 * np.pi, n)
y_train = np.sin(x_train) + np.random.randn(n) * 0.3
x_test = np.linspace(0, 2 * np.pi, 200)
y_test = np.sin(x_test)

# Fixed random frequencies and phases for reproducibility
rng = np.random.RandomState(123)
max_p = 120
omegas = rng.randn(max_p)
biases = rng.uniform(0, 2 * np.pi, max_p)


def make_features(x, p):
    """Random Fourier features: phi_j(x) = cos(omega_j * x + b_j)"""
    return np.cos(np.outer(x, omegas[:p]) + biases[:p])


feature_counts = list(range(1, 25)) + list(range(25, 121, 3))
train_errors, test_errors = [], []

for p in feature_counts:
    X_tr = make_features(x_train, p)
    X_te = make_features(x_test, p)

    if p < n:
        # Overdetermined: least-squares solution
        w, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
    else:
        # Underdetermined: minimum-norm solution w = X^T (X X^T)^{-1} y
        w = X_tr.T @ np.linalg.solve(X_tr @ X_tr.T + 1e-12 * np.eye(n), y_train)

    train_errors.append(np.mean((X_tr @ w - y_train) ** 2))
    test_errors.append(np.mean((X_te @ w - y_test) ** 2))

# Show key values from each regime
print("Double Descent Curve — Random Fourier Features (n=30)")
print("=" * 65)
for p, tr, te in zip(feature_counts, train_errors, test_errors):
    regime = "UNDER" if p < 20 else ("CRITICAL" if 25 <= p <= 35 else "OVER")
    if p in [3, 10, 20, 28, 30, 34, 50, 76, 100]:
        print(f"p={p:>3d}  train={tr:.4f}  test={te:.4f}  [{regime}]")

# Summary: find peak near interpolation threshold (p near n)
near_threshold = [(p, te) for p, te in zip(feature_counts, test_errors) if 15 <= p <= 35]
peak_p, peak_te = max(near_threshold, key=lambda v: v[1])
best_classical = min((te, p) for p, te in zip(feature_counts, test_errors) if 5 <= p <= 18)
best_over = min((te, p) for p, te in zip(feature_counts, test_errors) if p > 35)
print(f"\nBest classical (p<n): test={best_classical[0]:.4f} at p={best_classical[1]}")
print(f"Interpolation peak:   test={peak_te:.4f} at p={peak_p}")
print(f"Best overparameterized: test={best_over[0]:.4f} at p={best_over[1]}")
print(f"Final test error:     test={test_errors[-1]:.4f} at p={feature_counts[-1]}")


if __name__ == "__main__":
    pass  # Output printed above
