import numpy as np

def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian RBF kernel between all pairs of rows in x and y."""
    xx = np.sum(x ** 2, axis=1, keepdims=True)
    yy = np.sum(y ** 2, axis=1, keepdims=True)
    dists_sq = xx - 2 * x @ y.T + yy.T
    return np.exp(-dists_sq / (2 * sigma ** 2))

def mmd_squared(source, target, sigma=1.0):
    """Unbiased estimate of squared MMD with Gaussian RBF kernel."""
    m, n = len(source), len(target)

    K_ss = gaussian_kernel(source, source, sigma)
    K_tt = gaussian_kernel(target, target, sigma)
    K_st = gaussian_kernel(source, target, sigma)

    # Unbiased: exclude diagonal terms for within-domain sums
    np.fill_diagonal(K_ss, 0)
    np.fill_diagonal(K_tt, 0)

    term1 = K_ss.sum() / (m * (m - 1))   # E[k(x, x')]
    term2 = K_tt.sum() / (n * (n - 1))   # E[k(y, y')]
    term3 = K_st.sum() / (m * n)          # E[k(x, y)]

    return term1 + term2 - 2 * term3

# Example: identical distributions should give MMD near 0
source = np.random.randn(300, 10)
target = np.random.randn(300, 10)
print(f"Same dist MMD^2: {mmd_squared(source, target):.4f}")

# Shifted distribution should give higher MMD
target_shifted = target + 1.0
print(f"Shifted MMD^2:   {mmd_squared(source, target_shifted):.4f}")
