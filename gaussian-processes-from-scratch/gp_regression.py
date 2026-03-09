import numpy as np
from kernels import rbf_kernel

def gp_posterior(X_train, y_train, X_test, kernel_fn,
                 noise_var=0.01, **kernel_params):
    """Compute GP posterior mean and variance at test points."""
    n = len(X_train)
    K = kernel_fn(X_train, X_train, **kernel_params)
    K_noisy = K + noise_var * np.eye(n)
    K_star = kernel_fn(X_test, X_train, **kernel_params)
    K_ss = kernel_fn(X_test, X_test, **kernel_params)

    # Cholesky solve: L @ L.T @ alpha = y
    L = np.linalg.cholesky(K_noisy)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Posterior mean
    mu = K_star @ alpha

    # Posterior variance
    v = np.linalg.solve(L, K_star.T)
    var = np.diag(K_ss) - np.sum(v**2, axis=0)
    var = np.maximum(var, 1e-10)  # clamp for numerical safety

    return mu, var

if __name__ == "__main__":
    # Generate noisy observations of sin(x)
    np.random.seed(7)
    X_train = np.array([-4, -2.5, -1.5, -0.5, 0.8, 2.0, 3.5]).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(len(X_train))

    X_test = np.linspace(-5, 5, 201).reshape(-1, 1)
    mu, var = gp_posterior(X_train, y_train, X_test, rbf_kernel,
                           noise_var=0.01, length_scale=1.0, signal_var=1.0)
    std = np.sqrt(var)

    # Show predictions at key points
    test_indices = [0, 50, 100, 150, 200]
    print("  x     | true   | GP mean | GP ±2σ")
    print("--------+--------+---------+----------")
    for i in test_indices:
        x = X_test[i, 0]
        print(f"  {x:<5.1f} | {np.sin(x):<6.3f} | {mu[i]:<7.3f} | ±{2*std[i]:.3f}")

    # RMSE comparison
    gp_rmse = np.sqrt(np.mean((mu - np.sin(X_test).ravel())**2))
    baseline_rmse = np.sqrt(np.mean((np.mean(y_train) - np.sin(X_test).ravel())**2))
    print(f"\nGP RMSE:       {gp_rmse:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
