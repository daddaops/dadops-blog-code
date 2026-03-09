import numpy as np
import time
from kernels import rbf_kernel
from gp_regression import gp_posterior

def sparse_gp_nystrom(X_train, y_train, X_test, kernel_fn,
                       n_inducing=50, noise_var=0.01, **kernel_params):
    """Sparse GP using Nyström approximation with inducing points."""
    n = len(X_train)
    # Select inducing points evenly spaced from training range
    indices = np.linspace(0, n - 1, n_inducing).astype(int)
    X_m = X_train[indices]

    K_mm = kernel_fn(X_m, X_m, **kernel_params) + 1e-6 * np.eye(n_inducing)
    K_nm = kernel_fn(X_train, X_m, **kernel_params)
    K_star_m = kernel_fn(X_test, X_m, **kernel_params)

    # Nyström: K ≈ K_nm @ inv(K_mm) @ K_mn
    L_mm = np.linalg.cholesky(K_mm)
    V = np.linalg.solve(L_mm, K_nm.T)  # m x n

    # Approximate posterior
    Lambda = V @ V.T + noise_var * np.eye(n_inducing)
    L_lam = np.linalg.cholesky(Lambda)
    alpha = np.linalg.solve(L_lam.T, np.linalg.solve(L_lam, V @ y_train))

    V_star = np.linalg.solve(L_mm, K_star_m.T)
    mu = V_star.T @ alpha

    w = np.linalg.solve(L_lam, V_star)
    K_ss_diag = np.array([kernel_fn(X_test[i:i+1], X_test[i:i+1], **kernel_params)[0,0]
                          for i in range(len(X_test))])
    var = K_ss_diag - np.sum(V_star**2, axis=0) + np.sum(w**2, axis=0)
    var = np.maximum(var, 1e-10)

    return mu, var

if __name__ == "__main__":
    # Generate a larger dataset
    np.random.seed(123)
    n_large = 2000
    X_large = np.sort(np.random.uniform(-5, 5, n_large)).reshape(-1, 1)
    y_large = np.sin(X_large).ravel() + 0.1 * np.random.randn(n_large)
    X_test_sparse = np.linspace(-5, 5, 200).reshape(-1, 1)

    # Sparse GP with 50 inducing points
    t0 = time.time()
    mu_sparse, var_sparse = sparse_gp_nystrom(
        X_large, y_large, X_test_sparse, rbf_kernel,
        n_inducing=50, noise_var=0.01, length_scale=1.0, signal_var=1.0)
    t_sparse = time.time() - t0

    # Exact GP on same data
    t0 = time.time()
    mu_exact, var_exact = gp_posterior(
        X_large, y_large, X_test_sparse, rbf_kernel,
        noise_var=0.01, length_scale=1.0, signal_var=1.0)
    t_exact = time.time() - t0

    rmse_sparse = np.sqrt(np.mean((mu_sparse - np.sin(X_test_sparse).ravel())**2))
    rmse_exact = np.sqrt(np.mean((mu_exact - np.sin(X_test_sparse).ravel())**2))

    print(f"Dataset size: {n_large} points")
    print(f"Inducing points: 50")
    print(f"\n{'Method':{' '}<20} | {'RMSE':<8} | {'Time':<8}")
    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}")
    print(f"{'Exact GP':{' '}<20} | {rmse_exact:.4f}  | {t_exact:.3f}s")
    print(f"{'Sparse GP (m=50)':{' '}<20} | {rmse_sparse:.4f}  | {t_sparse:.3f}s")
    print(f"\nSpeedup: {t_exact/t_sparse:.1f}x with "
          f"{abs(rmse_sparse - rmse_exact)/rmse_exact:.1%} RMSE difference")
