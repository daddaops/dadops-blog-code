import numpy as np
from scipy.optimize import minimize
from kernels import rbf_kernel
from gp_regression import gp_posterior

def neg_log_marginal_likelihood(params, X, y):
    """Negative log marginal likelihood for RBF kernel GP."""
    length_scale = np.exp(params[0])  # optimize in log-space
    signal_var = np.exp(params[1])
    noise_var = np.exp(params[2])

    K = rbf_kernel(X, X, length_scale=length_scale, signal_var=signal_var)
    K_y = K + noise_var * np.eye(len(X))

    try:
        L = np.linalg.cholesky(K_y)
    except np.linalg.LinAlgError:
        return 1e6  # return large value if not positive definite

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det = 2 * np.sum(np.log(np.diag(L)))

    nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * len(X) * np.log(2 * np.pi)
    return nll

if __name__ == "__main__":
    # Generate noisy observations of sin(x)
    np.random.seed(7)
    X_train = np.array([-4, -2.5, -1.5, -0.5, 0.8, 2.0, 3.5]).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(len(X_train))
    X_test = np.linspace(-5, 5, 201).reshape(-1, 1)

    # Start from poor hyperparameters
    init_params = np.log([0.5, 0.5, 0.1])  # l=0.5, sv=0.5, nv=0.1
    result = minimize(neg_log_marginal_likelihood, init_params,
                      args=(X_train, y_train), method='L-BFGS-B')
    opt_params = np.exp(result.x)

    print("Initial hyperparameters (poor guesses):")
    print(f"  length_scale = 0.500, signal_var = 0.500, noise_var = 0.100")
    print(f"\nOptimized hyperparameters:")
    print(f"  length_scale = {opt_params[0]:.3f}, signal_var = {opt_params[1]:.3f}, "
          f"noise_var = {opt_params[2]:.6f}")

    # Compare predictions
    mu_bad, var_bad = gp_posterior(X_train, y_train, X_test, rbf_kernel,
                                   noise_var=0.1, length_scale=0.5, signal_var=0.5)
    mu_opt, var_opt = gp_posterior(X_train, y_train, X_test, rbf_kernel,
                                   noise_var=opt_params[2],
                                   length_scale=opt_params[0],
                                   signal_var=opt_params[1])

    rmse_bad = np.sqrt(np.mean((mu_bad - np.sin(X_test).ravel())**2))
    rmse_opt = np.sqrt(np.mean((mu_opt - np.sin(X_test).ravel())**2))
    print(f"\nRMSE with poor hyperparams:      {rmse_bad:.4f}")
    print(f"RMSE with optimized hyperparams: {rmse_opt:.4f}")
