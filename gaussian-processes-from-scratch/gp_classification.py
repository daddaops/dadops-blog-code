import numpy as np
from kernels import rbf_kernel

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def gp_classification_laplace(X_train, y_train, X_test, kernel_fn,
                               n_iters=20, **kernel_params):
    """GP binary classification with Laplace approximation."""
    n = len(X_train)
    K = kernel_fn(X_train, X_train, **kernel_params)
    K += 1e-6 * np.eye(n)

    # Newton's method to find the MAP estimate of latent f
    f = np.zeros(n)
    for _ in range(n_iters):
        pi = sigmoid(f)
        W = pi * (1 - pi)  # diagonal of the Hessian
        W_sqrt = np.sqrt(W)

        B = np.eye(n) + np.outer(W_sqrt, W_sqrt) * K
        L = np.linalg.cholesky(B)
        b = W * f + (y_train - pi)
        a = b - W_sqrt * np.linalg.solve(
            L.T, np.linalg.solve(L, W_sqrt * (K @ b)))
        f = K @ a

    # Predict at test points
    pi_final = sigmoid(f)
    K_star = kernel_fn(X_test, X_train, **kernel_params)
    f_mean = K_star @ (y_train - pi_final)

    # Predictive variance (approximate)
    W_final = pi_final * (1 - pi_final)
    W_sqrt_f = np.sqrt(W_final)
    B_f = np.eye(n) + np.outer(W_sqrt_f, W_sqrt_f) * K
    L_f = np.linalg.cholesky(B_f)
    v = np.linalg.solve(L_f, (W_sqrt_f[:, None] * K_star.T))
    K_ss_diag = np.array([kernel_fn(X_test[i:i+1], X_test[i:i+1], **kernel_params)[0,0]
                          for i in range(len(X_test))])
    f_var = K_ss_diag - np.sum(v**2, axis=0)

    # Convert latent predictions to class probabilities
    # Probit approximation: integrate sigmoid against Gaussian
    kappa = 1.0 / np.sqrt(1.0 + np.pi * f_var / 8.0)
    prob = sigmoid(kappa * f_mean)

    return prob, f_mean, f_var

if __name__ == "__main__":
    # Synthetic 1D classification dataset
    np.random.seed(99)
    X_cls = np.sort(np.random.uniform(-4, 4, 20)).reshape(-1, 1)
    y_cls = (np.sin(X_cls.ravel()) > 0.3).astype(float)
    # Add some noise near boundary
    y_cls[8] = 1 - y_cls[8]

    X_test_cls = np.linspace(-5, 5, 100).reshape(-1, 1)
    probs, _, _ = gp_classification_laplace(
        X_cls, y_cls, X_test_cls, rbf_kernel,
        length_scale=1.0, signal_var=1.5)

    # Show predictions at key points
    print("  x    | GP prob | Confidence")
    print("-------+---------+-----------")
    for i in [5, 25, 50, 75, 95]:
        x = X_test_cls[i, 0]
        p = probs[i]
        conf = max(p, 1-p)
        label = "class 1" if p > 0.5 else "class 0"
        print(f"  {x:<4.1f} | {p:.3f}   | {conf:.1%} ({label})")
