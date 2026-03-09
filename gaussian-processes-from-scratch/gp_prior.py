import numpy as np
from kernels import rbf_kernel, matern32_kernel, periodic_kernel

np.random.seed(42)

def sample_gp_prior(kernel_fn, X, n_samples=5, **kernel_params):
    """Sample functions from a GP prior."""
    K = kernel_fn(X, X, **kernel_params)
    K += 1e-8 * np.eye(len(X))  # jitter for numerical stability
    L = np.linalg.cholesky(K)   # K = L @ L.T

    samples = []
    for _ in range(n_samples):
        z = np.random.randn(len(X))
        f = L @ z  # transform standard normal to GP sample
        samples.append(f)
    return samples

if __name__ == "__main__":
    X = np.linspace(-5, 5, 200).reshape(-1, 1)

    rbf_samples = sample_gp_prior(rbf_kernel, X, n_samples=3, length_scale=1.0)
    matern_samples = sample_gp_prior(matern32_kernel, X, n_samples=3, length_scale=1.0)
    periodic_samples = sample_gp_prior(periodic_kernel, X, n_samples=3, length_scale=1.0, period=2.0)

    # Show a few values from each sample type
    print("RBF sample (infinitely smooth):")
    print(f"  f([-3, -1, 0, 1, 3]) = [{rbf_samples[0][40]:.2f}, {rbf_samples[0][80]:.2f}, "
          f"{rbf_samples[0][100]:.2f}, {rbf_samples[0][120]:.2f}, {rbf_samples[0][160]:.2f}]")

    print("Matérn 3/2 sample (realistically rough):")
    print(f"  f([-3, -1, 0, 1, 3]) = [{matern_samples[0][40]:.2f}, {matern_samples[0][80]:.2f}, "
          f"{matern_samples[0][100]:.2f}, {matern_samples[0][120]:.2f}, {matern_samples[0][160]:.2f}]")

    print("Periodic sample (repeating pattern, period=2):")
    print(f"  f([-3, -1, 0, 1, 3]) = [{periodic_samples[0][40]:.2f}, {periodic_samples[0][80]:.2f}, "
          f"{periodic_samples[0][100]:.2f}, {periodic_samples[0][120]:.2f}, {periodic_samples[0][160]:.2f}]")
