"""Natural gradient vs vanilla GD for Gaussian fitting."""
import numpy as np

# Fit a 1D Gaussian N(mu, sigma^2) to data using natural gradient vs vanilla GD
np.random.seed(42)
data = np.random.normal(loc=3.0, scale=2.0, size=200)

def neg_log_likelihood(mu, log_sigma, data):
    sigma = np.exp(log_sigma)
    return 0.5 * np.mean(((data - mu) / sigma)**2) + log_sigma

def grad_nll(mu, log_sigma, data):
    sigma = np.exp(log_sigma)
    d_mu = -np.mean((data - mu) / sigma**2)
    d_log_sigma = 1.0 - np.mean(((data - mu) / sigma)**2)
    return np.array([d_mu, d_log_sigma])

def fisher_matrix(mu, log_sigma):
    """Analytical Fisher for Gaussian: F = diag(1/sigma^2, 2)."""
    sigma2 = np.exp(2 * log_sigma)
    return np.array([[1.0 / sigma2, 0.0],
                     [0.0,          2.0]])

# Vanilla gradient descent
params_gd = np.array([0.0, 0.0])  # mu=0, sigma=1
lr = 0.05
gd_path = [params_gd.copy()]
for _ in range(100):
    g = grad_nll(params_gd[0], params_gd[1], data)
    params_gd = params_gd - lr * g
    gd_path.append(params_gd.copy())

# Natural gradient descent
params_ng = np.array([0.0, 0.0])
ng_path = [params_ng.copy()]
for _ in range(100):
    g = grad_nll(params_ng[0], params_ng[1], data)
    F = fisher_matrix(params_ng[0], params_ng[1])
    nat_grad = np.linalg.solve(F, g)
    params_ng = params_ng - lr * nat_grad
    ng_path.append(params_ng.copy())

mu_gd, sig_gd = params_gd[0], np.exp(params_gd[1])
mu_ng, sig_ng = params_ng[0], np.exp(params_ng[1])
print(f"True: mu=3.00, sigma=2.00")
print(f"GD after 100 steps:  mu={mu_gd:.2f}, sigma={sig_gd:.2f}")
print(f"NatGrad after 100:   mu={mu_ng:.2f}, sigma={sig_ng:.2f}")
# Natural gradient reaches the true parameters much faster
