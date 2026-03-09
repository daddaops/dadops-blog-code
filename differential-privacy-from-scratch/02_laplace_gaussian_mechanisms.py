import numpy as np

def laplace_mechanism(true_value, sensitivity, epsilon):
    """Add Laplace noise for pure epsilon-DP."""
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

def gaussian_mechanism(true_value, sensitivity, epsilon, delta=1e-5):
    """Add Gaussian noise for (epsilon, delta)-DP."""
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma)
    return true_value + noise

# Example: private mean salary query
np.random.seed(42)
salaries = np.random.lognormal(mean=11.0, sigma=0.4, size=100)
salaries = np.clip(salaries, 30000, 200000)
true_mean = np.mean(salaries)
sensitivity = (200000 - 30000) / 100  # max change from one record

print(f"True mean salary: ${true_mean:,.0f}\n")
for eps in [0.1, 1.0, 10.0]:
    lap = laplace_mechanism(true_mean, sensitivity, eps)
    gau = gaussian_mechanism(true_mean, sensitivity, eps)
    print(f"  epsilon={eps:>4}  Laplace: ${lap:,.0f}  Gaussian: ${gau:,.0f}")
