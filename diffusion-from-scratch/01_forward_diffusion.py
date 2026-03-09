import numpy as np

def linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from Ho et al. (2020)"""
    return np.linspace(beta_start, beta_end, T)

# Precompute the cumulative products
betas = linear_schedule(T=1000)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)  # alpha_bar[t] = product of alphas[0..t]

def q_sample(x0, t, noise=None):
    """Sample from q(x_t | x_0) — jump to noise level t in one step"""
    if noise is None:
        noise = np.random.randn(*x0.shape)
    sqrt_ab = np.sqrt(alpha_bar[t])
    sqrt_1m_ab = np.sqrt(1.0 - alpha_bar[t])
    return sqrt_ab * x0 + sqrt_1m_ab * noise

# Generate a spiral dataset (our "images")
def make_spiral(n=300):
    theta = np.linspace(0, 4 * np.pi, n)
    r = theta / (4 * np.pi) * 2
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x += np.random.randn(n, 2) * 0.05  # tiny jitter
    return x.astype(np.float32)

data = make_spiral(300)

# Watch a single point dissolve
x0 = np.array([-0.75, 1.20])  # a point on the outer spiral
np.random.seed(42)
eps = np.random.randn(2)
for t in [0, 250, 500, 750, 999]:
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    snr = alpha_bar[t] / (1 - alpha_bar[t])
    print(f"t={t:4d}  alpha_bar={alpha_bar[t]:.4f}  SNR={snr:8.2f}  xt={xt}")
# t=   0  alpha_bar=0.9999  SNR=9999.00  xt=[-0.74  1.20]  (original)
# t= 250  alpha_bar=0.5214  SNR=   1.09  xt=[-0.20  0.77]  (signal ≈ noise)
# t= 500  alpha_bar=0.0778  SNR=   0.08  xt=[ 0.27  0.20]  (mostly noise)
# t= 750  alpha_bar=0.0033  SNR=   0.00  xt=[ 0.45 -0.07]  (nearly pure noise)
# t= 999  alpha_bar=0.0000  SNR=   0.00  xt=[ 0.49 -0.13]  (pure noise)
