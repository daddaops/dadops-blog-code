import numpy as np

def linear_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from DDPM (Ho et al. 2020)."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)  # cumulative product
    return betas, alphas, alpha_bars

def forward_diffusion(x_0, t, alpha_bars):
    """Add noise to x_0 at timestep t using the closed-form shortcut.

    Returns the noised sample x_t and the noise that was added.
    """
    noise = np.random.randn(*x_0.shape)
    a_bar = alpha_bars[t]
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    x_t = np.sqrt(a_bar) * x_0 + np.sqrt(1 - a_bar) * noise
    return x_t, noise

# Demonstrate: noise an image at various timesteps
betas, alphas, alpha_bars = linear_noise_schedule(T=1000)

# Simulate a simple 8x8 "image" (checkerboard pattern)
x_0 = np.zeros((8, 8))
x_0[::2, ::2] = 1.0
x_0[1::2, 1::2] = 1.0

print("Signal preserved at each timestep (alpha_bar):")
for t in [0, 249, 499, 749, 999]:
    x_t, _ = forward_diffusion(x_0, t, alpha_bars)
    print(f"  t={t:4d}: alpha_bar={alpha_bars[t]:.4f}, "
          f"signal={np.sqrt(alpha_bars[t]):.3f}, "
          f"noise={np.sqrt(1 - alpha_bars[t]):.3f}")
# t=   0: alpha_bar=0.9999, signal=1.000, noise=0.010
# t= 249: alpha_bar=0.5241, signal=0.724, noise=0.690
# t= 499: alpha_bar=0.0786, signal=0.280, noise=0.960
# t= 749: alpha_bar=0.0034, signal=0.058, noise=0.998
# t= 999: alpha_bar=0.0000, signal=0.006, noise=1.000
