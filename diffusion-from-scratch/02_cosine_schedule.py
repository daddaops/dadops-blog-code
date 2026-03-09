import numpy as np

def linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from Ho et al. (2020)"""
    return np.linspace(beta_start, beta_end, T)

def cosine_schedule(T=1000, s=0.008):
    """Cosine noise schedule from Nichol & Dhariwal (2021)"""
    steps = np.arange(T + 1)
    f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    # Clip to prevent numerical issues
    alpha_bar = np.clip(alpha_bar, 1e-5, 1.0)
    # Derive betas from alpha_bar
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas = np.clip(betas, 1e-5, 0.999)
    return betas, alpha_bar[1:]

betas_cos, alpha_bar_cos = cosine_schedule()
betas_lin = linear_schedule()
alpha_bar_lin = np.cumprod(1.0 - betas_lin)

# Compare SNR at key timesteps
for t in [0, 250, 500, 750, 999]:
    snr_lin = alpha_bar_lin[t] / (1 - alpha_bar_lin[t])
    snr_cos = alpha_bar_cos[t] / (1 - alpha_bar_cos[t])
    print(f"t={t:4d}  linear SNR={snr_lin:8.2f}  cosine SNR={snr_cos:8.2f}")
# t=   0  linear SNR= 9999.00  cosine SNR=24221.33
# t= 250  linear SNR=    1.09  cosine SNR=    5.49
# t= 500  linear SNR=    0.08  cosine SNR=    0.97
# t= 750  linear SNR=    0.00  cosine SNR=    0.17
# t= 999  linear SNR=    0.00  cosine SNR=    0.00
