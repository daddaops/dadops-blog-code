import numpy as np

# --- Needed from block 1 ---
def linear_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from DDPM (Ho et al. 2020)."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def ddpm_sample(model, shape, T, betas, alphas, alpha_bars):
    """DDPM sampling: full 1000-step reverse process."""
    x = np.random.randn(*shape)  # start from pure noise

    for t in reversed(range(T)):
        noise_pred = model(x, t)

        # Compute x_{t-1} from x_t and predicted noise
        coeff1 = 1.0 / np.sqrt(alphas[t])
        coeff2 = betas[t] / np.sqrt(1.0 - alpha_bars[t])
        mean = coeff1 * (x - coeff2 * noise_pred)

        if t > 0:
            # Add stochastic noise (except at final step)
            sigma = np.sqrt(betas[t])
            x = mean + sigma * np.random.randn(*shape)
        else:
            x = mean

    return x

def ddim_sample(model, shape, T, alpha_bars, num_steps=50):
    """DDIM sampling: skip timesteps for faster generation.

    Implements the deterministic (eta=0) variant. With eta > 0 you'd add
    scaled noise at each step, recovering DDPM stochasticity at eta=1.
    """
    # Create a subsequence of timesteps (e.g., 50 evenly spaced)
    step_size = T // num_steps
    timesteps = list(range(0, T, step_size))[::-1]

    x = np.random.randn(*shape)

    for i, t in enumerate(timesteps):
        noise_pred = model(x, t)
        a_bar_t = alpha_bars[t]

        # Predict x_0 from x_t and predicted noise
        x0_pred = (x - np.sqrt(1 - a_bar_t) * noise_pred) / np.sqrt(a_bar_t)
        x0_pred = np.clip(x0_pred, -1, 1)  # stability clamp

        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            a_bar_prev = alpha_bars[t_prev]

            # Compute the "direction pointing to x_t"
            direction = np.sqrt(1 - a_bar_prev) * noise_pred

            # Jump to x_{t_prev}
            x = np.sqrt(a_bar_prev) * x0_pred + direction
        else:
            x = x0_pred

    return x

# DDPM: 1000 model calls -> high quality, slow
# DDIM:   50 model calls -> near-identical quality, 20x faster


# --- Demo: run both samplers with a dummy model ---
T = 100  # small T for demo speed
betas, alphas, alpha_bars = linear_noise_schedule(T=T)

# Dummy model that returns zeros (no denoising, just shows the mechanics)
def dummy_model(x, t):
    return np.zeros_like(x)

shape = (1, 8, 8)

print("Running DDPM sampling (100 steps)...")
sample_ddpm = ddpm_sample(dummy_model, shape, T, betas, alphas, alpha_bars)
print(f"  Output shape: {sample_ddpm.shape}, mean: {sample_ddpm.mean():.4f}, std: {sample_ddpm.std():.4f}")

print("\nRunning DDIM sampling (20 steps from 100)...")
sample_ddim = ddim_sample(dummy_model, shape, T, alpha_bars, num_steps=20)
print(f"  Output shape: {sample_ddim.shape}, mean: {sample_ddim.mean():.4f}, std: {sample_ddim.std():.4f}")

print("\nDDPM used 100 model calls, DDIM used 20 model calls (5x faster)")
