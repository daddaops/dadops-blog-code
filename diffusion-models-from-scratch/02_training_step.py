import numpy as np

# --- Needed from block 1 ---
def linear_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from DDPM (Ho et al. 2020)."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def train_step(model, optimizer, x_0, alpha_bars, T=1000):
    """One DDPM training step (Algorithm 1 from Ho et al. 2020).

    model: neural network that takes (x_t, t) and predicts noise
    optimizer: gradient-based optimizer (Adam, etc.)
    x_0: batch of clean training images, shape [B, C, H, W]
    """
    batch_size = x_0.shape[0]

    # 1. Sample random timesteps for each image in the batch
    t = np.random.randint(0, T, size=batch_size)

    # 2. Sample random noise
    noise = np.random.randn(*x_0.shape)

    # 3. Compute noised images using the closed-form shortcut
    a_bar = alpha_bars[t]  # shape: [B]
    # Reshape for broadcasting: [B, 1, 1, 1] for image tensors
    a_bar = a_bar.reshape(-1, 1, 1, 1)
    x_t = np.sqrt(a_bar) * x_0 + np.sqrt(1 - a_bar) * noise

    # 4. Network predicts the noise
    noise_pred = model(x_t, t)

    # 5. Loss = MSE between true noise and predicted noise
    loss = np.mean((noise - noise_pred) ** 2)

    # 6. Backpropagate and update (pseudo-code for framework)
    # loss.backward()
    # optimizer.step()

    return loss


# --- Demo: run one training step with a dummy model ---
betas, alphas, alpha_bars = linear_noise_schedule(T=1000)

# Dummy model that returns random noise (untrained baseline)
def dummy_model(x_t, t):
    return np.random.randn(*x_t.shape)

# Fake batch of 4 images, 1 channel, 8x8
x_0 = np.random.randn(4, 1, 8, 8)
loss = train_step(dummy_model, None, x_0, alpha_bars, T=1000)
print(f"Training step loss (random model): {loss:.4f}")
print("(Expected ~2.0 for random predictions of unit-variance noise)")
