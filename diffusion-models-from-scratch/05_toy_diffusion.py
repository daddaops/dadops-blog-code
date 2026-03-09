import numpy as np

# --- Needed from block 1 ---
def linear_noise_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from DDPM (Ho et al. 2020)."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars

# --- Needed from block 3 ---
def sinusoidal_embedding(t, dim=64):
    """Sinusoidal timestep embedding -- same math as positional encoding."""
    t = np.atleast_1d(np.array(t, dtype=np.float64))
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1)


# ---------- 2D Swiss Roll Dataset ----------
def make_swiss_roll(n=2000):
    """Generate 2D Swiss roll data points."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t) / (3 * np.pi)
    y = t * np.sin(t) / (3 * np.pi)
    return np.stack([x, y], axis=1)  # shape: [n, 2]

# ---------- Simple 2-Layer MLP as Denoiser ----------
class ToyDenoiser:
    """Minimal MLP denoiser for 2D data with timestep conditioning."""

    def __init__(self, hidden=128, t_dim=32):
        # Input: 2D point + t_dim timestep embedding = 2 + t_dim
        scale = 0.01
        self.W1 = np.random.randn(2 + t_dim, hidden) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * scale
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, 2) * scale
        self.b3 = np.zeros(2)

    def forward(self, x, t_emb):
        """x: [B, 2], t_emb: [B, t_dim] -> noise prediction [B, 2]"""
        h = np.concatenate([x, t_emb], axis=-1)
        h = np.maximum(0, h @ self.W1 + self.b1)  # ReLU
        h = np.maximum(0, h @ self.W2 + self.b2)  # ReLU
        return h @ self.W3 + self.b3

# ---------- Training Loop ----------
T = 200  # fewer steps for 2D toy data
betas, alphas, alpha_bars = linear_noise_schedule(T, 1e-4, 0.02)
model = ToyDenoiser(hidden=128, t_dim=32)
lr = 1e-3

losses = []
for step in range(5000):
    # 1. Sample clean data
    x_0 = make_swiss_roll(256)

    # 2. Sample random timesteps
    t = np.random.randint(0, T, size=256)

    # 3. Add noise
    noise = np.random.randn(256, 2)
    a_bar = alpha_bars[t].reshape(-1, 1)
    x_t = np.sqrt(a_bar) * x_0 + np.sqrt(1 - a_bar) * noise

    # 4. Timestep embedding
    t_emb = sinusoidal_embedding(t, dim=32)

    # 5. Predict noise and compute loss
    noise_pred = model.forward(x_t, t_emb)
    loss = np.mean((noise - noise_pred) ** 2)
    losses.append(loss)

    # 6. Gradient update -- in PyTorch/JAX you'd compute:
    #    loss.backward()
    #    optimizer.step()
    # (Omitted here since NumPy has no autograd)

    if step % 1000 == 0:
        print(f"Step {step:5d} | Loss: {loss:.4f}")

# Without autograd, loss stays ~1.0 (no weight updates happen):
# Step     0 | Loss: 1.1324
# Step  1000 | Loss: 0.9573
# Step  2000 | Loss: 0.9766
# Step  3000 | Loss: 0.9833
# Step  4000 | Loss: 0.9099
# With a real framework (PyTorch), loss would decrease as the model learns.
