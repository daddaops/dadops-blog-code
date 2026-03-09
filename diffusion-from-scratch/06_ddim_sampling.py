import numpy as np

# --- Dependencies from previous blocks ---

def linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from Ho et al. (2020)"""
    return np.linspace(beta_start, beta_end, T)

betas = linear_schedule(T=1000)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)

def sinusoidal_embedding(t, dim=64):
    half = dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(half) / half)
    args = t * freqs
    return np.concatenate([np.sin(args), np.cos(args)])

class DenoiseMLP:
    def __init__(self, t_dim=64, hidden=256):
        scale = 0.01
        self.W1 = np.random.randn(2 + t_dim, hidden) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * scale
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, 2) * scale
        self.b3 = np.zeros(2)

    def forward(self, x_t, t_emb):
        inp = np.concatenate([x_t, t_emb])
        h = inp @ self.W1 + self.b1
        h = np.maximum(h, 0.01 * h)
        h = h @ self.W2 + self.b2
        h = np.maximum(h, 0.01 * h)
        return h @ self.W3 + self.b3

model = DenoiseMLP()

# --- DDIM Sampling ---

def ddim_sample(model, T=1000, alpha_bar=alpha_bar, num_steps=50):
    """Generate a new 2D point using DDIM (deterministic, fewer steps)."""
    # Create a sub-sequence of timesteps
    step_size = T // num_steps
    timesteps = list(range(T - 1, -1, -step_size))  # e.g. [999, 979, ...]
    if timesteps[-1] != 0:
        timesteps.append(0)

    x = np.random.randn(2)

    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]

        t_emb = sinusoidal_embedding(t_cur)
        eps_pred = model.forward(x, t_emb)

        # Predict x_0
        x0_pred = (x - np.sqrt(1 - alpha_bar[t_cur]) * eps_pred)
        x0_pred /= np.sqrt(alpha_bar[t_cur])

        # DDIM update — deterministic (no noise added)
        x = (np.sqrt(alpha_bar[t_next]) * x0_pred +
             np.sqrt(1 - alpha_bar[t_next]) * eps_pred)

    return x

# Compare step counts (untrained model demo)
print("DDIM sampling with different step counts (untrained model demo):")
for num_steps in [50, 10]:
    np.random.seed(0)
    sample = ddim_sample(model, num_steps=num_steps)
    print(f"  {num_steps:3d} steps -> {sample}")
# DDPM: 1000 steps -> high quality, slow
# DDIM:   50 steps -> nearly the same quality, 20x faster
# DDIM:   10 steps -> slightly degraded, 100x faster
