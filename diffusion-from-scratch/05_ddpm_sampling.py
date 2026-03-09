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

# --- DDPM Sampling ---

def ddpm_sample(model, T=1000, alpha_bar=alpha_bar, betas=betas):
    """Generate a new 2D point using DDPM reverse process."""
    alphas = 1.0 - betas
    # Start from pure noise
    x = np.random.randn(2)

    for t in reversed(range(T)):   # t = 999, 998, ..., 1, 0
        t_emb = sinusoidal_embedding(t)
        eps_pred = model.forward(x, t_emb)

        # Compute the mean of p(x_{t-1} | x_t)
        coef1 = 1.0 / np.sqrt(alphas[t])
        coef2 = betas[t] / np.sqrt(1.0 - alpha_bar[t])
        mean = coef1 * (x - coef2 * eps_pred)

        if t > 0:
            # Add noise (stochastic step) — variance from posterior
            sigma = np.sqrt(betas[t])
            x = mean + sigma * np.random.randn(2)
        else:
            x = mean  # final step is deterministic

    return x  # a brand new data point!

# Generate 300 new points (note: model is untrained so output will be random)
print("Generating samples with DDPM (untrained model demo)...")
samples = np.array([ddpm_sample(model) for _ in range(5)])
for i, s in enumerate(samples):
    print(f"  Sample {i}: {s}")
# With a trained model, the points would form a spiral.
