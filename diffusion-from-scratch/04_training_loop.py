import numpy as np

# --- Dependencies from previous blocks ---

def linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from Ho et al. (2020)"""
    return np.linspace(beta_start, beta_end, T)

betas = linear_schedule(T=1000)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)

def make_spiral(n=300):
    theta = np.linspace(0, 4 * np.pi, n)
    r = theta / (4 * np.pi) * 2
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x += np.random.randn(n, 2) * 0.05
    return x.astype(np.float32)

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

# --- Training loop ---

data = make_spiral(300)
model = DenoiseMLP()

def train_diffusion(model, data, T=1000, alpha_bar=alpha_bar,
                     lr=1e-4, steps=10000, batch_size=64):
    """Train a diffusion model with the simplified MSE objective."""
    # Adam optimizer state
    params = [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]
    m = [np.zeros_like(p) for p in params]  # first moment
    v = [np.zeros_like(p) for p in params]  # second moment

    losses = []
    for step in range(steps):
        # 1. Sample a batch of clean data
        idx = np.random.randint(0, len(data), batch_size)
        x0_batch = data[idx]

        # 2. Sample random timesteps
        t_batch = np.random.randint(0, T, batch_size)

        # 3. Sample noise
        eps_batch = np.random.randn(batch_size, 2)

        total_loss = 0.0
        # Accumulate gradients over the batch
        grads = [np.zeros_like(p) for p in params]

        for i in range(batch_size):
            t = t_batch[i]
            x0 = x0_batch[i]
            eps = eps_batch[i]

            # 4. Create noisy sample
            sqrt_ab = np.sqrt(alpha_bar[t])
            sqrt_1m = np.sqrt(1.0 - alpha_bar[t])
            xt = sqrt_ab * x0 + sqrt_1m * eps

            # 5. Forward pass — predict noise
            t_emb = sinusoidal_embedding(t)
            eps_pred = model.forward(xt, t_emb)

            # 6. MSE loss
            loss = np.sum((eps - eps_pred) ** 2)
            total_loss += loss

            # Backprop (manual — computing gradients through the MLP)
            # ... (gradient computation omitted for brevity —
            #      in practice, use autograd like PyTorch)

        avg_loss = total_loss / batch_size
        losses.append(avg_loss)

        # Adam update on each parameter
        # ... (standard Adam update from our optimizers post)

        if step % 1000 == 0:
            print(f"Step {step:5d}  Loss: {avg_loss:.4f}")

    return losses

losses = train_diffusion(model, data)
# With autograd-based gradients, loss would decrease:
# Step     0  Loss: ~2.0  (random predictions)
# Step  5000  Loss: ~0.3  (learning spiral structure)
# Step  9000  Loss: ~0.2  (converged)
# Note: gradient computation omitted — loss stays ~2.0 without updates
