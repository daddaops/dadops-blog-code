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

# --- Conditional model and classifier-free guidance ---

class ConditionalDenoiseMLP:
    """MLP that accepts an optional class label for conditional generation."""
    def __init__(self, num_classes=3, t_dim=64, c_dim=16, hidden=256):
        scale = 0.01
        # Class embedding (plus one slot for "no class" / unconditional)
        self.class_emb = np.random.randn(num_classes + 1, c_dim) * scale
        input_dim = 2 + t_dim + c_dim
        self.W1 = np.random.randn(input_dim, hidden) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * scale
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, 2) * scale
        self.b3 = np.zeros(2)

    def forward(self, x_t, t_emb, class_id=None):
        """Predict noise. class_id=None means unconditional."""
        if class_id is None:
            c_emb = self.class_emb[-1]  # null class embedding
        else:
            c_emb = self.class_emb[class_id]
        inp = np.concatenate([x_t, t_emb, c_emb])
        h = inp @ self.W1 + self.b1
        h = np.maximum(h, 0.01 * h)
        h = h @ self.W2 + self.b2
        h = np.maximum(h, 0.01 * h)
        return h @ self.W3 + self.b3

def guided_sample(model, class_id, w=3.0, num_steps=50,
                   T=1000, alpha_bar=alpha_bar):
    """Generate with classifier-free guidance."""
    step_size = T // num_steps
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)

    x = np.random.randn(2)

    for i in range(len(timesteps) - 1):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        t_emb = sinusoidal_embedding(t_cur)

        # Two forward passes: conditional and unconditional
        eps_cond = model.forward(x, t_emb, class_id=class_id)
        eps_uncond = model.forward(x, t_emb, class_id=None)

        # Guided noise prediction — extrapolate away from unconditional
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

        # DDIM update with guided prediction
        x0_pred = (x - np.sqrt(1 - alpha_bar[t_cur]) * eps_guided)
        x0_pred /= np.sqrt(alpha_bar[t_cur])
        x = (np.sqrt(alpha_bar[t_next]) * x0_pred +
             np.sqrt(1 - alpha_bar[t_next]) * eps_guided)

    return x

# Demo with untrained conditional model
model = ConditionalDenoiseMLP(num_classes=3)
print("Classifier-free guidance sampling (untrained model demo):")
for class_id in range(3):
    for w in [1.0, 5.0]:
        np.random.seed(42)
        sample = guided_sample(model, class_id=class_id, w=w)
        print(f"  class={class_id}, w={w:.1f} -> {sample}")
# With a trained model:
# guided_sample(model, class_id=0, w=1.0)  -> loose spiral
# guided_sample(model, class_id=0, w=5.0)  -> tight, confident spiral
# guided_sample(model, class_id=1, w=5.0)  -> tight circles
