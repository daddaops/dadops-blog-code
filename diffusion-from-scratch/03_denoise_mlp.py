import numpy as np

def sinusoidal_embedding(t, dim=64):
    """Encode timestep t as a vector using sin/cos — same idea as
    positional encoding in transformers."""
    half = dim // 2
    freqs = np.exp(-np.log(10000) * np.arange(half) / half)
    args = t * freqs
    return np.concatenate([np.sin(args), np.cos(args)])

class DenoiseMLP:
    """Simple MLP that predicts noise given (x_t, t).
    Architecture: [2 + 64] -> 256 -> 256 -> 2
    """
    def __init__(self, t_dim=64, hidden=256):
        scale = 0.01
        self.W1 = np.random.randn(2 + t_dim, hidden) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * scale
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, 2) * scale
        self.b3 = np.zeros(2)

    def forward(self, x_t, t_emb):
        """Predict the noise epsilon given noisy x_t and time embedding."""
        inp = np.concatenate([x_t, t_emb])   # [2 + 64] = [66]
        h = inp @ self.W1 + self.b1           # [256]
        h = np.maximum(h, 0.01 * h)           # LeakyReLU
        h = h @ self.W2 + self.b2             # [256]
        h = np.maximum(h, 0.01 * h)           # LeakyReLU
        return h @ self.W3 + self.b3          # [2] — predicted noise

model = DenoiseMLP()
# Test: predict noise for a noisy point at t=500
t_emb = sinusoidal_embedding(500)
eps_pred = model.forward(np.array([0.5, -0.3]), t_emb)
print(f"Predicted noise: {eps_pred}")
# Predicted noise: [ 0.0012 -0.0008]  (random — untrained model)
