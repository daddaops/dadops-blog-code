import numpy as np

# --- Generate Swiss Roll data ---
def make_swiss_roll(n=2000):
    t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    x = t * np.cos(t) / 10.0
    y = t * np.sin(t) / 10.0
    return np.stack([x, y], axis=1)

# --- Simple MLP velocity network ---
class VelocityNet:
    """2-layer MLP that predicts velocity given (x, t).
    Input: [x_dim + 1] (position + time)
    Output: [x_dim] (velocity)
    """
    def __init__(self, x_dim=2, hidden=128):
        scale = 0.01
        self.W1 = np.random.randn(x_dim + 1, hidden) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * scale
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, x_dim) * scale
        self.b3 = np.zeros(x_dim)

    def forward(self, x, t):
        # Concatenate position and time
        t_col = np.full((x.shape[0], 1), t) if np.isscalar(t) else t.reshape(-1, 1)
        inp = np.concatenate([x, t_col], axis=1)
        # Two hidden layers with SiLU activation
        h = inp @ self.W1 + self.b1
        h = h * (1 / (1 + np.exp(-h)))  # SiLU = x * sigmoid(x)
        h = h @ self.W2 + self.b2
        h = h * (1 / (1 + np.exp(-h)))  # SiLU
        return h @ self.W3 + self.b3

    __call__ = forward  # make instances callable: model(x, t)

# --- Generate samples with Euler ---
def generate(model, n=500, steps=20):
    x = np.random.randn(n, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        v = model(x, t)
        x = x + dt * v
    return x

if __name__ == "__main__":
    np.random.seed(42)

    # --- Training loop ---
    data = make_swiss_roll(5000)
    model = VelocityNet(x_dim=2, hidden=128)
    lr = 1e-3

    for step in range(5000):
        # Sample a batch
        idx = np.random.choice(len(data), 256)
        x_1 = data[idx]                                      # data
        x_0 = np.random.randn(256, 2)                        # noise
        t = np.random.uniform(0, 1, (256, 1))                # time

        # Flow matching: interpolate, compute target, predict
        x_t = (1 - t) * x_0 + t * x_1                       # path
        target = x_1 - x_0                                    # velocity
        pred = model(x_t, t)                          # prediction
        loss = np.mean((pred - target) ** 2)                  # MSE loss

        # (In practice, use PyTorch autograd for backprop here)
        # This shows the forward pass logic — the training objective is just MSE

        if step % 1000 == 0:
            print(f"Step {step}: loss = {loss:.4f}")

    # --- Generate samples ---
    samples_50 = generate(model, steps=50)
    samples_20 = generate(model, steps=20)
    samples_5  = generate(model, steps=5)
    print(f"Generated samples: 50-step {samples_50.shape}, 20-step {samples_20.shape}, 5-step {samples_5.shape}")
