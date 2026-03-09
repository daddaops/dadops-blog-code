"""BYOL (Bootstrap Your Own Latent) — learns without negative pairs."""
import numpy as np

class BYOL:
    """Bootstrap Your Own Latent -- learns without negative pairs."""
    def __init__(self, in_dim=8, hid=16, proj=8):
        # Online network: encoder -> projector -> predictor
        self.W_enc = np.random.randn(in_dim, hid) * 0.3
        self.W_proj = np.random.randn(hid, proj) * 0.3
        self.W_pred = np.random.randn(proj, proj) * 0.3  # only online has this!

        # Target network: encoder -> projector (NO predictor)
        self.W_enc_t = self.W_enc.copy()
        self.W_proj_t = self.W_proj.copy()

    def online(self, x):
        """Full online path: encode -> project -> predict."""
        h = np.tanh(x @ self.W_enc)
        z = np.tanh(h @ self.W_proj)
        return np.tanh(z @ self.W_pred)

    def target(self, x):
        """Target path: encode -> project (no predictor, no gradients)."""
        h_t = np.tanh(x @ self.W_enc_t)
        return np.tanh(h_t @ self.W_proj_t)

    def compute_loss(self, view1, view2):
        """Cosine distance between online(view1) and target(view2)."""
        p = self.online(view1)
        z = self.target(view2)  # stop-gradient: target not updated by loss
        p_n = p / (np.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        z_n = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        return 2 - 2 * np.mean(np.sum(p_n * z_n, axis=-1))

    def ema_update(self, tau=0.996):
        """Target slowly tracks online weights."""
        self.W_enc_t = tau * self.W_enc_t + (1 - tau) * self.W_enc
        self.W_proj_t = tau * self.W_proj_t + (1 - tau) * self.W_proj

# Demonstration
np.random.seed(42)
data = np.random.randn(50, 8)
model = BYOL()

# Compute BYOL loss on augmented views
view1 = data[:16] + np.random.randn(16, 8) * 0.1
view2 = data[:16] + np.random.randn(16, 8) * 0.1
loss = model.compute_loss(view1, view2)
model.ema_update(tau=0.996)

# Check representation diversity (healthy model = varied outputs)
reps = model.online(data)
print(f"BYOL loss: {loss:.4f}")
print(f"Output diversity (std): {np.std(reps):.4f}")
print(f"Output range: [{np.min(reps):.3f}, {np.max(reps):.3f}]")

# What does COLLAPSE look like?
collapsed = np.full_like(reps, 0.5)  # every input -> same vector
print(f"\nCollapsed diversity: {np.std(collapsed):.4f}  (all identical!)")
print("\nThree things prevent collapse:")
print("  1. Predictor MLP adds asymmetry between networks")
print("  2. Stop-gradient keeps target from chasing online")
print("  3. EMA updates keep target slowly drifting, staying informative")
print("Remove ANY one and representations collapse to a constant.")
