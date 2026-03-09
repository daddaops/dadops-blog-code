import numpy as np

class WassersteinCritic:
    """WGAN critic: outputs unbounded score, no sigmoid."""
    def __init__(self, input_dim=2, hidden_dim=64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)
        self.score = self.h @ self.W2 + self.b2     # no sigmoid!
        return self.score

    def backward(self, grad_score, lr=0.0001):
        grad_W2 = self.h.T @ grad_score
        grad_b2 = grad_score.sum(axis=0)
        grad_h = grad_score @ self.W2.T
        grad_h = grad_h * (self.h > 0)
        grad_W1 = self.x.T @ grad_h
        grad_b1 = grad_h.sum(axis=0)
        self.W2 -= lr * grad_W2 / len(grad_score)
        self.b2 -= lr * grad_b2 / len(grad_score)
        self.W1 -= lr * grad_W1 / len(grad_score)
        self.b1 -= lr * grad_b1 / len(grad_score)
        return grad_h @ self.W1.T   # gradient to input

    def clip_weights(self, clip_value=0.01):
        """Enforce Lipschitz constraint by clamping weights."""
        self.W1 = np.clip(self.W1, -clip_value, clip_value)
        self.W2 = np.clip(self.W2, -clip_value, clip_value)

def wasserstein_loss(critic_real, critic_fake):
    """WGAN loss: maximize E[f(real)] - E[f(fake)] for critic.
    No logs, no sigmoid — just the raw mean difference."""
    w_distance = np.mean(critic_real) - np.mean(critic_fake)
    # Critic gradient: ascend on real scores, descend on fake scores
    grad_real = np.ones_like(critic_real)                        # +1
    grad_fake = -np.ones_like(critic_fake)                       # -1
    return w_distance, grad_real, grad_fake
