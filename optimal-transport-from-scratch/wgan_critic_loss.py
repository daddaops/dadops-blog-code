"""WGAN critic loss with gradient penalty.

Demonstrates Wasserstein GAN's critic objective:
maximize separation between real and fake scores
while enforcing 1-Lipschitz constraint via gradient penalty.
"""
import numpy as np

def wgan_critic_loss(critic_fn, real_data, fake_data, lambda_gp=10.0):
    """WGAN-GP critic loss with gradient penalty.
    critic_fn(x): maps data points to scalar scores.
    Critic wants: high scores for real, low for fake."""
    # Wasserstein estimate: E[critic(real)] - E[critic(fake)]
    real_scores = np.array([critic_fn(x) for x in real_data])
    fake_scores = np.array([critic_fn(x) for x in fake_data])
    wasserstein_est = real_scores.mean() - fake_scores.mean()

    # Gradient penalty: enforce ||grad(critic)|| ≈ 1
    # Interpolate between real and fake samples
    batch_size = min(len(real_data), len(fake_data))
    alpha = np.random.rand(batch_size, 1)
    interpolated = alpha * real_data[:batch_size] + \
                   (1 - alpha) * fake_data[:batch_size]

    # Approximate gradient via finite differences
    eps = 1e-4
    grad_norms = []
    for x in interpolated:
        grads = []
        for d in range(x.shape[0]):
            x_plus = x.copy(); x_plus[d] += eps
            x_minus = x.copy(); x_minus[d] -= eps
            grads.append((critic_fn(x_plus) - critic_fn(x_minus)) / (2 * eps))
        grad_norms.append(np.sqrt(sum(g ** 2 for g in grads)))
    grad_penalty = np.mean([(gn - 1) ** 2 for gn in grad_norms])

    # Critic loss: minimize -wasserstein + penalty
    loss = -wasserstein_est + lambda_gp * grad_penalty
    return loss, wasserstein_est, grad_penalty

# Demo with a simple linear critic
def simple_critic(x):
    return 0.8 * x[0] + 0.6 * x[1]

np.random.seed(42)
real = np.random.normal([3, 3], 0.5, (32, 2))
fake = np.random.normal([0, 0], 0.5, (32, 2))

loss, w_est, gp = wgan_critic_loss(simple_critic, real, fake)
print(f"Wasserstein estimate: {w_est:.3f}")
print(f"Gradient penalty:     {gp:.3f}")
print(f"Total critic loss:    {loss:.3f}")
