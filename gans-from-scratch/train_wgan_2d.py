import numpy as np
from generator import Generator
from wgan import WassersteinCritic, wasserstein_loss

def sample_8_gaussians(n, radius=2.0, std=0.05):
    """Sample from 8 Gaussians arranged in a circle."""
    centers = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    centers = np.array(centers)

    # Pick random centers, add Gaussian noise
    indices = np.random.randint(0, 8, size=n)
    samples = centers[indices] + np.random.randn(n, 2) * std
    return samples, indices

def train_wgan_2d(epochs=3000, batch_size=256, n_critic=5):
    """Train WGAN on 8-Gaussians target. Returns training history."""
    G = Generator(noise_dim=2, hidden_dim=128, output_dim=2)
    C = WassersteinCritic(input_dim=2, hidden_dim=128)

    history = []
    for epoch in range(epochs):
        # --- Train Critic for n_critic steps ---
        for _ in range(n_critic):
            real, _ = sample_8_gaussians(batch_size)
            noise = np.random.randn(batch_size, 2)
            fake = G.forward(noise)
            c_real = C.forward(real)
            c_fake = C.forward(fake)
            w_dist, grad_r, grad_f = wasserstein_loss(c_real, c_fake)
            # Fake activations are current — process fake first
            C.backward(-grad_f, lr=0.0002)  # negate: we maximize
            C.forward(real)  # restore real activations
            C.backward(-grad_r, lr=0.0002)
            C.clip_weights(0.01)

        # --- Train Generator ---
        noise = np.random.randn(batch_size, 2)
        fake = G.forward(noise)
        c_fake = C.forward(fake)
        # G wants to maximize critic score → minimize -f(G(z))
        grad_g = -np.ones_like(c_fake)
        grad_input = C.backward(grad_g, lr=0)
        G.backward(grad_input, lr=0.0001)

        if epoch % 100 == 0:
            history.append((epoch, w_dist, fake.copy()))

    return G, history

if __name__ == "__main__":
    np.random.seed(42)
    print("Training WGAN on 8-Gaussians (3000 epochs)...")
    G, history = train_wgan_2d(epochs=3000)
    print(f"Training complete. {len(history)} checkpoints recorded.")
    print(f"Final W-distance: {history[-1][1]:.4f}")
    print(f"Final generated samples shape: {history[-1][2].shape}")
