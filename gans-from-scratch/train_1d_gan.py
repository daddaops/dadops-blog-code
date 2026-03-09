import numpy as np
from generator import Generator
from discriminator import Discriminator
from losses import discriminator_loss, generator_loss_nonsaturating

if __name__ == "__main__":
    np.random.seed(42)

    # Training a 1D GAN: Generator learns to match N(3, 0.5)
    G = Generator(noise_dim=1, hidden_dim=64, output_dim=1)
    D = Discriminator(input_dim=1, hidden_dim=64)

    target_mean, target_std = 3.0, 0.5
    batch_size = 256
    d_lr, g_lr = 0.002, 0.001  # D gets a higher learning rate

    for epoch in range(2001):
        # --- Train Discriminator ---
        real_data = np.random.randn(batch_size, 1) * target_std + target_mean
        noise = np.random.randn(batch_size, 1)
        fake_data = G.forward(noise)

        d_real = D.forward(real_data)
        d_loss_r, grad_r, _ = discriminator_loss(d_real, np.zeros_like(d_real))
        D.backward(grad_r, lr=d_lr)

        d_fake = D.forward(fake_data)
        d_loss_f, _, grad_f = discriminator_loss(np.ones_like(d_fake), d_fake)
        D.backward(grad_f, lr=d_lr)

        # --- Train Generator (non-saturating) ---
        noise = np.random.randn(batch_size, 1)
        fake_data = G.forward(noise)
        d_fake = D.forward(fake_data)
        g_loss, g_grad = generator_loss_nonsaturating(d_fake)

        # Backprop: gradient flows D -> fake_data -> G
        grad_fake_data = D.backward(g_grad, lr=0)     # don't update D here
        G.backward(grad_fake_data, lr=g_lr)

        if epoch % 500 == 0:
            samples = G.forward(np.random.randn(1000, 1))
            print(f"Epoch {epoch}: G mean={samples.mean():.3f}, "
                  f"G std={samples.std():.3f}, D loss={d_loss_r+d_loss_f:.3f}")
