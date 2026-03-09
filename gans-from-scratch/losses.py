import numpy as np

def discriminator_loss(d_real, d_fake):
    """Binary cross-entropy loss for the discriminator.
    d_real: D(x) for real samples, d_fake: D(G(z)) for generated samples.
    D wants d_real -> 1 and d_fake -> 0."""
    eps = 1e-8  # numerical stability
    loss_real = -np.mean(np.log(d_real + eps))       # E[-log D(x)]
    loss_fake = -np.mean(np.log(1 - d_fake + eps))   # E[-log(1 - D(G(z)))]
    # Gradients w.r.t. D's output probabilities
    grad_real = -1.0 / (d_real + eps)                 # d/dp [-log(p)]
    grad_fake = 1.0 / (1 - d_fake + eps)              # d/dp [-log(1-p)]
    return loss_real + loss_fake, grad_real, grad_fake

def generator_loss_nonsaturating(d_fake):
    """Non-saturating generator loss: maximize log D(G(z)).
    Provides strong gradients even when D easily spots fakes."""
    eps = 1e-8
    loss = -np.mean(np.log(d_fake + eps))             # E[-log D(G(z))]
    grad = -1.0 / (d_fake + eps)                      # steep near 0!
    return loss, grad
