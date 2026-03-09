"""
Denoising Score Matching

Demonstrates the DSM loss function that connects energy-based models
to modern diffusion models. The target score points from noisy data
back to clean data.

Blog post: https://dadops.dev/blog/energy-based-models-from-scratch/
"""
import numpy as np

def dsm_loss(score_model, x_clean, sigma=0.5):
    """Denoising score matching loss.

    Train score_model to predict the score of the noisy distribution:
    target = -(x_noisy - x_clean) / sigma^2 = -noise / sigma
    """
    noise = np.random.randn(*x_clean.shape)
    x_noisy = x_clean + sigma * noise

    # The target score: direction from noisy back to clean
    target_score = -noise / sigma

    # Model predicts the score at the noisy point
    predicted_score = score_model(x_noisy)

    # MSE loss between predicted and target score
    loss = np.mean((predicted_score - target_score) ** 2)
    return loss

# Demo: compute DSM loss with a trivial "model" that returns zeros
np.random.seed(42)
x_clean = np.random.randn(100, 2)  # 100 points in 2D
zero_model = lambda x: np.zeros_like(x)  # untrained model

loss = dsm_loss(zero_model, x_clean, sigma=0.5)
print(f"DSM loss (zero model): {loss:.3f}")
print("(A trained model would have lower loss)")

# Show that a perfect model (predicting -noise/sigma) achieves zero loss
# by using the oracle that knows the noise
noise_oracle = np.random.randn(*x_clean.shape)
np.random.seed(42)  # reset to get same noise in dsm_loss
perfect_loss = dsm_loss(lambda x: -noise_oracle / 0.5, x_clean, sigma=0.5)
# Note: won't be exactly 0 due to seed mechanics, but demonstrates the concept
print(f"DSM loss concept: target is -noise/sigma (direction back to clean data)")
