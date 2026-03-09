import numpy as np

def flow_matching_training_step(model, data_batch):
    """One training step of Conditional Flow Matching.

    Compare with DDPM, which requires:
      - A noise schedule (beta_1, ..., beta_T)
      - Cumulative products (alpha_bar_t)
      - The reparameterization x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Flow matching needs none of that.
    """
    batch_size = data_batch.shape[0]

    # Sample noise and random timestep
    x_0 = np.random.randn(*data_batch.shape)       # noise
    x_1 = data_batch                                 # data
    t = np.random.uniform(0, 1, (batch_size, 1))    # time ~ U[0, 1]

    # Straight-line interpolation (the "path")
    x_t = (1 - t) * x_0 + t * x_1

    # The target velocity: constant along the straight line
    velocity_target = x_1 - x_0

    # Network predicts the velocity at (x_t, t)
    velocity_pred = model(x_t, t)

    # MSE loss — that's it
    loss = np.mean((velocity_pred - velocity_target) ** 2)

    return loss

if __name__ == "__main__":
    # Quick demonstration with a dummy model
    class DummyModel:
        def __call__(self, x, t):
            return np.zeros_like(x)

    np.random.seed(42)
    data = np.random.randn(64, 2)
    loss = flow_matching_training_step(DummyModel(), data)
    print(f"CFM training step loss (dummy model): {loss:.4f}")
