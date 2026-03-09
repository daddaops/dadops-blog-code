import numpy as np
from swiss_roll_training import VelocityNet, generate

# --- Reflow: straighten the learned paths ---
def generate_reflow_pairs(model, n=5000, steps=50):
    """Create (noise, data) pairs by running the trained model.
    These pairs have deterministic coupling — less crossing than random pairs.
    """
    x_0 = np.random.randn(n, 2)
    z_1 = x_0.copy()
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        v = model(z_1, t)
        z_1 = z_1 + dt * v
    return x_0, z_1

if __name__ == "__main__":
    np.random.seed(42)

    # Create an initial model (in practice, this would be pre-trained)
    model = VelocityNet(x_dim=2, hidden=128)

    # Generate reflowed training pairs
    noise_paired, data_paired = generate_reflow_pairs(model, n=5000)

    # Retrain on the reflowed pairs (same training loop, different data source)
    model_v2 = VelocityNet(x_dim=2, hidden=128)

    for step in range(5000):
        idx = np.random.choice(len(noise_paired), 256)
        x_0 = noise_paired[idx]
        x_1 = data_paired[idx]
        t = np.random.uniform(0, 1, (256, 1))

        x_t = (1 - t) * x_0 + t * x_1
        target = x_1 - x_0
        pred = model_v2(x_t, t)
        loss = np.mean((pred - target) ** 2)

        # (backprop and optimizer step in practice)

    # After reflow: fewer steps needed for the same quality
    samples_reflow_5  = generate(model_v2, steps=5)
    samples_reflow_1  = generate(model_v2, steps=1)
    # Compare: 5-step reflow samples should match 50-step original
    print(f"Reflow 5-step samples: {samples_reflow_5.shape}")
    print(f"Reflow 1-step samples: {samples_reflow_1.shape}")
