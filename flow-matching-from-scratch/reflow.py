import numpy as np
from ode_solvers import euler_sample

def reflow(model_old, data_samples, num_ode_steps=50):
    """Reflow: generate straighter (noise, data) couplings.

    Instead of pairing random noise with random data (lots of crossing),
    we pair each noise sample with the data point that the current model
    *actually transports it to*. This deterministic coupling has fewer
    crossings, so the retrained model learns straighter paths.
    """
    n = data_samples.shape[0]

    # Sample fresh noise
    x_0 = np.random.randn(n, data_samples.shape[1])

    # Transport noise to data using the current model
    z_1 = euler_sample(model_old, x_0, num_steps=num_ode_steps)

    # The new training pairs: (x_0, z_1) instead of (x_0, x_1_random)
    # These pairs have less crossing => straighter retraining targets
    return x_0, z_1

def measure_straightness(model, x_0, x_1, num_eval_points=20):
    """Measure how straight the learned paths are.

    A perfectly straight path has x_t = (1-t)*x_0 + t*x_1 and velocity
    dx/dt = x_1 - x_0 everywhere. Straightness measures deviation from this.
    S = 0 means perfectly straight. Larger S means more curvature.
    """
    dt = 1.0 / num_eval_points
    total_deviation = 0.0
    x = x_0.copy()
    ideal_velocity = x_1 - x_0  # constant velocity for a straight line

    for i in range(num_eval_points):
        t = i * dt
        predicted_velocity = model(x, t)
        deviation = np.mean((predicted_velocity - ideal_velocity) ** 2)
        total_deviation += deviation
        x = x + dt * predicted_velocity

    return total_deviation / num_eval_points

if __name__ == "__main__":
    # Quick demonstration with a constant velocity model
    class ConstantVelocity:
        def __call__(self, x, t):
            return np.ones_like(x)

    np.random.seed(42)
    x_0 = np.zeros((10, 2))
    x_1 = np.ones((10, 2))
    straightness = measure_straightness(ConstantVelocity(), x_0, x_1)
    print(f"Straightness of constant velocity model: {straightness:.6f}")

    data = np.random.randn(100, 2)
    noise, transported = reflow(ConstantVelocity(), data)
    print(f"Reflow pairs generated: {noise.shape[0]}")
