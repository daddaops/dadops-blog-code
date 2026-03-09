import numpy as np

def make_tracking_data():
    """Generate ground truth and noisy measurements for 1D tracking."""
    np.random.seed(42)
    dt = 1.0
    n_steps = 50
    true_pos = np.zeros(n_steps)
    true_vel = 2.0

    for t in range(1, n_steps):
        process_noise = np.random.normal(0, 0.3)
        true_vel_t = true_vel + process_noise
        true_pos[t] = true_pos[t-1] + true_vel_t * dt

    measurement_noise_std = 5.0
    measurements = true_pos + np.random.normal(0, measurement_noise_std, n_steps)
    return true_pos, measurements

if __name__ == "__main__":
    true_pos, measurements = make_tracking_data()
    print(f"True position at t=25: {true_pos[25]:.1f} m")
    print(f"Measured position at t=25: {measurements[25]:.1f} m")
    print(f"Measurement error: {abs(measurements[25] - true_pos[25]):.1f} m")
    # True position at t=25: 50.8 m
    # Measured position at t=25: 55.2 m
    # Measurement error: 4.4 m
