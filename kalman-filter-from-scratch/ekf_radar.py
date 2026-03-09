import numpy as np

def ekf_radar(measurements, dt=1.0, q=0.5, r_range=5.0, r_bearing=0.05):
    """Extended Kalman Filter for radar tracking (range + bearing measurements)."""
    # State: [px, py, vx, vy]
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])

    Q = np.eye(4) * q * np.array([dt**4/4, dt**4/4, dt**2, dt**2])
    R = np.diag([r_range**2, r_bearing**2])

    x = np.array([measurements[0, 0] * np.cos(measurements[0, 1]),
                  measurements[0, 0] * np.sin(measurements[0, 1]), 0.0, 0.0])
    P = np.eye(4) * 100.0
    estimates = []

    for z in measurements:
        # --- Predict ---
        x = F @ x
        P = F @ P @ F.T + Q

        # --- Nonlinear observation function h(x) ---
        px, py = x[0], x[1]
        r_pred = np.sqrt(px**2 + py**2)
        theta_pred = np.arctan2(py, px)
        z_pred = np.array([r_pred, theta_pred])

        # --- Jacobian of h(x) ---
        r = max(r_pred, 1e-6)  # avoid division by zero
        J = np.array([[px/r,      py/r,     0, 0],
                      [-py/r**2,  px/r**2,  0, 0]])

        # --- Update (using Jacobian J instead of H) ---
        S = J @ P @ J.T + R
        K = P @ J.T @ np.linalg.inv(S)
        innovation = z - z_pred
        innovation[1] = (innovation[1] + np.pi) % (2*np.pi) - np.pi  # wrap angle
        x = x + K @ innovation
        P = (np.eye(4) - K @ J) @ P
        estimates.append(x.copy())

    return np.array(estimates)

if __name__ == "__main__":
    # Simulate radar tracking a circling target
    np.random.seed(42)
    n = 80
    angles = np.linspace(0, 2*np.pi, n)
    true_x = 100 + 40 * np.cos(angles)
    true_y = 80 + 40 * np.sin(angles)

    # Radar at origin measures range and bearing with noise
    true_range = np.sqrt(true_x**2 + true_y**2)
    true_bearing = np.arctan2(true_y, true_x)
    z_radar = np.column_stack([
        true_range + np.random.normal(0, 5.0, n),
        true_bearing + np.random.normal(0, 0.05, n)
    ])

    est = ekf_radar(z_radar, dt=1.0, q=0.5, r_range=5.0, r_bearing=0.05)

    raw_x = z_radar[:, 0] * np.cos(z_radar[:, 1])
    raw_y = z_radar[:, 0] * np.sin(z_radar[:, 1])
    raw_err = np.mean(np.sqrt((raw_x - true_x)**2 + (raw_y - true_y)**2))
    filt_err = np.mean(np.sqrt((est[:, 0] - true_x)**2 + (est[:, 1] - true_y)**2))

    print(f"Raw measurement error:  {raw_err:.1f} m")
    print(f"EKF filtered error:     {filt_err:.1f} m")
    print(f"Improvement:            {raw_err/filt_err:.1f}x")
    # Raw measurement error:  6.8 m
    # EKF filtered error:     2.3 m
    # Improvement:            3.0x
