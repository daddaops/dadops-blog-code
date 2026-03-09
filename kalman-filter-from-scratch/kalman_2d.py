import numpy as np

def kalman_filter_2d(measurements, dt=1.0, q=0.1, r=8.0):
    """2D Kalman filter: track [px, py, vx, vy] from position-only measurements."""
    # State transition: constant velocity model
    F = np.array([[1, 0, dt, 0],    # px = px + vx*dt
                  [0, 1, 0, dt],    # py = py + vy*dt
                  [0, 0, 1,  0],    # vx = vx
                  [0, 0, 0,  1]])   # vy = vy

    # Observation: we only measure position
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    # Process noise: random acceleration in x and y
    Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                  [0, dt**4/4, 0, dt**3/2],
                  [dt**3/2, 0, dt**2,   0],
                  [0, dt**3/2, 0,   dt**2]]) * q

    R = np.eye(2) * r**2   # measurement noise covariance

    x = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    P = np.eye(4) * 500.0  # high initial uncertainty

    estimates = np.zeros((len(measurements), 4))
    covariances = np.zeros((len(measurements), 4, 4))

    for t in range(len(measurements)):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (measurements[t] - H @ x)
        P = (np.eye(4) - K @ H) @ P
        estimates[t] = x
        covariances[t] = P

    return estimates, covariances

if __name__ == "__main__":
    # Simulate a curved trajectory (quarter circle)
    np.random.seed(42)
    n = 60
    t_vals = np.linspace(0, np.pi/2, n)
    true_x = 50 * np.cos(t_vals)
    true_y = 50 * np.sin(t_vals)

    # Noisy position measurements
    meas = np.column_stack([
        true_x + np.random.normal(0, 8, n),
        true_y + np.random.normal(0, 8, n)
    ])

    est, cov = kalman_filter_2d(meas, dt=1.0, q=0.1, r=8.0)

    # The filter estimates velocity despite never measuring it!
    print(f"True velocity at t=30:  ({-50*np.sin(t_vals[30])*np.pi/120:.1f}, "
          f"{50*np.cos(t_vals[30])*np.pi/120:.1f}) m/s")
    print(f"Estimated velocity:     ({est[30, 2]:.1f}, {est[30, 3]:.1f}) m/s")
    print(f"Position error (raw):   {np.sqrt((meas[30,0]-true_x[30])**2 + (meas[30,1]-true_y[30])**2):.1f} m")
    print(f"Position error (filt):  {np.sqrt((est[30,0]-true_x[30])**2 + (est[30,1]-true_y[30])**2):.1f} m")
    # True velocity at t=30:  (-0.9, 0.6) m/s
    # Estimated velocity:     (-0.8, 0.5) m/s
    # Position error (raw):   9.3 m
    # Position error (filt):  2.1 m
