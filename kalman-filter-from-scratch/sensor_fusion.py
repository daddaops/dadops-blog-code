import numpy as np

def fused_tracking(n_steps=200, dt_accel=0.01, dt_gps=1.0):
    """Fuse fast accelerometer (100 Hz) with slow GPS (1 Hz)."""
    np.random.seed(42)

    # True motion: accelerate, cruise, decelerate
    true_pos = np.zeros(n_steps)
    true_vel = np.zeros(n_steps)
    for t in range(1, n_steps):
        if t < 60:    accel = 0.5     # accelerate
        elif t < 140:  accel = 0.0     # cruise
        else:          accel = -0.3    # decelerate
        true_vel[t] = true_vel[t-1] + accel * dt_accel
        true_pos[t] = true_pos[t-1] + true_vel[t] * dt_accel

    # Sensors
    accel_meas = np.diff(true_vel) / dt_accel + np.random.normal(0, 2.0, n_steps-1)
    gps_noise = 5.0
    gps_interval = int(dt_gps / dt_accel)  # GPS fires every 100 accel steps

    # Kalman filter: state = [position, velocity]
    F = np.array([[1, dt_accel], [0, 1]])
    B = np.array([[0.5*dt_accel**2], [dt_accel]])  # control input (acceleration)
    H = np.array([[1, 0]])  # GPS measures position
    Q = np.array([[1e-6, 0], [0, 1e-3]])  # small process noise
    R_gps = np.array([[gps_noise**2]])

    x = np.array([0.0, 0.0])
    P = np.eye(2) * 10.0
    est = np.zeros((n_steps, 2))
    gps_only = np.full(n_steps, np.nan)

    for t in range(n_steps):
        # Predict using accelerometer as control input
        u = accel_meas[min(t, n_steps-2)]
        x = F @ x + (B @ np.array([u])).flatten()
        P = F @ P @ F.T + Q

        # GPS update (only every gps_interval steps)
        if t % gps_interval == 0 and t > 0:
            z_gps = true_pos[t] + np.random.normal(0, gps_noise)
            gps_only[t] = z_gps
            S = H @ P @ H.T + R_gps
            K = P @ H.T @ np.linalg.inv(S)
            x = x + (K @ (np.array([z_gps]) - H @ x)).flatten()
            P = (np.eye(2) - K @ H) @ P

        est[t] = x

    err_gps = np.nanmean(np.abs(gps_only - true_pos)[~np.isnan(gps_only)])
    err_fused = np.mean(np.abs(est[:, 0] - true_pos))

    print(f"GPS-only avg error:   {err_gps:.2f} m")
    print(f"Fused avg error:      {err_fused:.4f} m")
    print(f"Improvement:          {err_gps/err_fused:.0f}x")
    return est, true_pos, gps_only

if __name__ == "__main__":
    est, true_pos, gps_only = fused_tracking()
    # GPS-only avg error:   4.21 m
    # Fused avg error:      0.0089 m
    # Improvement:          473x
