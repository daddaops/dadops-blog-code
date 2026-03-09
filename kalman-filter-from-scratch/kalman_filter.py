import numpy as np
from data_setup import make_tracking_data

def kalman_filter(measurements, F, H, Q, R, x0, P0):
    """Run Kalman filter on a sequence of measurements."""
    n = len(measurements)
    dim = F.shape[0]
    estimates = np.zeros((n, dim))
    covariances = np.zeros((n, dim, dim))

    x = x0.copy()
    P = P0.copy()

    for t in range(n):
        # --- Predict ---
        x = F @ x                    # project state forward
        P = F @ P @ F.T + Q          # project covariance forward

        # --- Update ---
        S = H @ P @ H.T + R          # innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
        y = measurements[t] - H @ x  # innovation (surprise)
        x = x + K @ y                # corrected state
        P = (np.eye(dim) - K @ H) @ P  # corrected covariance

        estimates[t] = x
        covariances[t] = P

    return estimates, covariances

if __name__ == "__main__":
    true_pos, measurements = make_tracking_data()

    # Setup: state = [position, velocity], measure position only
    dt = 1.0
    F = np.array([[1, dt],   # position = old_position + velocity * dt
                  [0,  1]])   # velocity = old_velocity (constant velocity model)
    H = np.array([[1, 0]])    # we only measure position

    Q = np.array([[0.25, 0.5],  # process noise (random acceleration)
                  [0.5,  1.0]]) * 0.09
    R = np.array([[25.0]])     # measurement noise variance (std=5m)^2

    x0 = np.array([0.0, 0.0])               # initial guess: origin, zero velocity
    P0 = np.array([[100.0, 0.0],            # high initial uncertainty
                   [0.0,  100.0]])

    estimates, covariances = kalman_filter(measurements, F, H, Q, R, x0, P0)

    print(f"Filtered position at t=25: {estimates[25, 0]:.1f} m (true: {true_pos[25]:.1f} m)")
    print(f"Position uncertainty (1-sigma): {np.sqrt(covariances[25, 0, 0]):.2f} m")
    # Filtered position at t=25: 51.2 m (true: 50.8 m)
    # Position uncertainty (1-sigma): 2.14 m
