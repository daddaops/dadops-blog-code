import numpy as np

def track_kalman_gain(n_steps, Q_scale, R_val):
    """Track how Kalman gain evolves over time."""
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * Q_scale
    R = np.array([[R_val]])
    P = np.array([[100.0, 0.0], [0.0, 100.0]])

    gains = []
    uncertainties = []
    for t in range(n_steps):
        P = F @ P @ F.T + Q                    # predict
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)         # gain
        P = (np.eye(2) - K @ H) @ P            # update
        gains.append(K[0, 0])                   # position gain
        uncertainties.append(np.sqrt(P[0, 0]))  # position std

    return gains, uncertainties

if __name__ == "__main__":
    # Compare: low measurement noise vs high measurement noise
    gains_lo, unc_lo = track_kalman_gain(50, Q_scale=0.09, R_val=4.0)
    gains_hi, unc_hi = track_kalman_gain(50, Q_scale=0.09, R_val=100.0)

    print("Low R (precise sensor): gain converges to {:.3f}".format(gains_lo[-1]))
    print("High R (noisy sensor):  gain converges to {:.3f}".format(gains_hi[-1]))
    print("Low R uncertainty:  {:.2f} m".format(unc_lo[-1]))
    print("High R uncertainty: {:.2f} m".format(unc_hi[-1]))
    # Low R (precise sensor): gain converges to 0.312
    # High R (noisy sensor):  gain converges to 0.068
    # Low R uncertainty:  1.12 m
    # High R uncertainty: 5.21 m
