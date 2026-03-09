"""Euler and RK4 ODE solvers applied to a 2D spiral.

Demonstrates that RK4 is far more accurate than Euler with
the same number of steps (15).
"""
import numpy as np

def spiral_ode(h, t):
    """dh/dt for a 2D spiral: rotation + mild contraction."""
    x, y = h
    return np.array([-0.1 * x - y, x - 0.1 * y])

def euler_solve(f, h0, t_span, n_steps):
    """Forward Euler: h_{k+1} = h_k + dt * f(h_k, t_k)."""
    ts = np.linspace(t_span[0], t_span[1], n_steps + 1)
    dt = ts[1] - ts[0]
    traj = [h0]
    for i in range(n_steps):
        traj.append(traj[-1] + dt * f(traj[-1], ts[i]))
    return ts, np.array(traj)

def rk4_solve(f, h0, t_span, n_steps):
    """Classic RK4: four evaluations per step, O(dt^4) accuracy."""
    ts = np.linspace(t_span[0], t_span[1], n_steps + 1)
    dt = ts[1] - ts[0]
    traj = [h0]
    for i in range(n_steps):
        t, h = ts[i], traj[-1]
        k1 = f(h, t)
        k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(h + dt * k3, t + dt)
        traj.append(h + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4))
    return ts, np.array(traj)


if __name__ == "__main__":
    h0 = np.array([2.0, 0.0])
    _, euler_traj = euler_solve(spiral_ode, h0, (0, 6), n_steps=15)
    _, rk4_traj   = rk4_solve(spiral_ode, h0, (0, 6), n_steps=15)

    print(f"Euler final:  [{euler_traj[-1][0]:.4f}, {euler_traj[-1][1]:.4f}]")
    print(f"RK4 final:    [{rk4_traj[-1][0]:.4f}, {rk4_traj[-1][1]:.4f}]")

    # High-resolution reference
    _, ref = rk4_solve(spiral_ode, h0, (0, 6), n_steps=10000)
    print(f"Reference:    [{ref[-1][0]:.4f}, {ref[-1][1]:.4f}]")

    euler_err = np.linalg.norm(euler_traj[-1] - ref[-1])
    rk4_err = np.linalg.norm(rk4_traj[-1] - ref[-1])
    print(f"\nEuler error: {euler_err:.4f}")
    print(f"RK4 error:   {rk4_err:.6f}")
    print(f"RK4 is {euler_err / rk4_err:.0f}x more accurate")
