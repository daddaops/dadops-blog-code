import numpy as np

def euler_sample(model, x_0, num_steps=20):
    """Euler method: simplest ODE solver.
    1 network evaluation per step. Error: O(h) total.
    For straight paths, this is nearly exact even with few steps.
    """
    x = x_0.copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        v = model(x, t)
        x = x + dt * v
    return x

def midpoint_sample(model, x_0, num_steps=20):
    """Midpoint method: 2nd-order ODE solver.
    2 network evaluations per step. Error: O(h^2) total.
    Same quality as Euler in half the steps.
    """
    x = x_0.copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        # Evaluate at current point
        v1 = model(x, t)
        # Take a half-step to the midpoint
        x_mid = x + 0.5 * dt * v1
        # Evaluate at midpoint
        v2 = model(x_mid, t + 0.5 * dt)
        # Full step using midpoint velocity
        x = x + dt * v2
    return x

def rk4_sample(model, x_0, num_steps=20):
    """Runge-Kutta 4th order: classic high-accuracy solver.
    4 network evaluations per step. Error: O(h^4) total.
    Diminishing returns for straight paths — overkill if flow is well-trained.
    """
    x = x_0.copy()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        k1 = model(x, t)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

if __name__ == "__main__":
    # Quick demonstration with a constant velocity model
    class ConstantVelocity:
        def __call__(self, x, t):
            return np.ones_like(x)  # constant unit velocity

    np.random.seed(42)
    x_0 = np.zeros((3, 2))
    print("Euler:", euler_sample(ConstantVelocity(), x_0, num_steps=10)[-1])
    print("Midpoint:", midpoint_sample(ConstantVelocity(), x_0, num_steps=10)[-1])
    print("RK4:", rk4_sample(ConstantVelocity(), x_0, num_steps=10)[-1])
