import numpy as np

def expert_policy(state):
    """Expert follows a sinusoidal path: target_y = sin(x)."""
    x, y = state
    target_y = np.sin(x * 2)
    dy = np.clip(target_y - y, -0.3, 0.3)
    return np.array([0.2, dy])  # constant forward speed, corrective steering

def collect_expert_demos(n_trajectories=20, horizon=30):
    """Collect expert demonstrations."""
    states, actions = [], []
    for _ in range(n_trajectories):
        s = np.array([0.0, np.random.randn() * 0.1])
        for t in range(horizon):
            a = expert_policy(s)
            states.append(s.copy())
            actions.append(a.copy())
            s = s + a + np.random.randn(2) * 0.02  # small noise
    return np.array(states), np.array(actions)

# Simple linear policy for behavioral cloning
class LinearBC:
    def __init__(self, state_dim, action_dim):
        self.W = np.zeros((action_dim, state_dim))
        self.b = np.zeros(action_dim)

    def predict(self, state):
        return self.W @ state + self.b

    def train(self, states, actions, lr=0.1, epochs=200):
        for _ in range(epochs):
            pred = states @ self.W.T + self.b
            error = pred - actions
            self.W -= lr * (error.T @ states) / len(states)
            self.b -= lr * error.mean(axis=0)

if __name__ == "__main__":
    # Collect data and train
    np.random.seed(42)
    states, actions = collect_expert_demos()
    bc = LinearBC(2, 2)
    bc.train(states, actions)

    # Evaluate: roll out the BC policy
    s = np.array([0.0, 0.0])
    bc_trajectory = [s.copy()]
    for t in range(30):
        a = bc.predict(s)
        s = s + a + np.random.randn(2) * 0.02
        bc_trajectory.append(s.copy())
    bc_traj = np.array(bc_trajectory)

    print(f"BC final position: ({bc_traj[-1, 0]:.2f}, {bc_traj[-1, 1]:.2f})")
    print(f"Expert final pos:  ({6.00:.2f}, {np.sin(6.0 * 2):.2f})")
