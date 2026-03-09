"""OpenAI Evolution Strategies for CartPole.

Evolves a single parameter vector by estimating a Gaussian-smoothed
fitness gradient via population-level perturbation and evaluation.
"""
import numpy as np


# --- Minimal CartPole environment (no gym dependency) ---
class CartPole:
    """CartPole-v1 physics: Euler integration, ±12°/±2.4m termination."""
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.state = None

    def reset(self, rng=None):
        r = rng if rng else np.random
        self.state = r.uniform(-0.05, 0.05, size=4)
        return self.state.copy(), {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costh, sinth = np.cos(theta), np.sin(theta)
        total_mass = self.masscart + self.masspole
        temp = (force + self.masspole * self.length * theta_dot**2 * sinth) / total_mass
        theta_acc = (self.gravity * sinth - costh * temp) / (
            self.length * (4.0/3.0 - self.masspole * costh**2 / total_mass))
        x_acc = temp - self.masspole * self.length * theta_acc * costh / total_mass
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = abs(x) > 2.4 or abs(theta) > 12 * np.pi / 180
        return self.state.copy(), 1.0, done, False, {}


def mlp_forward(weights, x):
    """2-layer MLP: 4 inputs -> 8 hidden (tanh) -> 1 output."""
    W1 = weights[:32].reshape(8, 4)
    b1 = weights[32:40]
    W2 = weights[40:48].reshape(1, 8)
    b2 = weights[48:49]
    h = np.tanh(W1 @ x + b1)
    return (W2 @ h + b2)[0]

def evaluate(weights, env, episodes=3):
    """Average reward across episodes."""
    total = 0
    for ep in range(episodes):
        obs, _ = env.reset(rng=np.random.RandomState(ep))
        for t in range(500):
            action = 1 if mlp_forward(weights, obs) > 0 else 0
            obs, reward, done, _, _ = env.step(action)
            total += reward
            if done:
                break
    return total / episodes

def es_cartpole(n_params=49, pop_size=50, generations=40,
                sigma=0.1, alpha=0.03):
    """OpenAI-style Evolution Strategies for CartPole."""
    env = CartPole()
    theta = np.zeros(n_params)

    np.random.seed(42)
    for gen in range(generations):
        # Sample perturbations
        epsilons = [np.random.randn(n_params) for _ in range(pop_size)]

        # Evaluate perturbed candidates
        rewards = np.array([
            evaluate(theta + sigma * eps, env) for eps in epsilons
        ])

        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Estimated gradient: weighted sum of perturbations
        grad = sum(r * eps for r, eps in zip(rewards, epsilons))
        theta += (alpha / (pop_size * sigma)) * grad

        score = evaluate(theta, env)
        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen:3d} | Score: {score:.0f}")

    return theta


if __name__ == "__main__":
    es_cartpole()
