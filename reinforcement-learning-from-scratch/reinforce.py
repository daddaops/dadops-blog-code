"""REINFORCE policy gradient on a simple environment."""
import numpy as np

class PolicyNetwork:
    """A simple 2-layer policy network for discrete actions."""
    def __init__(self, state_dim, hidden_dim, n_actions, lr=0.01):
        # Xavier initialization
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(n_actions)
        self.lr = lr

    def forward(self, state):
        """Forward pass: state -> action probabilities."""
        self.z1 = state @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)  # ReLU
        self.logits = self.h1 @ self.W2 + self.b2
        # Softmax for action probabilities
        exp_logits = np.exp(self.logits - np.max(self.logits))
        self.probs = exp_logits / exp_logits.sum()
        return self.probs

    def select_action(self, state):
        """Sample an action from the policy distribution."""
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action

    def update(self, states, actions, returns):
        """REINFORCE update: increase prob of actions with high returns."""
        for state, action, G in zip(states, actions, returns):
            probs = self.forward(state)
            # Gradient of log pi(a|s) for the chosen action
            # d/d(logits) log(softmax(logits)[a]) = (one_hot(a) - probs)
            grad_logits = -probs.copy()
            grad_logits[action] += 1.0  # one_hot - softmax

            # Scale by return: good returns -> increase this action's prob
            grad_logits *= G

            # Backprop through the network
            grad_W2 = self.h1.reshape(-1, 1) @ grad_logits.reshape(1, -1)
            grad_b2 = grad_logits
            grad_h1 = grad_logits @ self.W2.T
            grad_z1 = grad_h1 * (self.z1 > 0)  # ReLU gradient
            grad_W1 = state.reshape(-1, 1) @ grad_z1.reshape(1, -1)
            grad_b1 = grad_z1

            # Gradient ascent (we want to maximize return)
            self.W2 += self.lr * grad_W2
            self.b2 += self.lr * grad_b2
            self.W1 += self.lr * grad_W1
            self.b1 += self.lr * grad_b1

# Quick smoke test: train on a simple task
np.random.seed(42)
policy = PolicyNetwork(state_dim=4, hidden_dim=16, n_actions=2, lr=0.01)

# Simulate a few episodes
for ep in range(50):
    states, actions, rewards = [], [], []
    state = np.random.randn(4)
    for t in range(20):
        action = policy.select_action(state)
        reward = 1.0 if action == (1 if state[0] > 0 else 0) else -0.5
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = np.random.randn(4)

    # Compute discounted returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    policy.update(states, actions, returns)

# Test the trained policy
correct = 0
for _ in range(100):
    s = np.random.randn(4)
    a = np.argmax(policy.forward(s))
    if a == (1 if s[0] > 0 else 0):
        correct += 1
print(f"REINFORCE accuracy on simple task: {correct}%")
