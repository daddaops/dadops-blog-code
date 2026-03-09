"""Actor-Critic with shared layer and separate heads."""
import numpy as np

class ActorCritic:
    """Actor (policy) + Critic (value function) for variance reduction."""
    def __init__(self, state_dim, hidden_dim, n_actions, lr=0.005):
        # Shared first layer, separate heads
        s = np.sqrt(2.0 / state_dim)
        h = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(state_dim, hidden_dim) * s
        self.b1 = np.zeros(hidden_dim)
        # Actor head: outputs action probabilities
        self.W_actor = np.random.randn(hidden_dim, n_actions) * h
        self.b_actor = np.zeros(n_actions)
        # Critic head: outputs state value (scalar)
        self.W_critic = np.random.randn(hidden_dim, 1) * h
        self.b_critic = np.zeros(1)
        self.lr = lr

    def forward(self, state):
        self.z1 = state @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        # Actor: action probabilities
        logits = self.h1 @ self.W_actor + self.b_actor
        exp_l = np.exp(logits - np.max(logits))
        self.probs = exp_l / exp_l.sum()
        # Critic: state value
        self.value = (self.h1 @ self.W_critic + self.b_critic)[0]
        return self.probs, self.value

    def update(self, states, actions, returns):
        for state, action, G in zip(states, actions, returns):
            probs, value = self.forward(state)
            advantage = G - value  # How much better than expected?

            # Actor update: policy gradient scaled by advantage
            grad_logits = -probs.copy()
            grad_logits[action] += 1.0
            grad_logits *= advantage

            # Critic update: minimize (G - V(s))^2
            critic_grad = -2.0 * advantage  # d/dV (G - V)^2

            # Backprop actor head
            grad_W_actor = self.h1.reshape(-1, 1) @ grad_logits.reshape(1, -1)
            grad_h1_actor = grad_logits @ self.W_actor.T

            # Backprop critic head
            grad_W_critic = self.h1.reshape(-1, 1) * critic_grad
            grad_h1_critic = (self.W_critic * critic_grad).flatten()

            # Combined gradient through shared layer
            grad_h1 = grad_h1_actor + grad_h1_critic * 0.5  # scale critic
            grad_z1 = grad_h1 * (self.z1 > 0)

            # Gradient ascent for actor, descent for critic
            self.W_actor += self.lr * grad_W_actor
            self.W_critic -= self.lr * 0.5 * grad_W_critic
            self.W1 += self.lr * state.reshape(-1, 1) @ grad_z1.reshape(1, -1)
            self.b1 += self.lr * grad_z1

# Quick smoke test
np.random.seed(42)
ac = ActorCritic(state_dim=4, hidden_dim=16, n_actions=2, lr=0.005)

for ep in range(50):
    states, actions, rewards = [], [], []
    state = np.random.randn(4)
    for t in range(20):
        probs, value = ac.forward(state)
        action = np.random.choice(2, p=probs)
        reward = 1.0 if action == (1 if state[0] > 0 else 0) else -0.5
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = np.random.randn(4)

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    ac.update(states, actions, returns)

correct = 0
for _ in range(100):
    s = np.random.randn(4)
    probs, _ = ac.forward(s)
    a = np.argmax(probs)
    if a == (1 if s[0] > 0 else 0):
        correct += 1
print(f"Actor-Critic accuracy on simple task: {correct}%")
