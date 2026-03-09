"""Q-learning on a 5x5 gridworld."""
import numpy as np
from helpers import Gridworld

def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Train a Q-table on the gridworld using Q-learning."""
    # Q-table: state (r, c) x action (4 directions)
    Q = np.zeros((env.size, env.size, 4))

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            r, c = state
            # Epsilon-greedy: explore with prob epsilon, exploit otherwise
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[r, c])

            next_state, reward, done = env.step(action)
            nr, nc = next_state

            # The Q-learning update
            td_target = reward + gamma * np.max(Q[nr, nc]) * (1 - done)
            td_error = td_target - Q[r, c, action]
            Q[r, c, action] += alpha * td_error

            state = next_state

    return Q

env = Gridworld()
Q = q_learning(env, episodes=1000)
# Extract the learned policy
policy = np.argmax(Q, axis=2)  # Best action at each state
# action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}

action_names = {0: 'up', 1: 'dn', 2: 'lt', 3: 'rt'}
print("Learned Q-learning policy:")
for r in range(env.size):
    row = ""
    for c in range(env.size):
        if (r, c) == env.goal:
            row += "  G  "
        elif (r, c) in env.walls:
            row += "  #  "
        else:
            row += f" {action_names[policy[r, c]]:>2}  "
    print(row)
