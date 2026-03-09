"""Value iteration on a 5x5 gridworld."""
import numpy as np
from helpers import Gridworld

def value_iteration(env, gamma=0.99, theta=1e-6):
    """Find optimal value function by iterating the Bellman equation."""
    V = np.zeros((env.size, env.size))

    while True:
        delta = 0
        for r in range(env.size):
            for c in range(env.size):
                if (r, c) in env.walls or (r, c) == env.goal:
                    continue
                old_v = V[r, c]
                # Try each action, pick the best
                action_values = []
                for a in range(4):
                    dr, dc = env.actions[a]
                    nr = max(0, min(env.size - 1, r + dr))
                    nc = max(0, min(env.size - 1, c + dc))
                    if (nr, nc) in env.walls:
                        nr, nc = r, c  # bounce back
                    reward = 1.0 if (nr, nc) == env.goal else -0.01
                    action_values.append(reward + gamma * V[nr, nc])
                V[r, c] = max(action_values)
                delta = max(delta, abs(old_v - V[r, c]))
        if delta < theta:
            break

    # Extract policy: at each state, pick the best action
    action_names = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    policy = {}
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in env.walls or (r, c) == env.goal:
                continue
            best_a, best_v = 0, -float('inf')
            for a in range(4):
                dr, dc = env.actions[a]
                nr = max(0, min(env.size - 1, r + dr))
                nc = max(0, min(env.size - 1, c + dc))
                if (nr, nc) in env.walls:
                    nr, nc = r, c
                reward = 1.0 if (nr, nc) == env.goal else -0.01
                val = reward + gamma * V[nr, nc]
                if val > best_v:
                    best_v = val
                    best_a = a
            policy[(r, c)] = action_names[best_a]

    return V, policy

env = Gridworld()
V, policy = value_iteration(env)
print("Optimal values:")
print(np.round(V, 2))
print("\nOptimal policy:")
for r in range(env.size):
    row = ""
    for c in range(env.size):
        if (r, c) == env.goal:
            row += "  G  "
        elif (r, c) in env.walls:
            row += "  #  "
        else:
            row += f" {policy[(r,c)][:2]:>2}  "
    print(row)
