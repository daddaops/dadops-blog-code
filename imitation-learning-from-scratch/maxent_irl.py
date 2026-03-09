import numpy as np
from feature_irl import make_features

def maxent_irl(expert_trajectories, feature_fn, n_features,
               grid_size=5, gamma=0.9, n_iters=100, lr=0.1):
    """Maximum Entropy IRL with soft value iteration."""
    expert_feat = np.zeros(n_features)
    for traj in expert_trajectories:
        for t, s in enumerate(traj):
            expert_feat += (gamma ** t) * feature_fn(s)
    expert_feat /= len(expert_trajectories)

    w = np.zeros(n_features)
    actions = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
    for iteration in range(n_iters):
        # Compute reward for each state
        R = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                R[r, c] = w @ feature_fn((r, c))

        # Soft value iteration (Bellman backup with log-sum-exp)
        V = np.zeros((grid_size, grid_size))
        for _ in range(30):
            V_new = np.full_like(V, -1e9)
            for r in range(grid_size):
                for c in range(grid_size):
                    vals = []
                    for dr, dc in actions:
                        nr = max(0, min(grid_size-1, r+dr))
                        nc = max(0, min(grid_size-1, c+dc))
                        vals.append(R[r,c] + gamma * V[nr, nc])
                    max_v = max(vals)
                    V_new[r,c] = max_v + np.log(
                        sum(np.exp(v - max_v) for v in vals))
            V = V_new

        # Softmax policy rollouts for feature expectations
        policy_feat = np.zeros(n_features)
        np.random.seed(iteration * 17 + 7)
        for _ in range(30):
            s = (0, 0)
            for t in range(15):
                r, c = s
                q_vals, next_states = [], []
                for dr, dc in actions:
                    nr = max(0, min(grid_size-1, r+dr))
                    nc = max(0, min(grid_size-1, c+dc))
                    q_vals.append(R[r,c] + gamma * V[nr, nc])
                    next_states.append((nr, nc))
                q_arr = np.array(q_vals)
                probs = np.exp(q_arr - q_arr.max())
                probs /= probs.sum()
                idx = np.random.choice(len(actions), p=probs)
                s = next_states[idx]
                policy_feat += (gamma ** t) * feature_fn(s)
        policy_feat /= 30
        w += lr * (expert_feat - policy_feat)

    # Recompute final reward grid
    R = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            R[r, c] = w @ feature_fn((r, c))
    return w, R

if __name__ == "__main__":
    feat_fn, n_feat = make_features()

    expert_trajs = [
        [(0,0),(1,1),(2,2),(3,3),(4,4)],
        [(0,0),(0,1),(1,2),(2,3),(3,4),(4,4)],
        [(0,0),(1,0),(2,1),(3,2),(4,3),(4,4)],
    ]

    np.random.seed(42)
    w_maxent, R_maxent = maxent_irl(expert_trajs, feat_fn, n_feat)
    print("MaxEnt reward weights:", np.round(w_maxent, 3))
    R_n = (R_maxent - R_maxent.min()) / (R_maxent.max() - R_maxent.min())
    R_d = R_n * 6 - 5
    print("\nMaxEnt recovered reward grid (normalized):")
    for r in range(5):
        print("  ", [f"{R_d[r,c]:+.2f}" for c in range(5)])
