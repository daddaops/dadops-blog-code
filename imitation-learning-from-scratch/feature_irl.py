import numpy as np

def feature_irl(expert_trajectories, feature_fn, n_features,
                grid_size=5, n_iters=100, lr=0.2):
    """Feature-matching IRL: learn reward weights from expert demos."""
    # Compute expert feature expectations
    expert_feat = np.zeros(n_features)
    for traj in expert_trajectories:
        for s in traj:
            expert_feat += feature_fn(s)
    expert_feat /= len(expert_trajectories)

    # Learn reward weights by matching feature expectations
    w = np.zeros(n_features)
    for iteration in range(n_iters):
        R = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                R[r, c] = w @ feature_fn((r, c))

        # Boltzmann state distribution: P(s) ∝ exp(R(s))
        flat_R = R.flatten()
        probs = np.exp(flat_R - flat_R.max())
        probs /= probs.sum()

        # Expected features under current reward
        policy_feat = np.zeros(n_features)
        for r in range(grid_size):
            for c in range(grid_size):
                policy_feat += probs[r * grid_size + c] * feature_fn((r, c))

        # Gradient: expert features - policy features
        w += lr * (expert_feat - policy_feat)

    # Recompute final reward grid
    R = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            R[r, c] = w @ feature_fn((r, c))
    return w, R

# Setup: 5x5 grid, expert navigates to (4,4)
def make_features(grid_size=5):
    """Features: row, column, proximity to goal."""
    goal = (grid_size-1, grid_size-1)
    max_d = np.sqrt(2) * (grid_size - 1)
    def feat(s):
        d = np.sqrt((s[0]-goal[0])**2 + (s[1]-goal[1])**2)
        return np.array([s[0]/(grid_size-1), s[1]/(grid_size-1), 1 - d/max_d])
    return feat, 3

if __name__ == "__main__":
    feat_fn, n_feat = make_features()

    # Expert trajectories: go from (0,0) to (4,4)
    expert_trajs = [
        [(0,0),(1,1),(2,2),(3,3),(4,4)],
        [(0,0),(0,1),(1,2),(2,3),(3,4),(4,4)],
        [(0,0),(1,0),(2,1),(3,2),(4,3),(4,4)],
    ]

    np.random.seed(42)
    w_learned, R_learned = feature_irl(expert_trajs, feat_fn, n_feat)
    print("Learned reward weights:", np.round(w_learned, 3))
    # Normalize for display
    R_n = (R_learned - R_learned.min()) / (R_learned.max() - R_learned.min())
    R_d = R_n * 6 - 5
    print("\nRecovered reward grid (normalized):")
    for r in range(5):
        print("  ", [f"{R_d[r,c]:+.2f}" for c in range(5)])
