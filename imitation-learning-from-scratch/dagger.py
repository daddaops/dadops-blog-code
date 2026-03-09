import numpy as np
from behavioral_cloning import expert_policy, collect_expert_demos, LinearBC

def dagger(expert_fn, n_rounds=8, horizon=30, demos_per_round=5):
    """DAgger: iteratively aggregate expert data on learner states."""
    # Round 0: collect expert demos
    all_states, all_actions = collect_expert_demos(n_trajectories=demos_per_round)
    policy = LinearBC(2, 2)
    round_errors = []

    for r in range(n_rounds):
        # Train on all aggregated data
        policy.train(all_states, all_actions)

        # Roll out current policy, query expert at visited states
        new_states, new_actions = [], []
        total_err = 0
        for _ in range(demos_per_round):
            s = np.array([0.0, np.random.randn() * 0.1])
            for t in range(horizon):
                agent_a = policy.predict(s)
                expert_a = expert_fn(s)  # query expert at learner's state
                new_states.append(s.copy())
                new_actions.append(expert_a)
                total_err += np.linalg.norm(agent_a - expert_a)
                s = s + agent_a + np.random.randn(2) * 0.02
        avg_err = total_err / (demos_per_round * horizon)
        round_errors.append(avg_err)

        # Aggregate new data
        all_states = np.vstack([all_states, np.array(new_states)])
        all_actions = np.vstack([all_actions, np.array(new_actions)])

    return policy, round_errors

if __name__ == "__main__":
    np.random.seed(42)
    _, dagger_errors = dagger(expert_policy)
    print("DAgger average error per round:")
    for r, e in enumerate(dagger_errors):
        print(f"  Round {r+1}: {e:.4f}")
