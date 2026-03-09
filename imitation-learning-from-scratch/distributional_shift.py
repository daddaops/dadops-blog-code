import numpy as np
from behavioral_cloning import expert_policy, collect_expert_demos, LinearBC

def rollout(policy_fn, horizon=30, noise=0.02):
    """Roll out a policy and measure error vs expert."""
    s = np.array([0.0, 0.0])
    errors = []
    for t in range(horizon):
        expert_a = expert_policy(s)
        agent_a = policy_fn(s)
        errors.append(np.linalg.norm(agent_a - expert_a))
        s = s + agent_a + np.random.randn(2) * noise
    return errors

if __name__ == "__main__":
    # Train BC first
    np.random.seed(42)
    states, actions = collect_expert_demos()
    bc = LinearBC(2, 2)
    bc.train(states, actions)

    # Compare BC errors at different horizons
    np.random.seed(99)
    bc_errors = rollout(bc.predict, horizon=50, noise=0.03)
    cumulative = np.cumsum(bc_errors)

    print("Cumulative BC error at different horizons:")
    for t in [10, 20, 30, 40, 50]:
        print(f"  T={t:2d}: cumulative error = {cumulative[t-1]:.3f}")
