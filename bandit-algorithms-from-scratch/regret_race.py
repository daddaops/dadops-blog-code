"""Regret race — compare all four bandit algorithms on the same instance.

Runs greedy, epsilon-greedy, UCB1, and Thompson Sampling for 10,000 steps
and compares cumulative regret.
"""
import numpy as np
from bandit_testbed import bernoulli_bandit, run_greedy
from epsilon_greedy import run_epsilon_greedy
from ucb1 import run_ucb1
from thompson_sampling import run_thompson

if __name__ == "__main__":
    probs = bernoulli_bandit(K=5, seed=42)
    T = 10000

    # Run all four algorithms on the same bandit
    _, _, greedy_regret, _ = run_greedy(probs, T=T, seed=0)
    reg_eps, _ = run_epsilon_greedy(probs, T=T, epsilon=0.1, seed=0)
    reg_ucb, _ = run_ucb1(probs, T=T, seed=0)
    reg_ts, _, _, _ = run_thompson(probs, T=T, seed=0)

    print(f"Cumulative regret after {T:,} steps:")
    print(f"  Greedy:           {greedy_regret:.1f}")
    print(f"  Epsilon-greedy:   {reg_eps[-1]:.1f}")
    print(f"  UCB1:             {reg_ucb[-1]:.1f}")
    print(f"  Thompson:         {reg_ts[-1]:.1f}")
    print(f"\nAsymptotic rates:")
    print(f"  Greedy:           O(T)          -- linear, catastrophic")
    print(f"  Eps-greedy:       O(eps * T)    -- linear, proportional to eps")
    print(f"  UCB1:             O(sqrt(KT logT)) -- sublinear, logarithmic")
    print(f"  Thompson:         Lai-Robbins   -- optimal, can't do better")
