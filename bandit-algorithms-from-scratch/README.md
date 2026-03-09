# Bandit Algorithms from Scratch

Verified, runnable code from the [Bandit Algorithms from Scratch](https://dadops.io/blog/bandit-algorithms-from-scratch/) blog post.

## Scripts

- `bandit_testbed.py` — Bernoulli bandit environment + pure greedy agent
- `epsilon_greedy.py` — Epsilon-greedy with fixed and decaying epsilon
- `ucb1.py` — UCB1 (Upper Confidence Bound) algorithm
- `thompson_sampling.py` — Thompson Sampling with Beta posteriors
- `regret_race.py` — Head-to-head regret comparison of all four algorithms
- `linucb.py` — LinUCB contextual bandit for personalized recommendation

## Run

```bash
pip install -r requirements.txt
python3 bandit_testbed.py
python3 epsilon_greedy.py
python3 ucb1.py
python3 thompson_sampling.py
python3 regret_race.py
python3 linucb.py
```
