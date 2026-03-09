# Imitation Learning from Scratch

Verified, runnable code from the [Imitation Learning from Scratch](https://dadops.dev/blog/imitation-learning-from-scratch/) blog post.

## Scripts

- **behavioral_cloning.py** — Behavioral cloning with linear policy on sinusoidal path
- **distributional_shift.py** — Demonstrates cumulative error growth (O(T²) drift)
- **dagger.py** — DAgger: Dataset Aggregation for online expert querying
- **feature_irl.py** — Feature-matching Inverse RL on 5x5 grid
- **maxent_irl.py** — Maximum Entropy IRL with soft value iteration

## Run

```bash
pip install -r requirements.txt
python behavioral_cloning.py
python distributional_shift.py
python dagger.py
python feature_irl.py
python maxent_irl.py
```
