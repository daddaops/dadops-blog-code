# Causal Inference from Scratch — Code Blocks

Python code extracted from the [Causal Inference from Scratch](https://dadops.co/blog/causal-inference-from-scratch/) blog post.

## Scripts

| # | Script | Topic |
|---|--------|-------|
| 1 | `simpsons_paradox.py` | Simpson's Paradox with kidney stone data (Charig et al., 1986) |
| 2 | `potential_outcomes.py` | Rubin Causal Model, ATE, and selection bias decomposition |
| 3 | `causal_dag.py` | Causal DAG class with backdoor criterion checking |
| 4 | `propensity_matching.py` | Propensity Score Matching (PSM) from scratch |
| 5 | `diff_in_diff.py` | Difference-in-Differences (DiD) estimation |
| 6 | `instrumental_variables.py` | Instrumental Variables / Two-Stage Least Squares (2SLS) |

## Usage

```bash
pip install -r requirements.txt
python simpsons_paradox.py
python potential_outcomes.py
python causal_dag.py
python propensity_matching.py
python diff_in_diff.py
python instrumental_variables.py
```
