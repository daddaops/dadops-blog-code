# Bayesian Optimization from Scratch

Verified, runnable code from the [Bayesian Optimization from Scratch](https://dadops.co/blog/bayesian-optimization-from-scratch/) blog post.

## Scripts

| Script | Description |
|--------|-------------|
| `gp_bo_core.py` | Core GP and acquisition function components (shared module) |
| `gp_demo.py` | GP posterior — uncertainty near vs far from data |
| `acquisition_demo.py` | EI, UCB, PI acquisition function comparison |
| `bo_loop.py` | Complete BO loop on a 1D function |
| `hp_tuning.py` | Grid vs Random vs BO for hyperparameter tuning |
| `convergence_benchmark.py` | BO vs Random on Six-Hump Camel (10 trials) |
| `branin_test.py` | BO on the Branin test function |

## Quick Start

```bash
pip install -r requirements.txt
python gp_demo.py
python acquisition_demo.py
python bo_loop.py
python hp_tuning.py
python convergence_benchmark.py
python branin_test.py
```
