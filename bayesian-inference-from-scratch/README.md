# Bayesian Inference from Scratch

Verified, runnable code from the [Bayesian Inference from Scratch](https://dadops.co/blog/bayesian-inference-from-scratch/) blog post.

## Scripts

| Script | Description |
|--------|-------------|
| `bayes_theorem.py` | Bayes' theorem (medical test) + grid-based coin updating |
| `conjugate_priors.py` | Beta-Binomial and Normal-Normal conjugate updates |
| `map_vs_mle.py` | MAP vs MLE polynomial fitting with L2 regularization |
| `mcmc.py` | Metropolis-Hastings MCMC for Gaussian parameter inference |
| `bayesian_linreg.py` | Bayesian linear regression with predictive uncertainty |
| `mc_dropout.py` | MC Dropout uncertainty estimation with a neural net |

## Quick Start

```bash
pip install -r requirements.txt
python bayes_theorem.py
python conjugate_priors.py
python map_vs_mle.py
python mcmc.py
python bayesian_linreg.py
python mc_dropout.py
```
