# Monte Carlo Methods from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/monte-carlo-methods-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `estimate_pi.py` | Estimating π with random darts (1/√N convergence) |
| `mc_integration.py` | Monte Carlo integration in 1D, 2D, and 10D |
| `importance_sampling.py` | Importance sampling for rare event P(X > 4) |
| `rejection_sampling.py` | Rejection sampling from a bimodal mixture |
| `metropolis_hastings.py` | Metropolis-Hastings MCMC on bimodal target |
| `bayesian_regression.py` | MCMC for Bayesian linear regression with credible intervals |

## Usage

```bash
python estimate_pi.py         # π estimation at increasing N
python mc_integration.py      # MC integration in 1D, 2D, 10D
python importance_sampling.py # Rare event estimation
python rejection_sampling.py  # Rejection sampling demo
python metropolis_hastings.py # MCMC sampling demo
python bayesian_regression.py # Bayesian regression with MCMC
```

Dependencies: numpy, scipy.
