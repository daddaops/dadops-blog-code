# Mixture Density Networks from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/mixture-density-networks-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `inverse_problem.py` | MSE regression fails on multimodal inverse mapping |
| `gmm_em.py` | Fit a 2-component GMM via the EM algorithm |
| `mdn_architecture.py` | MDN class: forward pass producing mixture parameters |
| `mdn_loss.py` | Negative log-likelihood loss with log-sum-exp trick |
| `mdn_sampling.py` | Two-step sampling from a Gaussian mixture |
| `mdn_vs_mse.py` | NLL comparison: MDN vs single-Gaussian MSE model |

## Usage

```bash
python inverse_problem.py    # MSE prediction on inverse problem
python gmm_em.py             # EM algorithm on bimodal data
python mdn_architecture.py   # MDN forward pass demo
python mdn_loss.py           # NLL loss computation
python mdn_sampling.py       # Sample from mixture distribution
python mdn_vs_mse.py         # MDN vs MSE likelihood comparison
```

Dependencies: numpy, scikit-learn (for inverse_problem.py only).
