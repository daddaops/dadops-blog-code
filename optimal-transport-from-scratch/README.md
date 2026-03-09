# Optimal Transport from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/optimal-transport-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `monge_hungarian.py` | Monge's optimal transport via Hungarian algorithm |
| `kantorovich_lp.py` | Kantorovich LP formulation with mass splitting |
| `wasserstein_vs_kl.py` | W1 vs KL vs JS divergence comparison |
| `sinkhorn_algorithm.py` | Sinkhorn's entropy-regularized OT |
| `wasserstein_barycenter.py` | Wasserstein barycenter of 1D histograms |
| `wgan_critic_loss.py` | WGAN-GP critic loss with gradient penalty |

## Usage

```bash
python monge_hungarian.py       # Optimal 1-to-1 assignment
python kantorovich_lp.py        # LP with mass splitting
python wasserstein_vs_kl.py     # Distance metric comparison
python sinkhorn_algorithm.py    # Entropy-regularized OT
python wasserstein_barycenter.py # Barycenter computation
python wgan_critic_loss.py      # WGAN critic objective
```

Dependencies: numpy, scipy.
