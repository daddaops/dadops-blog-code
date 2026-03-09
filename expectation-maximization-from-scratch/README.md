# Expectation-Maximization from Scratch

Verified, runnable code from the [Expectation-Maximization from Scratch](https://dadops.dev/blog/expectation-maximization-from-scratch/) blog post.

## Scripts

- **kmeans_elliptical.py** — K-means on elliptical clusters (motivation for EM)
- **gmm_data.py** — Generate synthetic data from a 3-component GMM
- **em_steps.py** — E-step and M-step implementation
- **em_full.py** — Complete EM algorithm with convergence tracking
- **kmeans_vs_em.py** — Direct accuracy comparison of k-means vs EM
- **bic_selection.py** — BIC-based model selection for optimal K

## Run

```bash
pip install -r requirements.txt
python kmeans_elliptical.py
python gmm_data.py
python em_steps.py
python em_full.py
python kmeans_vs_em.py
python bic_selection.py
```
