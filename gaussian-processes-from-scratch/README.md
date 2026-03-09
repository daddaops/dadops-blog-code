# Gaussian Processes from Scratch

Verified, runnable code from the [Gaussian Processes from Scratch](https://dadops.dev/blog/gaussian-processes-from-scratch/) blog post.

## Scripts

- **kernels.py** — RBF, Matérn 3/2, Matérn 5/2, and periodic kernel functions
- **gp_prior.py** — Sampling functions from a GP prior
- **gp_regression.py** — GP posterior regression with Cholesky solve
- **hyperparameter_opt.py** — Marginal likelihood optimization with L-BFGS-B
- **gp_classification.py** — GP binary classification with Laplace approximation
- **sparse_gp.py** — Sparse GP with Nyström approximation

## Run

```bash
pip install -r requirements.txt
python kernels.py
python gp_prior.py
python gp_regression.py
python hyperparameter_opt.py
python gp_classification.py
python sparse_gp.py
```
