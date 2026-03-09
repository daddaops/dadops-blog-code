# Naive Bayes from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/naive-bayes-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `gaussian_nb.py` | Gaussian Naive Bayes classifier with 2D example |
| `multinomial_nb.py` | Multinomial NB for text classification (spam filter) |
| `bernoulli_nb.py` | Bernoulli NB with word presence/absence modeling |
| `correlated_features.py` | Independence paradox: NB with correlated features |
| `scenario_comparison.py` | NB accuracy across 4 practical scenarios |

## Usage

```bash
python gaussian_nb.py         # Gaussian NB on 2D clusters
python multinomial_nb.py      # Spam classifier with word counts
python bernoulli_nb.py        # Bernoulli NB spam classifier
python correlated_features.py # NB with correlated features demo
python scenario_comparison.py # Multi-scenario accuracy comparison
```

Dependencies: numpy.
