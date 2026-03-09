# Online Learning from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/online-learning-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `weighted_majority.py` | Weighted Majority Algorithm for expert advice |
| `multiplicative_weights.py` | Multiplicative Weights Update (adversarial) |
| `online_gd.py` | Online Gradient Descent for linear regression |
| `ftrl_comparison.py` | FTRL: MW (entropic) vs OGD (L2) comparison |
| `online_to_batch.py` | Polyak-Ruppert averaging (online-to-batch) |
| `adagrad.py` | AdaGrad vs uniform OGD on sparse data |

## Usage

```bash
python weighted_majority.py      # Weighted Majority Algorithm
python multiplicative_weights.py # MW with shifting best expert
python online_gd.py              # Online GD convergence
python ftrl_comparison.py        # FTRL unification demo
python online_to_batch.py        # Polyak-Ruppert averaging
python adagrad.py                # AdaGrad on sparse features
```

Dependencies: numpy.
