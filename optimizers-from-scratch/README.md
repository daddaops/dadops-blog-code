# Optimizers from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/optimizers-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `sgd_bowl.py` | Vanilla SGD on a simple bowl surface |
| `sgd_ravine.py` | Vanilla SGD on a ravine (ill-conditioned) surface |
| `momentum_ravine.py` | SGD with momentum on the ravine surface |
| `optimizer_comparison.py` | SGD vs Momentum vs RMSProp vs Adam on Beale-like surface |

## Usage

```bash
python sgd_bowl.py              # SGD on easy bowl
python sgd_ravine.py            # SGD struggles on ravine
python momentum_ravine.py       # Momentum accelerates along ravine
python optimizer_comparison.py  # Head-to-head comparison
```

Dependencies: numpy.
