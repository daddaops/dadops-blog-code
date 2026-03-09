# Neural Tangent Kernels from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/neural-tangent-kernels-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `empirical_ntk.py` | Compute NTK via Jacobian J @ J^T |
| `ntk_stability.py` | NTK convergence with increasing width |
| `training_dynamics.py` | Actual GD vs NTK-predicted loss curves |
| `rich_vs_lazy.py` | Rich (feature learning) vs lazy (NTK) regimes |
| `analytic_ntk.py` | Infinite-width NTK via arccosine kernel |

## Usage

```bash
python empirical_ntk.py      # Empirical NTK computation
python ntk_stability.py      # Width convergence demo
python training_dynamics.py  # GD vs NTK prediction
python rich_vs_lazy.py       # Rich/lazy regime transition
python analytic_ntk.py       # Analytic NTK for ReLU nets
```

Dependencies: numpy.
