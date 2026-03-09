# Neural ODEs from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/neural-odes-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `ode_solvers.py` | Euler and RK4 solvers for 2D spiral ODE |
| `neural_ode.py` | Minimal Neural ODE with RK4 integration |
| `adjoint_method.py` | Adjoint method for O(1) memory gradients |
| `cnf.py` | Continuous Normalizing Flows with Hutchinson trace |
| `augmented_ode.py` | Augmented Neural ODE for topology breaking |
| `latent_ode.py` | Latent ODE for irregular time series |

## Usage

```bash
python ode_solvers.py      # Euler vs RK4 comparison
python neural_ode.py       # Neural ODE forward pass
python adjoint_method.py   # Adjoint gradient computation
python cnf.py              # Continuous normalizing flows
python augmented_ode.py    # Augmented Neural ODE
python latent_ode.py       # Latent ODE for time series
```

Dependencies: numpy.
