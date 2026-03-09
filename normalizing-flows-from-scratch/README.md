# Normalizing Flows from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/normalizing-flows-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `change_of_variables.py` | 1D change of variables with Newton inversion |
| `planar_flow.py` | Planar flow layers stacked on 2D Gaussian |
| `affine_coupling.py` | RealNVP affine coupling layer with invertibility test |
| `train_realnvp.py` | Train RealNVP on two-moons data |
| `invertible_conv.py` | Glow-style 1x1 invertible convolution with LU |
| `maf.py` | Masked Autoregressive Flow on two-moons data |

## Usage

```bash
python change_of_variables.py   # 1D density computation
python planar_flow.py           # Planar flow demo
python affine_coupling.py       # Coupling layer invertibility
python train_realnvp.py         # RealNVP training (slow, finite-diff)
python invertible_conv.py       # Invertible 1x1 convolution
python maf.py                   # MAF training (slow, finite-diff)
```

Dependencies: numpy, scipy.
