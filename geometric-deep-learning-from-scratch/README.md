# Geometric Deep Learning from Scratch

Verified, runnable code from the [Geometric Deep Learning from Scratch](https://dadops.dev/blog/geometric-deep-learning-from-scratch/) blog post.

## Scripts

- **invariance_equivariance.py** — Invariance (sum) vs equivariance (convolution) demo
- **group_theory.py** — C4 rotations and S_n permutations on graphs
- **fourier_equivariance.py** — Deriving convolution from translation equivariance via DFT
- **graph_equivariance.py** — Graph equivariance via polynomial in adjacency matrix
- **attention_equivariance.py** — Self-attention permutation equivariance proof
- **group_conv_c4.py** — C4 group convolution with rotated kernels

## Run

```bash
pip install -r requirements.txt
python invariance_equivariance.py
python group_theory.py
python fourier_equivariance.py
python graph_equivariance.py
python attention_equivariance.py
python group_conv_c4.py
```
