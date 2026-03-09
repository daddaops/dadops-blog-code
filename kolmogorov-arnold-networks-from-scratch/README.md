# Kolmogorov-Arnold Networks from Scratch

Verified, runnable code from the [Kolmogorov-Arnold Networks from Scratch](https://dadops.dev/blog/kolmogorov-arnold-networks-from-scratch/) blog post.

## Scripts

- **ka_decomposition.py** — KA theorem: decomposing f(x,y) = sin(x)*exp(y) into univariate functions
- **bspline_basis.py** — Cox-de Boor B-spline basis functions, partition of unity, sin approximation
- **kan_layer.py** — KAN layer with B-spline edge activations and SiLU residual
- **training.py** — Training a [2,1] KAN with grid refinement (G=3 → G=8)
- **comparison.py** — KAN vs MLP: parameter efficiency and extrapolation comparison
- **fourier_kan.py** — FourierKAN variant using Fourier series as basis functions

## Run

```bash
pip install -r requirements.txt
python ka_decomposition.py
python bspline_basis.py
python kan_layer.py
python training.py
python comparison.py
python fourier_kan.py
```
