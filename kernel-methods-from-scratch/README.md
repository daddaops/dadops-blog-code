# Kernel Methods from Scratch

Verified, runnable code from the [Kernel Methods from Scratch](https://dadops.dev/blog/kernel-methods-from-scratch/) blog post.

## Scripts

- **feature_map.py** — Explicit polynomial feature map and dimensionality explosion
- **kernel_trick.py** — Kernel trick verification and timing comparison
- **mercer.py** — Mercer's theorem: PSD Gram matrix test for valid kernels
- **kernel_zoo.py** — Six kernels (linear, polynomial, RBF, Laplacian, Matern, cosine) heatmap ranges
- **kernel_composition.py** — Kernel composition (sum) and kernel alignment
- **kernel_algorithms.py** — Kernel ridge regression and kernel PCA

## Run

```bash
pip install -r requirements.txt
python feature_map.py
python kernel_trick.py
python mercer.py
python kernel_zoo.py
python kernel_composition.py
python kernel_algorithms.py
```
