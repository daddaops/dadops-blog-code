# PCA from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/pca-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `pca_eigendecomposition.py` | PCA via covariance matrix eigendecomposition |
| `pca_svd.py` | PCA via SVD (numerically stable) |
| `scree_plot_variance.py` | Scree plot and cumulative variance analysis |
| `kernel_pca_rings.py` | Kernel PCA for nonlinear concentric rings |
| `eigenfaces.py` | Eigenface decomposition with synthetic data |

## Usage

```bash
python pca_eigendecomposition.py  # Core PCA algorithm
python pca_svd.py                 # SVD verification
python scree_plot_variance.py     # Dimensionality selection
python kernel_pca_rings.py        # Nonlinear PCA
python eigenfaces.py              # Eigenfaces demo
```

Dependencies: numpy.
