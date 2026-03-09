# pca-from-scratch — Homework Status

## Current Phase: VERIFY complete

### Scripts verified (all 5 pass)
1. `pca_eigendecomposition.py` — 97.3%/2.7% variance, components differ in sign only ✓
2. `pca_svd.py` — SVD matches eigendecomposition: True ✓
3. `scree_plot_variance.py` — PC1=0.2711, 95%→10 components (was 0.2354, 11) ✓
4. `kernel_pca_rings.py` — KPCA range [-0.59, 0.48] (was [-5.82, 8.41]) ✓
5. `eigenfaces.py` — top eigenface 39.5%, 95%→51 components (was 18.3%, 32) ✓

### Blog updates
- Block 1: component directions updated (sign convention)
- Block 3: all variance ratios and component counts corrected
- Block 4: kernel PCA PC1 range corrected
- Block 5: eigenface stats and prose corrected
- Verified Code badge added
