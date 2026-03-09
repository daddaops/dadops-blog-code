# SVM from Scratch — Extracted Code

Python code extracted from the [DadOps blog post on Support Vector Machines](https://dadops.dev/blog/svm-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `01_hard_margin_svm.py` | Hard-margin SVM using sub-gradient descent on hinge loss with high C penalty. Trains on linearly separable 2D data and reports weights, bias, and support vector count. |
| `02_soft_margin_svm.py` | Soft-margin SVM demonstrating the bias-variance tradeoff by varying C (100, 1, 0.01) on noisy overlapping data. Shows how lower C widens the margin at the cost of accuracy. |
| `03_kernel_svm.py` | Kernel SVM with simplified SMO optimizer supporting linear, polynomial, and RBF kernels. Tests on concentric circles dataset to show that non-linear kernels solve non-linearly-separable problems. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_hard_margin_svm.py
python 02_soft_margin_svm.py
python 03_kernel_svm.py
```
